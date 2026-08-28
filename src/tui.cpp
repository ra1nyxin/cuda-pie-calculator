#include "tui.hpp"

#include "resources.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>
#endif

namespace pie {
namespace {

enum class Key {
    None,
    Start,
    Pause,
    Cancel,
    Quit,
    ToggleMode,
    EditTarget,
    ConfirmInput,
    DismissInput,
    Backspace,
    DigitInput,
    ScrollUp,
    ScrollDown,
    PreviousDevice,
    NextDevice,
};

struct KeyEvent {
    Key key = Key::None;
    char character = '\0';
};

struct TargetEditor {
    bool open = false;
    CalculationMode mode = CalculationMode::ExactDigits;
    std::string input;
    std::string error;
};

struct TerminalSize {
    unsigned rows = 30;
    unsigned columns = 100;
};

class TerminalSession {
public:
    TerminalSession() {
#if defined(_WIN32)
        input_ = GetStdHandle(STD_INPUT_HANDLE);
        output_ = GetStdHandle(STD_OUTPUT_HANDLE);
        if (input_ == INVALID_HANDLE_VALUE || output_ == INVALID_HANDLE_VALUE ||
            GetConsoleMode(input_, &oldInputMode_) == 0 || GetConsoleMode(output_, &oldOutputMode_) == 0) {
            throw std::runtime_error("运行 TUI 需要 Windows 控制台。");
        }
        const DWORD newInputMode = (oldInputMode_ | ENABLE_WINDOW_INPUT) & ~(ENABLE_ECHO_INPUT | ENABLE_LINE_INPUT);
        const DWORD newOutputMode = oldOutputMode_ | ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (SetConsoleMode(input_, newInputMode) == 0 || SetConsoleMode(output_, newOutputMode) == 0) {
            throw std::runtime_error("无法启用 Windows 10/11 的虚拟终端支持。");
        }
        SetConsoleOutputCP(CP_UTF8);
#else
        if (isatty(STDIN_FILENO) == 0 || isatty(STDOUT_FILENO) == 0) {
            throw std::runtime_error("运行 TUI 需要交互式终端。");
        }
        if (tcgetattr(STDIN_FILENO, &oldAttributes_) != 0) {
            throw std::runtime_error("无法读取终端设置。");
        }
        termios raw = oldAttributes_;
        raw.c_lflag &= static_cast<tcflag_t>(~(ICANON | ECHO));
        raw.c_cc[VMIN] = 0;
        raw.c_cc[VTIME] = 0;
        if (tcsetattr(STDIN_FILENO, TCSANOW, &raw) != 0) {
            throw std::runtime_error("无法启用终端原始输入模式。");
        }
        oldInputFlags_ = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (oldInputFlags_ >= 0) {
            fcntl(STDIN_FILENO, F_SETFL, oldInputFlags_ | O_NONBLOCK);
        }
#endif
        active_ = true;
        std::cout << "\x1b[?1049h\x1b[2J\x1b[H\x1b[?25l" << std::flush;
    }

    ~TerminalSession() {
        if (!active_) {
            return;
        }
        std::cout << "\x1b[?25h\x1b[0m\x1b[?1049l" << std::flush;
#if defined(_WIN32)
        SetConsoleMode(input_, oldInputMode_);
        SetConsoleMode(output_, oldOutputMode_);
#else
        tcsetattr(STDIN_FILENO, TCSANOW, &oldAttributes_);
        if (oldInputFlags_ >= 0) {
            fcntl(STDIN_FILENO, F_SETFL, oldInputFlags_);
        }
#endif
    }

    TerminalSize size() const {
        TerminalSize result;
#if defined(_WIN32)
        CONSOLE_SCREEN_BUFFER_INFO info{};
        if (GetConsoleScreenBufferInfo(output_, &info) != 0) {
            result.columns = static_cast<unsigned>(info.srWindow.Right - info.srWindow.Left + 1);
            result.rows = static_cast<unsigned>(info.srWindow.Bottom - info.srWindow.Top + 1);
        }
#else
        winsize window{};
        if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &window) == 0) {
            if (window.ws_col > 0) {
                result.columns = window.ws_col;
            }
            if (window.ws_row > 0) {
                result.rows = window.ws_row;
            }
        }
#endif
        result.columns = std::max(result.columns, 40U);
        result.rows = std::max(result.rows, 12U);
        return result;
    }

    KeyEvent pollKey() {
#if defined(_WIN32)
        INPUT_RECORD record{};
        DWORD pending = 0;
        while (GetNumberOfConsoleInputEvents(input_, &pending) != 0 && pending > 0) {
            DWORD received = 0;
            if (ReadConsoleInputW(input_, &record, 1, &received) == 0 || received == 0) {
                break;
            }
            if (record.EventType != KEY_EVENT || record.Event.KeyEvent.bKeyDown == 0) {
                continue;
            }
            const WORD virtualKey = record.Event.KeyEvent.wVirtualKeyCode;
            if (virtualKey == VK_UP) return {Key::ScrollUp};
            if (virtualKey == VK_DOWN) return {Key::ScrollDown};
            const wchar_t character = record.Event.KeyEvent.uChar.UnicodeChar;
            return mapCharacter(static_cast<char>(character));
        }
#else
        if (!pendingEvents_.empty()) {
            const KeyEvent event = pendingEvents_.front();
            pendingEvents_.pop_front();
            return event;
        }
        char input[32]{};
        const ssize_t count = read(STDIN_FILENO, input, sizeof(input));
        if (count <= 0) {
            return {};
        }
        for (ssize_t index = 0; index < count; ++index) {
            if (input[index] == '\x1b' && index + 2 < count && input[index + 1] == '[') {
                if (input[index + 2] == 'A') {
                    pendingEvents_.push_back({Key::ScrollUp});
                    index += 2;
                    continue;
                }
                if (input[index + 2] == 'B') {
                    pendingEvents_.push_back({Key::ScrollDown});
                    index += 2;
                    continue;
                }
            }
            const KeyEvent event = mapCharacter(input[index]);
            if (event.key != Key::None) {
                pendingEvents_.push_back(event);
            }
        }
        if (!pendingEvents_.empty()) {
            const KeyEvent event = pendingEvents_.front();
            pendingEvents_.pop_front();
            return event;
        }
#endif
        return {};
    }

private:
    static KeyEvent mapCharacter(char character) {
        switch (character) {
            case 's': case 'S': return {Key::Start};
            case 'p': case 'P': return {Key::Pause};
            case 'c': case 'C': return {Key::Cancel};
            case 'm': case 'M': return {Key::ToggleMode};
            case 'q': case 'Q': return {Key::Quit};
            case 'e': case 'E': return {Key::EditTarget};
            case '\r': case '\n': return {Key::ConfirmInput};
            case '\x1b': return {Key::DismissInput};
            case '\b': case '\x7f': return {Key::Backspace};
            case 'k': case 'K': return {Key::ScrollUp};
            case 'j': case 'J': return {Key::ScrollDown};
            case '[': return {Key::PreviousDevice};
            case ']': return {Key::NextDevice};
            default:
                if (character >= '0' && character <= '9') {
                    return {Key::DigitInput, character};
                }
                return {};
        }
    }

    bool active_ = false;
#if defined(_WIN32)
    HANDLE input_ = INVALID_HANDLE_VALUE;
    HANDLE output_ = INVALID_HANDLE_VALUE;
    DWORD oldInputMode_ = 0;
    DWORD oldOutputMode_ = 0;
#else
    termios oldAttributes_{};
    int oldInputFlags_ = -1;
    std::deque<KeyEvent> pendingEvents_;
#endif
};

struct Utf8Unit {
    std::size_t bytes = 1;
    char32_t codePoint = 0;
};

bool isContinuationByte(unsigned char value) {
    return (value & 0xc0U) == 0x80U;
}

Utf8Unit readUtf8Unit(std::string_view text, std::size_t offset) {
    const unsigned char first = static_cast<unsigned char>(text[offset]);
    if (first < 0x80U) {
        return {1, first};
    }
    const auto continuation = [&](std::size_t index) {
        return index < text.size() && isContinuationByte(static_cast<unsigned char>(text[index]));
    };
    if ((first & 0xe0U) == 0xc0U && continuation(offset + 1)) {
        return {2, static_cast<char32_t>(((first & 0x1fU) << 6U) | (static_cast<unsigned char>(text[offset + 1]) & 0x3fU))};
    }
    if ((first & 0xf0U) == 0xe0U && continuation(offset + 1) && continuation(offset + 2)) {
        return {3, static_cast<char32_t>(((first & 0x0fU) << 12U) |
                                          ((static_cast<unsigned char>(text[offset + 1]) & 0x3fU) << 6U) |
                                          (static_cast<unsigned char>(text[offset + 2]) & 0x3fU))};
    }
    if ((first & 0xf8U) == 0xf0U && continuation(offset + 1) && continuation(offset + 2) && continuation(offset + 3)) {
        return {4, static_cast<char32_t>(((first & 0x07U) << 18U) |
                                          ((static_cast<unsigned char>(text[offset + 1]) & 0x3fU) << 12U) |
                                          ((static_cast<unsigned char>(text[offset + 2]) & 0x3fU) << 6U) |
                                          (static_cast<unsigned char>(text[offset + 3]) & 0x3fU))};
    }
    return {1, first};
}

std::size_t displayCellWidth(char32_t codePoint) {
    if (codePoint == 0 || (codePoint >= 0x300U && codePoint <= 0x36fU)) {
        return 0;
    }
    if ((codePoint >= 0x1100U && codePoint <= 0x115fU) ||
        (codePoint >= 0x2e80U && codePoint <= 0xa4cfU) ||
        (codePoint >= 0xac00U && codePoint <= 0xd7a3U) ||
        (codePoint >= 0xf900U && codePoint <= 0xfaffU) ||
        (codePoint >= 0xfe10U && codePoint <= 0xfe6fU) ||
        (codePoint >= 0xff00U && codePoint <= 0xff60U) ||
        (codePoint >= 0xffe0U && codePoint <= 0xffe6U) ||
        (codePoint >= 0x1f300U && codePoint <= 0x1faffU) ||
        (codePoint >= 0x20000U && codePoint <= 0x3fffdU)) {
        return 2;
    }
    return 1;
}

std::size_t displayWidth(std::string_view text) {
    std::size_t width = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const Utf8Unit unit = readUtf8Unit(text, offset);
        width += displayCellWidth(unit.codePoint);
        offset += unit.bytes;
    }
    return width;
}

std::string crop(std::string_view text, std::size_t width) {
    std::string visible;
    std::size_t used = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const Utf8Unit unit = readUtf8Unit(text, offset);
        const std::size_t unitWidth = displayCellWidth(unit.codePoint);
        if (used + unitWidth > width) {
            break;
        }
        visible.append(text.substr(offset, unit.bytes));
        used += unitWidth;
        offset += unit.bytes;
    }
    return visible;
}

void appendLine(std::ostringstream& output, const std::string& text, std::size_t width, bool advance = true) {
    const std::string visible = crop(text, width);
    output << visible;
    const std::size_t visibleWidth = displayWidth(visible);
    if (visibleWidth < width) {
        output << std::string(width - visibleWidth, ' ');
    }
    output << "\x1b[K";
    if (advance) {
        output << "\r\n";
    }
}

std::string progressBar(unsigned completed, unsigned total, std::size_t availableWidth) {
    const std::size_t barWidth = std::clamp<std::size_t>(availableWidth, 8, 32);
    const double ratio = total == 0 ? 0.0 : std::min(1.0, static_cast<double>(completed) / static_cast<double>(total));
    const std::size_t filled = static_cast<std::size_t>(ratio * static_cast<double>(barWidth));
    std::ostringstream output;
    output << '[' << std::string(filled, '#') << std::string(barWidth - filled, '-') << "] "
           << std::setw(3) << static_cast<unsigned>(ratio * 100.0) << '%';
    return output.str();
}

bool jobIsActive(JobState state) {
    return state == JobState::Preparing || state == JobState::Running || state == JobState::Paused || state == JobState::Cancelling;
}

std::vector<std::string> resultLines(const std::string& result, std::size_t width, CalculationMode selectedMode) {
    std::vector<std::string> lines;
    if (result.empty()) {
        lines.emplace_back(selectedMode == CalculationMode::MonteCarlo
                               ? "等待当前 CUDA GPU 的蒙特卡洛计算结果。程序没有 CPU 计算路径。"
                               : "等待当前 CUDA GPU 的精确计算结果。程序没有 CPU 计算路径。");
        return lines;
    }

    std::size_t segmentStart = 0;
    while (segmentStart <= result.size()) {
        const std::size_t lineBreak = result.find('\n', segmentStart);
        const std::size_t segmentEnd = lineBreak == std::string::npos ? result.size() : lineBreak;
        const std::string segment = result.substr(segmentStart, segmentEnd - segmentStart);
        if (segment.empty()) {
            lines.emplace_back();
        } else {
            std::string line;
            std::size_t lineWidth = 0;
            for (std::size_t offset = 0; offset < segment.size();) {
                const Utf8Unit unit = readUtf8Unit(segment, offset);
                const std::size_t unitWidth = displayCellWidth(unit.codePoint);
                if (!line.empty() && lineWidth + unitWidth > width) {
                    lines.push_back(std::move(line));
                    line.clear();
                    lineWidth = 0;
                }
                line.append(segment, offset, unit.bytes);
                lineWidth += unitWidth;
                offset += unit.bytes;
            }
            if (!line.empty()) {
                lines.push_back(std::move(line));
            }
        }
        if (lineBreak == std::string::npos) {
            break;
        }
        segmentStart = lineBreak + 1;
    }
    return lines;
}

std::string cpuLine(const ResourceStats& stats) {
    std::ostringstream output;
    output << "CPU：";
    if (stats.cpuAvailable) {
        output << std::fixed << std::setprecision(1) << stats.cpuPercent << '%';
    } else {
        output << "采样中";
    }
    output << "   内存：" << formatBytes(stats.memoryUsedBytes) << " / " << formatBytes(stats.memoryTotalBytes);
    return output.str();
}

std::string gpuLoadLine(const ResourceStats& stats) {
    if (!stats.gpuDetailsAvailable) {
        return "当前 GPU 负载：NVML 指标不可用";
    }
    std::ostringstream output;
    output << "当前 GPU 负载：" << stats.gpuUtilizationPercent << "%   显存："
           << formatBytes(stats.gpuMemoryUsedBytes) << " / " << formatBytes(stats.gpuMemoryTotalBytes)
           << "   温度：" << stats.gpuTemperatureCelsius << " C";
    return output.str();
}

std::string formatSampleCount(std::uint64_t samples) {
    std::ostringstream output;
    if (samples >= 100'000'000ULL) {
        output << std::fixed << std::setprecision(1) << static_cast<double>(samples) / 100'000'000.0 << " 亿";
    } else if (samples >= 10'000ULL) {
        output << std::fixed << std::setprecision(1) << static_cast<double>(samples) / 10'000.0 << " 万";
    } else {
        output << samples;
    }
    return output.str();
}

std::string formatSampleRate(double samplesPerSecond) {
    if (samplesPerSecond <= 0.0) {
        return "暂无";
    }
    return formatSampleCount(static_cast<std::uint64_t>(samplesPerSecond)) + "样本/秒";
}

std::string formatInteger(std::uint64_t value) {
    std::string text = std::to_string(value);
    for (std::size_t position = text.size(); position > 3; position -= 3) {
        text.insert(position - 3, 1, ',');
    }
    return text;
}

std::string editorTitle(CalculationMode mode) {
    return mode == CalculationMode::MonteCarlo ? "设置蒙特卡洛样本目标" : "设置精确目标位数";
}

std::string editorRange(CalculationMode mode) {
    if (mode == CalculationMode::MonteCarlo) {
        return "可输入范围：" + formatInteger(kMinimumMonteCarloSamples) + " 到 " + formatInteger(kMaximumMonteCarloSamples) + " 样本";
    }
    return "可输入范围：" + formatInteger(kMinimumDigits) + " 到 " + formatInteger(kMaximumDigits) + " 位";
}

std::string editorCurrentValue(CalculationMode mode, unsigned digits, std::uint64_t samples) {
    if (mode == CalculationMode::MonteCarlo) {
        return "当前目标：" + formatInteger(samples) + " 样本";
    }
    return "当前目标：" + formatInteger(digits) + " 位";
}

bool commitTargetEditor(TargetEditor& editor, unsigned& digits, std::uint64_t& samples) {
    if (editor.input.empty()) {
        editor.error = "请输入一个十进制整数。";
        return false;
    }

    std::uint64_t value = 0;
    const auto [end, error] = std::from_chars(editor.input.data(), editor.input.data() + editor.input.size(), value);
    if (error != std::errc{} || end != editor.input.data() + editor.input.size()) {
        editor.error = "输入值无效，请只输入十进制数字。";
        return false;
    }

    if (editor.mode == CalculationMode::MonteCarlo) {
        if (value < kMinimumMonteCarloSamples || value > kMaximumMonteCarloSamples) {
            editor.error = "样本目标超出允许范围。";
            return false;
        }
        samples = value;
    } else {
        if (value < kMinimumDigits || value > kMaximumDigits) {
            editor.error = "精确位数超出允许范围。";
            return false;
        }
        digits = static_cast<unsigned>(value);
    }
    editor.open = false;
    return true;
}

void renderEditorLine(
    std::ostringstream& output,
    std::size_t row,
    std::size_t column,
    std::size_t innerWidth,
    const std::string& text) {
    const std::string visible = crop(text, innerWidth);
    output << "\x1b[" << row << ';' << column << "H\x1b[97;48;5;238m|" << visible;
    const std::size_t visibleWidth = displayWidth(visible);
    if (visibleWidth < innerWidth) {
        output << std::string(innerWidth - visibleWidth, ' ');
    }
    output << "|\x1b[0m";
}

void renderTargetEditor(
    std::ostringstream& output,
    const TerminalSize& terminal,
    const TargetEditor& editor,
    unsigned selectedDigits,
    std::uint64_t selectedSamples) {
    const std::size_t modalWidth = std::min<std::size_t>(74, terminal.columns - 4);
    const std::size_t innerWidth = modalWidth - 2;
    const std::size_t modalHeight = 8;
    const std::size_t top = std::max<std::size_t>(1, (terminal.rows - modalHeight) / 2 + 1);
    const std::size_t left = std::max<std::size_t>(1, (terminal.columns - modalWidth) / 2 + 1);
    const std::string border = "+" + std::string(innerWidth, '-') + "+";

    output << "\x1b[" << top << ';' << left << "H\x1b[1;96;48;5;238m" << border << "\x1b[0m";
    renderEditorLine(output, top + 1, left, innerWidth, editorTitle(editor.mode));
    renderEditorLine(output, top + 2, left, innerWidth, editorRange(editor.mode));
    renderEditorLine(output, top + 3, left, innerWidth, editorCurrentValue(editor.mode, selectedDigits, selectedSamples));
    renderEditorLine(output, top + 4, left, innerWidth, "输入：" + editor.input + "_");
    renderEditorLine(output, top + 5, left, innerWidth, editor.error.empty() ? "仅可输入十进制数字。" : editor.error);
    renderEditorLine(output, top + 6, left, innerWidth, "Enter 确认   Esc 取消");
    output << "\x1b[" << (top + 7) << ';' << left << "H\x1b[1;96;48;5;238m" << border << "\x1b[0m";
}

std::string formatDigitRate(unsigned digits, double milliseconds) {
    if (milliseconds <= 0.0) {
        return "暂无";
    }
    const double digitsPerSecond = static_cast<double>(digits) * 1000.0 / milliseconds;
    std::ostringstream output;
    if (digitsPerSecond >= 10'000.0) {
        output << std::fixed << std::setprecision(1) << digitsPerSecond / 10'000.0 << "万位/秒";
    } else {
        output << std::fixed << std::setprecision(1) << digitsPerSecond << "位/秒";
    }
    return output.str();
}

std::string formatTheoreticalBandwidth(const GpuInfo& gpu) {
    if (gpu.memoryClockKHz <= 0 || gpu.memoryBusWidthBits <= 0) {
        return "未知";
    }
    const double gigabytesPerSecond = static_cast<double>(gpu.memoryClockKHz) * 1000.0 *
                                     static_cast<double>(gpu.memoryBusWidthBits) / 8.0 * 2.0 / 1'000'000'000.0;
    std::ostringstream output;
    output << std::fixed << std::setprecision(1) << gigabytesPerSecond << " GB/s";
    return output.str();
}

std::string deviceTaskLine(const JobSnapshot& job) {
    std::ostringstream output;
    output << "    任务：" << stateLabel(job.state);
    if (job.state == JobState::GpuUnavailable) {
        return output.str();
    }
    if (job.state == JobState::Idle && job.result.empty()) {
        output << " | 尚未计算";
        return output.str();
    }

    if (job.mode == CalculationMode::MonteCarlo) {
        if (job.sampleTarget > 0) {
            output << " | 蒙特卡洛 " << formatSampleCount(job.sampleTarget) << "样本";
        }
        if (job.samplesPerSecond > 0.0) {
            output << " | 吞吐：" << formatSampleRate(job.samplesPerSecond);
        }
        if (job.samplesCompleted > 0) {
            output << " | Pi约 " << std::fixed << std::setprecision(8) << job.monteCarloEstimate;
        }
    } else {
        output << " | 精确 " << job.requestedDigits << " 位";
        if (job.gpuMilliseconds > 0.0) {
            output << " | 用时：" << std::fixed << std::setprecision(2) << job.gpuMilliseconds << " ms"
                   << " | 吞吐：" << formatDigitRate(job.requestedDigits, job.gpuMilliseconds);
        }
        if (!job.result.empty()) {
            output << " | Pi=" << crop(job.result, 18);
        }
    }
    return output.str();
}

std::size_t deviceWindowStart(std::size_t deviceCount, std::size_t selectedSlot, std::size_t visibleSlots) {
    if (deviceCount <= visibleSlots) {
        return 0;
    }
    const std::size_t before = visibleSlots / 2;
    std::size_t first = selectedSlot > before ? selectedSlot - before : 0;
    if (first + visibleSlots > deviceCount) {
        first = deviceCount - visibleSlots;
    }
    return first;
}

void render(
    const TerminalSize& terminal,
    const std::vector<DeviceSnapshot>& devices,
    std::size_t selectedDeviceSlot,
    const JobSnapshot& job,
    const ResourceStats& resources,
    CalculationMode selectedMode,
    unsigned selectedDigits,
    std::uint64_t selectedSamples,
    const TargetEditor& editor,
    std::size_t& scrollOffset) {
    const std::size_t width = terminal.columns;
    const std::size_t bottomRows = 5;
    const std::size_t baseHeaderRows = 3;
    const std::size_t availableDeviceRows = terminal.rows > bottomRows + baseHeaderRows + 1
                                                ? terminal.rows - bottomRows - baseHeaderRows - 1
                                                : 2;
    const std::size_t visibleDeviceSlots = std::min(devices.size(), std::max<std::size_t>(1, availableDeviceRows / 2));
    const std::size_t firstDeviceSlot = deviceWindowStart(devices.size(), selectedDeviceSlot, visibleDeviceSlots);
    const std::size_t headerRows = baseHeaderRows + visibleDeviceSlots * 2;
    const std::size_t middleRows = std::max<std::size_t>(1, terminal.rows - bottomRows - headerRows);
    const auto lines = resultLines(job.result, width - 2, selectedMode);
    const std::size_t maxScroll = lines.size() > middleRows ? lines.size() - middleRows : 0;
    scrollOffset = std::min(scrollOffset, maxScroll);

    const bool dimBackground = editor.open;
    std::ostringstream output;
    output << "\x1b[H";
    if (dimBackground) {
        output << "\x1b[2;48;5;234m";
    }

    std::ostringstream deviceHeading;
    deviceHeading << "CUDA 设备（共 " << devices.size() << " 块，显示 " << (firstDeviceSlot + 1) << '-'
                  << (firstDeviceSlot + visibleDeviceSlots) << "；按 [ / ] 切换，运行中不可切换）";
    appendLine(output, deviceHeading.str(), width);
    for (std::size_t slot = firstDeviceSlot; slot < firstDeviceSlot + visibleDeviceSlots; ++slot) {
        const DeviceSnapshot& device = devices[slot];
        const GpuInfo& gpu = device.gpu;
        std::ostringstream hardware;
        hardware << (slot == selectedDeviceSlot ? "> " : "  ") << '[';
        if (gpu.deviceIndex >= 0) {
            hardware << gpu.deviceIndex;
        } else {
            hardware << '-';
        }
        hardware << "] " << gpu.name;
        if (gpu.available) {
            hardware << " | 计算能力 CC " << gpu.computeMajor << '.' << gpu.computeMinor
                     << " | " << gpu.multiprocessorCount << " SM"
                     << " | 显存 " << formatBytes(gpu.totalMemoryBytes)
                     << " | 理论带宽 " << formatTheoreticalBandwidth(gpu);
        } else {
            hardware << " | " << gpu.reason;
        }
        appendLine(output, hardware.str(), width);
        appendLine(output, deviceTaskLine(device.job), width);
    }

    std::ostringstream status;
    status << "状态：" << stateLabel(job.state) << "   模式：" << modeLabel(selectedMode) << "   ";
    if (selectedMode == CalculationMode::MonteCarlo) {
        status << "目标：" << formatSampleCount(selectedSamples) << "样本   ";
    } else {
        status << "目标：" << selectedDigits << " 位   ";
    }
    status << progressBar(job.completedSteps, job.totalSteps, width / 3);
    appendLine(output, status.str(), width);
    const CalculationMode outputMode = job.result.empty() ? selectedMode : job.mode;
    appendLine(output, outputMode == CalculationMode::MonteCarlo ? "计算输出：蒙特卡洛估计" : "计算输出：Pi 精确位数", width);

    for (std::size_t row = 0; row < middleRows; ++row) {
        const std::size_t line = scrollOffset + row;
        appendLine(output, line < lines.size() ? " " + lines[line] : "", width);
    }

    std::string phase = job.phase.empty() ? job.message : job.phase;
    if (job.mode == CalculationMode::MonteCarlo && job.samplesCompleted > 0) {
        phase += "   样本：" + formatSampleCount(job.samplesCompleted) + " / " + formatSampleCount(job.sampleTarget) +
                 "   圆内命中：" + std::to_string(job.hitsInsideCircle);
    }
    appendLine(output, phase, width);
    appendLine(output, cpuLine(resources), width);
    appendLine(output, gpuLoadLine(resources), width);
    const GpuInfo& selectedGpu = devices[selectedDeviceSlot].gpu;
    if (selectedGpu.available) {
        appendLine(output, "按键：s 开始  e 目标  m 模式  [ / ] 显卡  p 暂停  c 取消  j/k 滚动  q 退出", width);
    } else {
        appendLine(output, "当前 GPU 不可用：s 已禁用，q 退出", width);
    }
    if (job.gpuMilliseconds > 0.0) {
        std::ostringstream timing;
        timing << "CUDA 计时：" << std::fixed << std::setprecision(2) << job.gpuMilliseconds << " ms   " << job.message;
        appendLine(output, timing.str(), width, false);
    } else {
        appendLine(output, job.message, width, false);
    }
    if (dimBackground) {
        output << "\x1b[0m";
        renderTargetEditor(output, terminal, editor, selectedDigits, selectedSamples);
    }

    std::cout << output.str() << std::flush;
}

}  // namespace

int TerminalUi::run(PiEngine& engine) {
    TerminalSession terminal;
    std::size_t monitoredDeviceSlot = engine.selectedDeviceSlot();
    auto monitor = std::make_unique<ResourceMonitor>(engine.selectedGpu());
    CalculationMode selectedMode = CalculationMode::ExactDigits;
    unsigned selectedDigits = kDefaultDigits;
    std::uint64_t selectedSamples = 100'000'000ULL;
    TargetEditor editor;
    std::size_t scrollOffset = 0;
    bool running = true;

    while (running) {
        const KeyEvent event = terminal.pollKey();
        if (editor.open) {
            switch (event.key) {
                case Key::DigitInput:
                    if (editor.input.size() < 10) {
                        editor.input.push_back(event.character);
                        editor.error.clear();
                    }
                    break;
                case Key::Backspace:
                    if (!editor.input.empty()) {
                        editor.input.pop_back();
                        editor.error.clear();
                    }
                    break;
                case Key::ConfirmInput:
                    if (commitTargetEditor(editor, selectedDigits, selectedSamples)) {
                        scrollOffset = 0;
                    }
                    break;
                case Key::DismissInput:
                    editor.open = false;
                    break;
                default:
                    break;
            }
        } else {
            switch (event.key) {
            case Key::Start:
                if (selectedMode == CalculationMode::MonteCarlo) {
                    engine.startMonteCarlo(selectedSamples);
                } else {
                    engine.startExact(selectedDigits);
                }
                scrollOffset = 0;
                break;
            case Key::ToggleMode: {
                const JobState state = engine.snapshot().state;
                if (!jobIsActive(state)) {
                    selectedMode = selectedMode == CalculationMode::ExactDigits ? CalculationMode::MonteCarlo : CalculationMode::ExactDigits;
                    scrollOffset = 0;
                }
                break;
            }
            case Key::Pause:
                engine.togglePause();
                break;
            case Key::Cancel:
                engine.cancel();
                break;
            case Key::EditTarget:
                if (jobIsActive(engine.snapshot().state)) {
                    break;
                }
                editor.open = true;
                editor.mode = selectedMode;
                editor.input.clear();
                editor.error.clear();
                break;
            case Key::ScrollUp:
                scrollOffset = scrollOffset == 0 ? 0 : scrollOffset - 1;
                break;
            case Key::ScrollDown:
                ++scrollOffset;
                break;
            case Key::PreviousDevice:
            case Key::NextDevice: {
                const JobSnapshot currentJob = engine.snapshot();
                const std::vector<DeviceSnapshot> devices = engine.devices();
                if (!jobIsActive(currentJob.state) && devices.size() > 1) {
                    const std::size_t currentSlot = engine.selectedDeviceSlot();
                    const std::size_t nextSlot = event.key == Key::PreviousDevice
                                                     ? (currentSlot == 0 ? devices.size() - 1 : currentSlot - 1)
                                                     : (currentSlot + 1) % devices.size();
                    if (engine.selectDevice(nextSlot)) {
                        scrollOffset = 0;
                    }
                }
                break;
            }
            case Key::Quit:
                engine.stop();
                running = false;
                break;
            case Key::None:
            case Key::ConfirmInput:
            case Key::DismissInput:
            case Key::Backspace:
            case Key::DigitInput:
                break;
            }
        }

        const std::size_t selectedDeviceSlot = engine.selectedDeviceSlot();
        const std::vector<DeviceSnapshot> devices = engine.devices();
        if (selectedDeviceSlot != monitoredDeviceSlot) {
            monitor = std::make_unique<ResourceMonitor>(engine.selectedGpu());
            monitoredDeviceSlot = selectedDeviceSlot;
        }
        render(
            terminal.size(),
            devices,
            selectedDeviceSlot,
            devices[selectedDeviceSlot].job,
            monitor->sample(),
            selectedMode,
            selectedDigits,
            selectedSamples,
            editor,
            scrollOffset);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}

}  // namespace pie
