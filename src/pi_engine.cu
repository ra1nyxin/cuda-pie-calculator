#include "pi_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pie {
namespace {

constexpr std::uint32_t kBase = 10000;
constexpr unsigned kGuardDecimalDigits = 24;
constexpr unsigned kMonteCarloThreadsPerBlock = 256;
constexpr std::uint64_t kMonteCarloBatchSamples = 8ULL * 1024ULL * 1024ULL;

class Cancelled final : public std::exception {
public:
    const char* what() const noexcept override { return "任务已取消"; }
};

void checkCuda(cudaError_t error, const char* action) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(error));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count) {
        checkCuda(cudaMalloc(&data_, count * sizeof(T)), "CUDA 显存分配失败");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* data() const noexcept { return data_; }

private:
    T* data_ = nullptr;
};

__global__ void initializeAtanKernel(std::uint32_t* term, std::uint32_t* sum, std::size_t limbs, unsigned divisor) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    std::uint64_t remainder = 1;
    for (std::size_t i = 0; i < limbs; ++i) {
        const std::uint64_t dividend = remainder * kBase;
        const auto digit = static_cast<std::uint32_t>(dividend / divisor);
        term[i] = digit;
        sum[i] = digit;
        remainder = dividend % divisor;
    }
}

// One CUDA thread performs carry-sensitive fixed-point work. The data is never
// calculated on the host; the compact base-10000 representation is copied back
// only after the final GPU result is complete.
__global__ void atanStepKernel(
    std::uint32_t* term,
    std::uint32_t* sum,
    std::size_t limbs,
    unsigned iteration,
    unsigned qSquared,
    bool subtract) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const std::uint64_t numerator = 2ULL * iteration + 1ULL;
    const std::uint64_t denominator = (2ULL * iteration + 3ULL) * qSquared;

    std::uint64_t carry = 0;
    for (std::size_t index = limbs; index-- > 0;) {
        const std::uint64_t product = static_cast<std::uint64_t>(term[index]) * numerator + carry;
        term[index] = static_cast<std::uint32_t>(product % kBase);
        carry = product / kBase;
    }

    std::uint64_t remainder = carry;
    for (std::size_t index = 0; index < limbs; ++index) {
        const std::uint64_t dividend = remainder * kBase + term[index];
        term[index] = static_cast<std::uint32_t>(dividend / denominator);
        remainder = dividend % denominator;
    }

    if (subtract) {
        unsigned borrow = 0;
        for (std::size_t index = limbs; index-- > 0;) {
            const std::uint64_t subtrahend = static_cast<std::uint64_t>(term[index]) + borrow;
            if (static_cast<std::uint64_t>(sum[index]) >= subtrahend) {
                sum[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(sum[index]) - subtrahend);
                borrow = 0;
            } else {
                sum[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(sum[index]) + kBase - subtrahend);
                borrow = 1;
            }
        }
    } else {
        carry = 0;
        for (std::size_t index = limbs; index-- > 0;) {
            const std::uint64_t value = static_cast<std::uint64_t>(sum[index]) + term[index] + carry;
            sum[index] = static_cast<std::uint32_t>(value % kBase);
            carry = value / kBase;
        }
    }
}

__global__ void machInQuarterKernel(
    const std::uint32_t* atanOneFifth,
    const std::uint32_t* atanOne239,
    std::uint32_t* quarterPi,
    std::size_t limbs) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    std::uint64_t carry = 0;
    for (std::size_t index = limbs; index-- > 0;) {
        const std::uint64_t product = static_cast<std::uint64_t>(atanOneFifth[index]) * 4ULL + carry;
        quarterPi[index] = static_cast<std::uint32_t>(product % kBase);
        carry = product / kBase;
    }

    unsigned borrow = 0;
    for (std::size_t index = limbs; index-- > 0;) {
        const std::uint64_t subtrahend = static_cast<std::uint64_t>(atanOne239[index]) + borrow;
        if (static_cast<std::uint64_t>(quarterPi[index]) >= subtrahend) {
            quarterPi[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(quarterPi[index]) - subtrahend);
            borrow = 0;
        } else {
            quarterPi[index] = static_cast<std::uint32_t>(static_cast<std::uint64_t>(quarterPi[index]) + kBase - subtrahend);
            borrow = 1;
        }
    }
}

__global__ void scalePiKernel(std::uint32_t* quarterPi, std::size_t limbs, unsigned* integerPart) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    std::uint64_t carry = 0;
    for (std::size_t index = limbs; index-- > 0;) {
        const std::uint64_t product = static_cast<std::uint64_t>(quarterPi[index]) * 4ULL + carry;
        quarterPi[index] = static_cast<std::uint32_t>(product % kBase);
        carry = product / kBase;
    }
    *integerPart = static_cast<unsigned>(carry);
}

__device__ unsigned long long mixRandom64(unsigned long long value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ double uniformUnit(unsigned long long randomBits) {
    return static_cast<double>(randomBits >> 11U) * 1.1102230246251565e-16;
}

// This grid performs independent two-dimensional trials. Each block reduces
// its own hits in shared memory, issuing one global 64-bit atomic update.
__global__ void monteCarloSampleKernel(
    unsigned long long sampleOffset,
    unsigned long long sampleCount,
    unsigned long long seed,
    unsigned long long* hitsInsideCircle) {
    __shared__ unsigned long long blockHits[kMonteCarloThreadsPerBlock];

    const unsigned long long threadId = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned long long stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
    unsigned long long localHits = 0;

    for (unsigned long long index = threadId; index < sampleCount; index += stride) {
        const unsigned long long counter = sampleOffset + index;
        const double x = uniformUnit(mixRandom64(seed ^ (counter * 0xd2b74407b1ce6e93ULL)));
        const double y = uniformUnit(mixRandom64((seed + 0x9e3779b97f4a7c15ULL) ^ (counter * 0xca5a826395121157ULL)));
        localHits += (x * x + y * y <= 1.0) ? 1ULL : 0ULL;
    }

    blockHits[threadIdx.x] = localHits;
    __syncthreads();
    for (unsigned strideSize = blockDim.x / 2U; strideSize > 0U; strideSize >>= 1U) {
        if (threadIdx.x < strideSize) {
            blockHits[threadIdx.x] += blockHits[threadIdx.x + strideSize];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(hitsInsideCircle, blockHits[0]);
    }
}

// The host only displays these values. The Monte Carlo estimate and its normal
// approximation 95% confidence half-width are calculated on the GPU.
__global__ void monteCarloStatisticsKernel(
    const unsigned long long* hitsInsideCircle,
    unsigned long long sampleCount,
    double* statistics) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const double hitFraction = static_cast<double>(*hitsInsideCircle) / static_cast<double>(sampleCount);
    statistics[0] = 4.0 * hitFraction;
    statistics[1] = 1.959963984540054 * 4.0 * sqrt(hitFraction * (1.0 - hitFraction) / static_cast<double>(sampleCount));
}

unsigned numberOfTerms(unsigned q, unsigned decimalDigits) {
    const double denominator = 2.0 * std::log10(static_cast<double>(q));
    return static_cast<unsigned>(std::ceil(static_cast<double>(decimalDigits) / denominator)) + 3U;
}

std::string formatDigits(unsigned integerPart, const std::vector<std::uint32_t>& limbs, unsigned decimalDigits) {
    std::string output = std::to_string(integerPart);
    output.push_back('.');
    output.reserve(output.size() + limbs.size() * 4U);

    for (const auto limb : limbs) {
        char group[5]{};
        std::snprintf(group, sizeof(group), "%04u", limb);
        output.append(group);
    }

    output.resize(std::min(output.size(), std::to_string(integerPart).size() + 1U + decimalDigits));
    return output;
}

std::string formatMonteCarloResult(
    std::uint64_t samples,
    std::uint64_t hits,
    double estimate,
    double confidence95,
    double samplesPerSecond) {
    std::ostringstream output;
    output << "CUDA 蒙特卡洛估计结果\n"
           << "Pi 约为 " << std::fixed << std::setprecision(12) << estimate << '\n'
           << "95% 置信区间：+/- " << std::setprecision(12) << confidence95 << '\n'
           << "样本数：" << samples << "   圆内命中：" << hits << '\n'
           << "吞吐量：" << std::setprecision(1) << samplesPerSecond << " 样本/秒";
    return output.str();
}

bool jobIsActive(JobState state) {
    return state == JobState::Preparing || state == JobState::Running || state == JobState::Paused || state == JobState::Cancelling;
}

std::vector<GpuInfo> probeGpus() {
    std::vector<GpuInfo> devices;
    int deviceCount = 0;
    const cudaError_t countResult = cudaGetDeviceCount(&deviceCount);
    if (countResult != cudaSuccess || deviceCount <= 0) {
        GpuInfo info;
        const char* detail = countResult == cudaSuccess ? "CUDA 运行时未返回任何设备" : cudaGetErrorString(countResult);
        info.name = "未发现 CUDA GPU";
        info.reason = std::string("GPU 不可用：") + detail;
        devices.push_back(std::move(info));
        cudaGetLastError();
        return devices;
    }

    int driver = 0;
    int runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);

    devices.reserve(static_cast<std::size_t>(deviceCount));
    for (int deviceIndex = 0; deviceIndex < deviceCount; ++deviceIndex) {
        GpuInfo info;
        info.deviceIndex = deviceIndex;
        info.driverVersion = driver;
        info.runtimeVersion = runtime;

        cudaDeviceProp properties{};
        const cudaError_t propertiesResult = cudaGetDeviceProperties(&properties, deviceIndex);
        if (propertiesResult != cudaSuccess) {
            info.name = "CUDA 设备属性读取失败";
            info.reason = std::string("GPU 不可用：") + cudaGetErrorString(propertiesResult);
            cudaGetLastError();
            devices.push_back(std::move(info));
            continue;
        }

        info.available = true;
        info.name = properties.name;
        info.totalMemoryBytes = properties.totalGlobalMem;
        info.computeMajor = properties.major;
        info.computeMinor = properties.minor;
        info.multiprocessorCount = properties.multiProcessorCount;
        info.coreClockKHz = properties.clockRate;
        info.memoryClockKHz = properties.memoryClockRate;
        info.memoryBusWidthBits = properties.memoryBusWidth;
        devices.push_back(std::move(info));
    }
    return devices;
}

}  // namespace

const char* stateLabel(JobState state) {
    switch (state) {
        case JobState::GpuUnavailable: return "GPU 不可用";
        case JobState::Idle: return "就绪";
        case JobState::Preparing: return "正在准备 CUDA 任务";
        case JobState::Running: return "正在使用 GPU 计算";
        case JobState::Paused: return "已暂停";
        case JobState::Cancelling: return "正在停止";
        case JobState::Finished: return "已完成";
        case JobState::Cancelled: return "已取消";
        case JobState::Failed: return "失败";
    }
    return "未知状态";
}

const char* modeLabel(CalculationMode mode) {
    switch (mode) {
        case CalculationMode::ExactDigits: return "精确位数";
        case CalculationMode::MonteCarlo: return "蒙特卡洛";
    }
    return "未知模式";
}

PiEngine::PiEngine() : gpus_(probeGpus()) {
    deviceSnapshots_.reserve(gpus_.size());
    for (const GpuInfo& gpu : gpus_) {
        JobSnapshot snapshot;
        snapshot.state = gpu.available ? JobState::Idle : JobState::GpuUnavailable;
        snapshot.message = gpu.available ? "CUDA 设备已就绪，按 s 开始计算。" : gpu.reason;
        deviceSnapshots_.push_back(std::move(snapshot));
    }
}

PiEngine::~PiEngine() {
    stop();
}

std::vector<DeviceSnapshot> PiEngine::devices() const {
    std::scoped_lock lock(snapshotMutex_);
    std::vector<DeviceSnapshot> result;
    result.reserve(gpus_.size());
    for (std::size_t index = 0; index < gpus_.size(); ++index) {
        result.push_back(DeviceSnapshot{gpus_[index], deviceSnapshots_[index]});
    }
    return result;
}

GpuInfo PiEngine::selectedGpu() const {
    std::scoped_lock lock(snapshotMutex_);
    return selectedDeviceSlot_ < gpus_.size() ? gpus_[selectedDeviceSlot_] : GpuInfo{};
}

std::size_t PiEngine::selectedDeviceSlot() const {
    std::scoped_lock lock(snapshotMutex_);
    return selectedDeviceSlot_;
}

bool PiEngine::selectDevice(std::size_t deviceSlot) {
    std::scoped_lock lock(snapshotMutex_);
    if (deviceSlot >= gpus_.size() || jobIsActive(deviceSnapshots_[selectedDeviceSlot_].state)) {
        return false;
    }
    selectedDeviceSlot_ = deviceSlot;
    return true;
}

JobSnapshot PiEngine::snapshot() const {
    std::scoped_lock lock(snapshotMutex_);
    return deviceSnapshots_[selectedDeviceSlot_];
}

void PiEngine::setSnapshot(const JobSnapshot& value) {
    std::scoped_lock lock(snapshotMutex_);
    deviceSnapshots_[selectedDeviceSlot_] = value;
}

void PiEngine::updateProgress(unsigned complete, unsigned total, const std::string& phase) {
    std::scoped_lock lock(snapshotMutex_);
    JobSnapshot& snapshot = deviceSnapshots_[selectedDeviceSlot_];
    snapshot.completedSteps = complete;
    snapshot.totalSteps = total;
    snapshot.phase = phase;
    if (snapshot.state == JobState::Preparing) {
        snapshot.state = JobState::Running;
    }
}

void PiEngine::updateMonteCarloProgress(
    std::uint64_t completed,
    std::uint64_t target,
    std::uint64_t hits,
    double estimate,
    double confidence95,
    double samplesPerSecond) {
    std::scoped_lock lock(snapshotMutex_);
    JobSnapshot& snapshot = deviceSnapshots_[selectedDeviceSlot_];
    snapshot.samplesCompleted = completed;
    snapshot.sampleTarget = target;
    snapshot.hitsInsideCircle = hits;
    snapshot.monteCarloEstimate = estimate;
    snapshot.monteCarloConfidence95 = confidence95;
    snapshot.samplesPerSecond = samplesPerSecond;
    snapshot.completedSteps = static_cast<unsigned>((completed * 1000ULL) / target);
    snapshot.totalSteps = 1000;
    snapshot.phase = "正在使用 CUDA 进行蒙特卡洛采样";
    snapshot.result = formatMonteCarloResult(completed, hits, estimate, confidence95, samplesPerSecond);
    if (snapshot.state == JobState::Preparing) {
        snapshot.state = JobState::Running;
    }
}

bool PiEngine::beginJob(const JobSnapshot& initial) {
    {
        std::scoped_lock lock(snapshotMutex_);
        if (!gpus_[selectedDeviceSlot_].available) {
            JobSnapshot& snapshot = deviceSnapshots_[selectedDeviceSlot_];
            snapshot.state = JobState::GpuUnavailable;
            snapshot.message = gpus_[selectedDeviceSlot_].reason;
            return false;
        }
        if (jobIsActive(deviceSnapshots_[selectedDeviceSlot_].state)) {
            return false;
        }
    }
    if (worker_.joinable()) {
        worker_.join();
    }

    cancelRequested_.store(false);
    paused_.store(false);
    setSnapshot(initial);
    return true;
}

bool PiEngine::startExact(unsigned digits) {
    if (digits < kMinimumDigits || digits > kMaximumDigits) {
        JobSnapshot failed = snapshot();
        failed.state = JobState::Failed;
        failed.message = "请求精度超出支持范围：小数点后 10 到 21 亿位。";
        setSnapshot(failed);
        return false;
    }

    JobSnapshot initial;
    initial.state = JobState::Preparing;
    initial.mode = CalculationMode::ExactDigits;
    initial.requestedDigits = digits;
    initial.message = "正在分配 CUDA 定点计算缓冲区。";
    if (!beginJob(initial)) {
        return false;
    }
    worker_ = std::thread(&PiEngine::runExact, this, digits, selectedGpu().deviceIndex);
    return true;
}

bool PiEngine::startMonteCarlo(std::uint64_t samples) {
    if (samples < kMinimumMonteCarloSamples || samples > kMaximumMonteCarloSamples) {
        JobSnapshot failed = snapshot();
        failed.state = JobState::Failed;
        failed.message = "蒙特卡洛样本数超出支持范围：100 万到 40 亿。";
        setSnapshot(failed);
        return false;
    }

    JobSnapshot initial;
    initial.state = JobState::Preparing;
    initial.mode = CalculationMode::MonteCarlo;
    initial.sampleTarget = samples;
    initial.message = "正在准备 CUDA 蒙特卡洛采样网格。";
    if (!beginJob(initial)) {
        return false;
    }
    worker_ = std::thread(&PiEngine::runMonteCarlo, this, samples, selectedGpu().deviceIndex);
    return true;
}

void PiEngine::togglePause() {
    const JobState current = snapshot().state;
    if (current == JobState::Running) {
        paused_.store(true);
        std::scoped_lock lock(snapshotMutex_);
        deviceSnapshots_[selectedDeviceSlot_].state = JobState::Paused;
        deviceSnapshots_[selectedDeviceSlot_].message = "当前 CUDA 内核结束后将暂停。";
    } else if (current == JobState::Paused) {
        paused_.store(false);
        {
            std::scoped_lock lock(snapshotMutex_);
            deviceSnapshots_[selectedDeviceSlot_].state = JobState::Running;
            deviceSnapshots_[selectedDeviceSlot_].message = "已继续 CUDA 计算。";
        }
        pauseChanged_.notify_all();
    }
}

void PiEngine::cancel() {
    cancelRequested_.store(true);
    paused_.store(false);
    {
        std::scoped_lock lock(snapshotMutex_);
        JobSnapshot& snapshot = deviceSnapshots_[selectedDeviceSlot_];
        if (snapshot.state == JobState::Preparing || snapshot.state == JobState::Running || snapshot.state == JobState::Paused) {
            snapshot.state = JobState::Cancelling;
            snapshot.message = "正在停止 CUDA 任务。";
        }
    }
    pauseChanged_.notify_all();
}

void PiEngine::stop() {
    cancel();
    if (worker_.joinable()) {
        worker_.join();
    }
}

bool PiEngine::cancellationRequested() const noexcept {
    return cancelRequested_.load();
}

void PiEngine::waitWhilePaused() {
    std::unique_lock lock(pauseMutex_);
    pauseChanged_.wait(lock, [this] { return !paused_.load() || cancellationRequested(); });
}

void PiEngine::runExact(unsigned digits, int cudaDeviceIndex) {
    cudaStream_t stream = nullptr;
    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;

    try {
        checkCuda(cudaSetDevice(cudaDeviceIndex), "CUDA 设备选择失败");
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "CUDA 流创建失败");
        checkCuda(cudaEventCreate(&started), "CUDA 起始事件创建失败");
        checkCuda(cudaEventCreate(&finished), "CUDA 结束事件创建失败");

        const unsigned workingDigits = digits + kGuardDecimalDigits;
        const std::size_t limbs = (workingDigits + 3U) / 4U;
        const unsigned fifthTerms = numberOfTerms(5U, workingDigits);
        const unsigned twoThirtyNineTerms = numberOfTerms(239U, workingDigits);
        const unsigned totalSteps = fifthTerms + twoThirtyNineTerms + 3U;
        unsigned complete = 0;

        DeviceBuffer<std::uint32_t> atanFifth(limbs);
        DeviceBuffer<std::uint32_t> atanTwoThirtyNine(limbs);
        DeviceBuffer<std::uint32_t> term(limbs);
        DeviceBuffer<std::uint32_t> quarterPi(limbs);
        DeviceBuffer<unsigned> integerPart(1);

        checkCuda(cudaEventRecord(started, stream), "CUDA 起始事件记录失败");

        const auto calculateAtan = [&](unsigned q, DeviceBuffer<std::uint32_t>& destination, const char* phase) {
            initializeAtanKernel<<<1, 1, 0, stream>>>(term.data(), destination.data(), limbs, q);
            checkCuda(cudaGetLastError(), "CUDA 反正切初始化失败");
            checkCuda(cudaStreamSynchronize(stream), "CUDA 反正切初始化同步失败");
            ++complete;
            updateProgress(complete, totalSteps, phase);

            const unsigned qSquared = q * q;
            const unsigned termCount = q == 5U ? fifthTerms : twoThirtyNineTerms;
            for (unsigned iteration = 0; iteration + 1U < termCount; ++iteration) {
                waitWhilePaused();
                if (cancellationRequested()) {
                    throw Cancelled();
                }

                atanStepKernel<<<1, 1, 0, stream>>>(
                    term.data(), destination.data(), limbs, iteration, qSquared, (iteration % 2U) == 0U);
                checkCuda(cudaGetLastError(), "CUDA 反正切迭代失败");

                if ((iteration + 1U) % 8U == 0U || iteration + 2U == termCount) {
                    checkCuda(cudaStreamSynchronize(stream), "CUDA 反正切同步失败");
                }
                ++complete;
                if ((iteration + 1U) % 4U == 0U || iteration + 2U == termCount) {
                    updateProgress(complete, totalSteps, phase);
                }
            }
        };

        calculateAtan(5U, atanFifth, "正在 CUDA GPU 上计算 atan(1/5)");
        calculateAtan(239U, atanTwoThirtyNine, "正在 CUDA GPU 上计算 atan(1/239)");

        waitWhilePaused();
        if (cancellationRequested()) {
            throw Cancelled();
        }
        machInQuarterKernel<<<1, 1, 0, stream>>>(atanFifth.data(), atanTwoThirtyNine.data(), quarterPi.data(), limbs);
        checkCuda(cudaGetLastError(), "CUDA Machin 公式合并失败");
        checkCuda(cudaStreamSynchronize(stream), "CUDA Machin 公式合并同步失败");
        ++complete;
        updateProgress(complete, totalSteps, "正在 CUDA GPU 上合并 Machin 公式");

        scalePiKernel<<<1, 1, 0, stream>>>(quarterPi.data(), limbs, integerPart.data());
        checkCuda(cudaGetLastError(), "CUDA 圆周率缩放失败");
        checkCuda(cudaStreamSynchronize(stream), "CUDA 圆周率缩放同步失败");
        ++complete;
        updateProgress(complete, totalSteps, "正在整理 CUDA 定点计算结果");

        std::vector<std::uint32_t> hostLimbs(limbs);
        unsigned hostInteger = 0;
        checkCuda(cudaMemcpyAsync(hostLimbs.data(), quarterPi.data(), limbs * sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream),
                  "CUDA 结果复制失败");
        checkCuda(cudaMemcpyAsync(&hostInteger, integerPart.data(), sizeof(hostInteger), cudaMemcpyDeviceToHost, stream),
                  "CUDA 整数部分复制失败");
        checkCuda(cudaEventRecord(finished, stream), "CUDA 结束事件记录失败");
        checkCuda(cudaStreamSynchronize(stream), "CUDA 最终同步失败");
        float elapsedMilliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, started, finished), "CUDA 用时查询失败");
        ++complete;

        JobSnapshot result;
        result.state = JobState::Finished;
        result.mode = CalculationMode::ExactDigits;
        result.requestedDigits = digits;
        result.completedSteps = complete;
        result.totalSteps = totalSteps;
        result.gpuMilliseconds = elapsedMilliseconds;
        result.phase = "CUDA 计算完成";
        result.message = "结果已由当前 CUDA GPU 计算完成，程序没有 CPU 回退路径。";
        result.result = formatDigits(hostInteger, hostLimbs, digits);
        setSnapshot(result);
    } catch (const Cancelled&) {
        JobSnapshot cancelled = snapshot();
        cancelled.state = JobState::Cancelled;
        cancelled.message = "CUDA 计算已取消，未使用 CPU 回退。";
        setSnapshot(cancelled);
    } catch (const std::exception& error) {
        JobSnapshot failed = snapshot();
        failed.state = JobState::Failed;
        failed.message = std::string("CUDA 计算失败：") + error.what() + "。程序没有 CPU 回退路径。";
        setSnapshot(failed);
    }

    if (finished != nullptr) {
        cudaEventDestroy(finished);
    }
    if (started != nullptr) {
        cudaEventDestroy(started);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
}

void PiEngine::runMonteCarlo(std::uint64_t samples, int cudaDeviceIndex) {
    cudaStream_t stream = nullptr;
    cudaEvent_t started = nullptr;
    cudaEvent_t finished = nullptr;

    try {
        checkCuda(cudaSetDevice(cudaDeviceIndex), "CUDA 设备选择失败");
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "CUDA 流创建失败");
        checkCuda(cudaEventCreate(&started), "CUDA 起始事件创建失败");
        checkCuda(cudaEventCreate(&finished), "CUDA 结束事件创建失败");

        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, cudaDeviceIndex), "CUDA 设备属性查询失败");
        const int requestedBlocks = std::max(1, properties.multiProcessorCount * 32);
        const int blocks = std::min(requestedBlocks, properties.maxGridSize[0]);

        DeviceBuffer<unsigned long long> hitsInsideCircle(1);
        DeviceBuffer<double> statistics(2);
        checkCuda(cudaMemsetAsync(hitsInsideCircle.data(), 0, sizeof(unsigned long long), stream), "CUDA 命中计数器初始化失败");
        checkCuda(cudaEventRecord(started, stream), "CUDA 起始事件记录失败");

        const auto hostStarted = std::chrono::steady_clock::now();
        const unsigned long long seed = static_cast<unsigned long long>(hostStarted.time_since_epoch().count()) ^
                                        (static_cast<unsigned long long>(samples) * 0x9e3779b97f4a7c15ULL);
        std::uint64_t completed = 0;
        unsigned long long hostHits = 0;
        double hostStatistics[2]{};

        while (completed < samples) {
            waitWhilePaused();
            if (cancellationRequested()) {
                throw Cancelled();
            }

            const std::uint64_t batchSamples = std::min(kMonteCarloBatchSamples, samples - completed);
            monteCarloSampleKernel<<<blocks, kMonteCarloThreadsPerBlock, 0, stream>>>(
                static_cast<unsigned long long>(completed),
                static_cast<unsigned long long>(batchSamples),
                seed,
                hitsInsideCircle.data());
            checkCuda(cudaGetLastError(), "CUDA 蒙特卡洛采样内核失败");

            const std::uint64_t nextCompleted = completed + batchSamples;
            monteCarloStatisticsKernel<<<1, 1, 0, stream>>>(
                hitsInsideCircle.data(), static_cast<unsigned long long>(nextCompleted), statistics.data());
            checkCuda(cudaGetLastError(), "CUDA 蒙特卡洛统计内核失败");
            checkCuda(cudaMemcpyAsync(&hostHits, hitsInsideCircle.data(), sizeof(hostHits), cudaMemcpyDeviceToHost, stream),
                      "CUDA 蒙特卡洛命中数复制失败");
            checkCuda(cudaMemcpyAsync(hostStatistics, statistics.data(), sizeof(hostStatistics), cudaMemcpyDeviceToHost, stream),
                      "CUDA 蒙特卡洛统计结果复制失败");
            checkCuda(cudaStreamSynchronize(stream), "CUDA 蒙特卡洛批次同步失败");

            completed = nextCompleted;
            const double elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - hostStarted).count();
            const double throughput = elapsedSeconds > 0.0 ? static_cast<double>(completed) / elapsedSeconds : 0.0;
            updateMonteCarloProgress(completed, samples, hostHits, hostStatistics[0], hostStatistics[1], throughput);
        }

        checkCuda(cudaEventRecord(finished, stream), "CUDA 结束事件记录失败");
        checkCuda(cudaStreamSynchronize(stream), "CUDA 蒙特卡洛最终同步失败");
        float elapsedMilliseconds = 0.0F;
        checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, started, finished), "CUDA 用时查询失败");

        JobSnapshot result = snapshot();
        result.state = JobState::Finished;
        result.mode = CalculationMode::MonteCarlo;
        result.completedSteps = 1000;
        result.totalSteps = 1000;
        result.gpuMilliseconds = elapsedMilliseconds;
        result.phase = "CUDA 蒙特卡洛计算完成";
        result.message = "样本、估计值和置信区间均由当前 CUDA GPU 计算完成。";
        result.result = formatMonteCarloResult(
            result.samplesCompleted,
            result.hitsInsideCircle,
            result.monteCarloEstimate,
            result.monteCarloConfidence95,
            result.samplesPerSecond);
        setSnapshot(result);
    } catch (const Cancelled&) {
        JobSnapshot cancelled = snapshot();
        cancelled.state = JobState::Cancelled;
        cancelled.message = "CUDA 蒙特卡洛计算已取消，未使用 CPU 回退。";
        setSnapshot(cancelled);
    } catch (const std::exception& error) {
        JobSnapshot failed = snapshot();
        failed.state = JobState::Failed;
        failed.message = std::string("CUDA 蒙特卡洛计算失败：") + error.what() + "。程序没有 CPU 回退路径。";
        setSnapshot(failed);
    }

    if (finished != nullptr) {
        cudaEventDestroy(finished);
    }
    if (started != nullptr) {
        cudaEventDestroy(started);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
}

}  // namespace pie
