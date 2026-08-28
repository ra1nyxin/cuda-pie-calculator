# CUDA 圆周率计算器

<img width="100%" height="100%" alt="114514" src="https://github.com/user-attachments/assets/aebd78a0-9036-4fda-88c0-61baf19fb4a0" />

## 支持范围

- Windows 10 / Windows 11 x64
- Ubuntu 22.04+ x64
- Debian 12+ x64
- NVIDIA CUDA GPU 与对应驱动

## 计算模式

程序提供两个 CUDA 计算模式，可用 `m` 在未运行时切换。

### 精确位数

精确模式使用 Machin 公式：

```text
pi / 4 = 4 * atan(1 / 5) - atan(1 / 239)
```

每个反正切项使用基数 `10^4` 的定点向量表示。初始化、级数项更新、进位、借位、Machin 合并和最终缩放均在 CUDA 内核中执行。主机端只负责任务控制、TUI 绘制、系统负载采样，以及在完成后把 GPU 结果格式化为显示文本。

当前精度范围是小数点后 10 到 2,147,483,647 位（INT32_MAX），默认目标为 10,000 位。这个上限需要极大的 GPU 显存和计算时间；资源不足时 CUDA 会明确报错，程序不会使用 CPU 回退。GPU 内核使用额外保护位，输出前会截去保护位。

### 蒙特卡洛验证

蒙特卡洛模式使用大量 CUDA 线程在单位正方形内生成均匀随机点，以落在四分之一圆内的比例估计 Pi。每个 block 在共享内存中归约命中数，再进行全局 64 位原子累加；Pi 估计值和正态近似的 95% 置信区间也由 CUDA 内核计算。

这是一种概率估计，不应与精确位数混淆。默认目标为 1 亿样本，支持 100 万到 40 亿样本。TUI 会实时显示样本数、命中数、估计值、置信区间和采样吞吐量。

## 使用发布包

从仓库的唯一预发布 Release 下载与系统对应的资产后运行：

```bash
./cuda-pie-calculator
```

Windows 可在 PowerShell 或 Windows Terminal 中运行 `cuda-pie-calculator.exe`。程序需要交互式终端；Windows 10/11 会启用 Virtual Terminal 模式来显示 TUI。

按键：

- `s`：开始 CUDA 计算
- `m`：在精确位数与蒙特卡洛模式间切换（任务未运行时）
- `[` / `]`：切换当前 CUDA 设备（任务未运行时）
- `p`：暂停或继续
- `c`：取消当前 CUDA 任务
- `+` / `-`：精确模式按当前数量级以 100 到 1 亿位自适应调整；蒙特卡洛模式以 1,000 万样本调整目标
- `j` / `k` 或上下方向键：滚动结果区域
- `q`：退出

## 本地构建

需要 CMake 3.24+、支持 C++20 的编译器和 CUDA Toolkit 12.x。构建时可按实际 GPU 调整 `CMAKE_CUDA_ARCHITECTURES`。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

本项目不要求在没有 GPU 的机器上构建或运行。没有 CUDA 设备时，运行程序只会给出 GPU 不可用诊断，绝不会以 CPU 继续计算。

## 自动发布

推送到 `main` 或 `master` 会触发 GitHub Actions：

1. 在 CUDA Toolkit 环境中分别构建 Linux x64 与 Windows x64 二进制。
2. 上传 `cuda-pie-calculator-linux-amd64.tar.gz` 与 `cuda-pie-calculator-windows-amd64.zip`。
3. 删除仓库中的已有 Release，创建一个以工作流运行号和提交 SHA 自动生成标签的预发布 Release。

因此仓库始终只保留一条 Release，不需要手工维护应用版本号。
