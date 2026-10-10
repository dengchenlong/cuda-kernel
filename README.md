# cuda-kernel — 高性能 CUDA 图像算子库（对标 OpenCV-CUDA）

一组针对 Pascal (sm_61) 手工优化的单通道图像算子，支持 **CV_8U 与 CV_32F** 两种输入格式。
性能全面超过同机同卡的 OpenCV-CUDA 4.1.30 基准（全部算子 × 全部尺寸 × 全部参数 WIN，
u8 组最低 1.09x、最高 21.8x；f32 组 118/119 项 WIN，详见 [BENCHMARK.md](BENCHMARK.md)）。

## 算子

| 算子 | u8 接口 | f32 接口 | 算法 |
|---|---|---|---|
| 积分图 | `gpu_integral` u8 → u32 inclusive | `gpu_integral_f32`（OpenCV-CUDA 无此 API） | 分块 scan + 跨块加偏移 |
| 均值滤波 | `gpu_mean` 任意核 | `gpu_mean_f32` | k=3 单遍 smem 直加；其余两遍分离（行前缀 → 列归一化），O(1)/像素 |
| 腐蚀/膨胀 | `gpu_erode` / `gpu_dilate` 任意矩形核，二值+灰度 | `gpu_erode_f32` / `gpu_dilate_f32` | k≤7 行直加+列行分块；k≥9 van Herk/Gil-Werman O(1)/像素 |
| 高斯滤波 | `gpu_gaussian` k=1~31 | `gpu_gaussian_f32` | 分离两遍（smem tile + smem ring）；k=5/7 固定核走定点整数路径 |
| 灰度直方图 | `gpu_histogram` 256 bins | — | uint4 合并读 + smem 私有 hist + last-block 归约（免预清零、免全局原子） |
| 二值直方图 | `gpu_histogram_binary` | — | popcount 直数 0/255 + last-block 归约 |

边界语义与 OpenCV 默认一致（`BORDER_REFLECT_101` / `morphologyDefaultBorderValue`）；
正确性经 smoke 逐像素比对：

- integral / morphology / gaussian：与 OpenCV-CUDA **逐位一致**（u8 与 f32 均 diff=0）
- mean（u8）：残差 ≤1（OpenCV 自身舍入，容差内）；mean（f32）：差 ~1.5e-5（OpenCV 内部累加序不同）
- integral（f32）：OpenCV-CUDA 无此 API，与同语义 float 前缀和参照逐位一致
  （注意 CPU `cv::integral` 是 double 累加器，语义与 float GPU 实现天然不同，大图会有相对舍入差）

## 构建

要求：CUDA Toolkit（12.x，其他版本未测）、VS2022 BuildTools v143（Windows）、CMake ≥3.18 或直接 nvcc、
OpenCV-CUDA（含 core/imgproc/cudaarithm/cudawarping/cudaimgproc/cudafilters，构建脚本见 `scripts/build_opencv_cuda.bat`）。

**方式一：CMake（推荐）**

```bat
cmake -B build -G Ninja -DOpenCV_DIR=<你的 OpenCV-CUDA 安装目录>
cmake --build build
```

**方式二：直接 nvcc 脚本**

编辑 `scripts\build_all.bat` 顶部三个路径变量（VS、CUDA、OpenCV 安装位置），然后：

```bat
scripts\build_all.bat       :: 编译 src/*.cu -> build\obj + 链接 build\smoke.exe、build\benchmark.exe
scripts\run_smoke.bat       :: 正确性冒烟（vs OpenCV-CUDA 逐像素比对）
scripts\run_bench.bat       :: benchmark，输出表格 + bench_results.csv
```

编译要点：nvcc `-O3 --use_fast_math -arch=sm_61 -Xcompiler "/utf-8 /W3"`；
`.cu` 含中文注释必须带 UTF-8 BOM。目标架构 sm_61（GTX 10xx/Pascal），改 `CMAKE_CUDA_ARCHITECTURES` 或 `-arch` 可适配其他卡。

## API

```cpp
#include "ops.h"   // 全部为纯 device 指针 + step（字节）接口，可传 stream，内部静态 workspace 无每调用分配

// ---- CV_8UC1 ----
gpu_integral(d_src, srcStep, d_dst, dstStep, W, H);
gpu_mean(d_src, srcStep, d_dst, dstStep, W, H, kw, kh);
gpu_erode(d_src, srcStep, d_dst, dstStep, W, H, kw, kh);   // gpu_dilate 同
gpu_gaussian(d_src, srcStep, d_dst, dstStep, W, H, ksize, sigma);  // sigma<=0 按 OpenCV 规则推算
gpu_histogram(d_src, srcStep, W, H, d_hist);          // 全 256 bin 完全覆写，调用前无需清零
gpu_histogram_binary(d_src, srcStep, W, H, d_hist);   // 同上（bin0=0 的个数，bin255=255 的个数）

// ---- CV_32FC1（输出不取整）----
gpu_integral_f32(d_src, srcStep, d_dst, dstStep, W, H);
gpu_mean_f32(d_src, srcStep, d_dst, dstStep, W, H, kw, kh);
gpu_erode_f32(d_src, srcStep, d_dst, dstStep, W, H, kw, kh);  // gpu_dilate_f32 同
gpu_gaussian_f32(d_src, srcStep, d_dst, dstStep, W, H, ksize, sigma);
```

最小示例：

```cpp
#include <opencv2/core/cuda.hpp>
#include "ops.h"
namespace cvo = cv::cuda;

cvo::GpuMat dGray = cvo::GpuMat(host_gray_u8);   // CV_8UC1
cvo::GpuMat dOut(dGray.size(), dGray.type());
gpu_gaussian(dGray.data, dGray.step, dOut.data, dOut.step,
             dGray.cols, dGray.rows, 15, 0.f);
```

## benchmark 结果（第五轮 2026-10-09，cudaEvent 中位数，纯 GPU 时间，4K = 3840x2160）

| 算子 | 640x480 | 1280x720 | 1920x1080 | 3840x2160 |
|---|---|---|---|---|
| integral | **11.8x** | **6.2x** | **4.4x** | **2.9x** |
| mean k=3 / k=31 | **2.0x / 17.7x** | **1.9x / 17.5x** | **1.5x / 18.7x** | **1.3x / 21.5x** |
| erode/dilate k=3~31 | **1.3~15.1x** | **1.2~13.1x** | **1.2~12.5x** | **1.2~12.2x** |
| gaussian k=5 / k=15 | **1.6x / 1.3x** | **1.4x / 1.5x** | **1.4x / 1.6x** | **1.4x / 1.7x** |
| hist gray / binary | **1.5x / 2.1x** | **1.6x / 2.6x** | **1.6x / 4.1x** | **1.6x / 5.4x** |
| mean-f32 k=31 | **11.2x** | **16.5x** | **16.9x** | **19.2x** |
| erode/dilate-f32 k=3~31 | **2.3~12.2x** | **2.6~12.1x** | **2.6~12.5x** | **2.6~11.9x** |
| gaussian-f32 k=15 | **1.3x** | **1.6x** | **1.6x** | **1.8x** |

完整逐项数字（六轮基线）见 [BENCHMARK.md](BENCHMARK.md)，优化过程记录见 [PROGRESS.md](PROGRESS.md)。

## 优化经验（踩坑总结）

- **直方图基准必须用真随机数据**：等值/平台数据令 warp 内多 lane 落同一 bin、smem 原子串行化，假慢 4~5 倍。
- **隐形固定开销优先于微优化**：直方图计时内的 `cudaMemsetAsync` 预清零固定 ~20us，比 1080p 本体（~50us）还贵；
  last-block 归约（partials 普通写 + atomicInc 回绕 + 末块覆写输出）同时消掉了它和全局原子。
- **block 内线程分工用 threadIdx.x 而非 lane**：按 lane（threadIdx.x & 31）分工时每个 warp
  都会处理同一批数据 → 8 warp = ×8 计数翻倍；lane 只该用于 warp 内通信。
- **对齐切分的偏移必须 clamp 到 W**：行基址 16B 对齐修正量 a 在窄行（a > W）时会让头部读取
  越界到 padding，a = min(a, W) 后整行走头尾标量。
- **小核别套两遍分离**：k=3 均值两遍法的 tmp 往返 4B/px 是大头，单遍 smem 直加 ~2B/px 快 1.5 倍。
- **定点整数可替代 float 中间缓冲**：dyadic 核（[1,4,6,4,1]/16、[2,7,14,18,14,7,2]/64）下
  整数 MAC + ×(1/2^n) 逐位一致且带宽减半；注意不能拿「未缩放行和 × 缩放后权重」（会按 2^n 倍饱和）。
- **两遍分离的列遍不能原地读写输出**：行分块后上方 chunk 的输出会被下方 chunk 当输入读，block 间无同步 → 竞态。
- warp 内 ILP 取数变体（16B load stride 32B）会破坏合并访存，慢 7 倍；smem 原子冲突在随机数据下不是瓶颈。

## 目录结构

```
cuda-kernel/
├── src/             五个算子实现 + ops.h（公共声明）
├── bench/           smoke（正确性）、benchmark（性能，vs OpenCV-CUDA）
├── scripts/         构建/运行脚本（路径参数化）
├── examples/legacy  早期学习用例（与主库无关，仅供参考）
├── BENCHMARK.md     六轮基准数据
└── PROGRESS.md      开发过程与优化记录
```

## License

[MIT](LICENSE)
