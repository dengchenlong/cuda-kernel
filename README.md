# cuda-kernel — GTX 1050 Ti (sm_61) 高性能 CUDA 图像算子库

一组针对 Pascal (sm_61) 手工优化的图像算子，性能全面超过同机同卡的 OpenCV-CUDA 4.1.30 基准
（全部算子 × 全部尺寸 × 全部参数 ≥1，最低 1.09x，最高 21.8x，详见 [BENCHMARK.md](BENCHMARK.md)）。

## 算子

| 算子 | 签名 | 算法 |
|---|---|---|
| 积分图 | `gpu_integral` u8 → u32 inclusive | 分块 scan + 跨块加偏移 |
| 均值滤波 | `gpu_mean` 任意核 u8→u8 | k=3 单遍 smem 直加；其余两遍分离（行前缀/直加 → 列归一化），O(1)/像素 |
| 腐蚀/膨胀 | `gpu_erode` / `gpu_dilate` 任意矩形核，二值+灰度 | k≤7 行直加+列行分块；k≥9 van Herk/Gil-Werman O(1)/像素 |
| 高斯滤波 | `gpu_gaussian` u8→u8, k=1~31 | 分离两遍（smem tile + smem ring）；k=5 固定核走定点整数路径 |
| 灰度直方图 | `gpu_histogram` 256 bins | uint4 合并读 + smem 私有 hist + last-block 归约（免预清零、免全局原子） |
| 二值直方图 | `gpu_histogram_binary` | popcount 直数 0/255 + last-block 归约 |

边界语义与 OpenCV 默认一致（`BORDER_REFLECT_101`）；正确性经 smoke 逐像素比对：
integral/morph/gauss/hist 与 OpenCV **逐位一致**，mean 残差 ≤1（OpenCV 自身舍入，容差内）。

## 构建

要求：CUDA 12.5、VS2022 BuildTools v143（cl 19.44）、CMake/ninja（可选，仓库用直接 nvcc 脚本）、
OpenCV-CUDA（`D:\opencv-cuda-build\install`，含 core/imgproc/cudaarithm/cudawarping/cudaimgproc/cudafilters）。

```bat
D:\deps\build_all.bat      :: 编译 src/*.cu -> build\obj + 链接 build\smoke.exe、build\benchmark.exe
build\smoke.exe            :: 正确性冒烟（vs OpenCV 逐像素比对）
D:\deps\run_bench.bat      :: 跑 benchmark，输出表格 + build\bench_results.csv
```

编译要点：nvcc `-O3 --use_fast_math -arch=sm_61 -Xcompiler "/utf-8 /W3"`；
`.cu` 含中文注释必须带 UTF-8 BOM。

## API

```cpp
#include "ops.h"   // 全部为纯 device 指针 + step（字节）接口，可传 stream，内部静态 workspace 无每调用分配

gpu_integral(d_src, srcStep, d_dst, dstStep, W, H);
gpu_mean(d_src, srcStep, d_dst, dstStep, W, H, kw, kh);
gpu_erode(d_src, srcStep, d_dst, dstStep, W, H, k);   // gpu_dilate 同
gpu_gaussian(d_src, srcStep, d_dst, dstStep, W, H, ksize, sigma);  // sigma<=0 按 OpenCV 规则推算
gpu_histogram(d_src, srcStep, W, H, d_hist);          // 全 256 bin 完全覆写，调用前无需清零
gpu_histogram_binary(d_src, srcStep, W, H, d_hist);   // 同上（bin0=0 的个数，bin255=255 的个数）
```

## benchmark 结果（2026-09-28 第三轮，cudaEvent 中位数，纯 GPU 时间，4K = 3840x2160）

| 算子 | 640x480 | 1280x720 | 1920x1080 | 3840x2160 |
|---|---|---|---|---|
| integral | **7.3x** | **3.9x** | **2.9x** | **2.0x** |
| mean k=3 | **1.7x** | **1.5x** | **1.2x** | **1.2x** |
| mean k=31 | **18.8x** | **18.0x** | **19.2x** | **21.8x** |
| erode/dilate k=3~31 | **1.1~15.5x** | **1.3~13.5x** | **1.1~11.1x** | **1.1~10.6x** |
| gaussian k=5 / k=15 | **1.6x / 1.4x** | **1.5x / 1.6x** | **1.4x / 1.6x** | **1.4x / 1.8x** |
| hist gray / binary | **2.8x / 2.5x** | **2.3x / 3.4x** | **2.0x / 5.2x** | **1.4x / 6.5x** |

完整逐项数字（三轮基线）见 [BENCHMARK.md](BENCHMARK.md)，进度与优化记录见 [PROGRESS.md](PROGRESS.md)。

## 优化经验（踩坑总结）

- **直方图基准必须用真随机数据**：等值/平台数据令 warp 内多 lane 落同一 bin、smem 原子串行化，假慢 4~5 倍。
- **隐形固定开销优先于微优化**：直方图计时内的 `cudaMemsetAsync` 预清零固定 ~20us，比 1080p 本体（~50us）还贵；
  last-block 归约（partials 普通写 + atomicInc 回绕 + 末块覆写输出）同时消掉了它和全局原子。
- **block 内线程分工用 threadIdx.x 而非 lane**：按 lane（threadIdx.x & 31）分工时每个 warp
  都会处理同一批数据 → 8 warp = ×8 计数翻倍（probe_hfix11）；lane 只该用于 warp 内通信。
- **对齐切分的偏移必须 clamp 到 W**：行基址 16B 对齐修正量 a 在窄行（a > W）时会让头部读取
  越界到 padding（probe_hfix12），a = min(a, W) 后整行走头尾标量。
- **小核别套两遍分离**：k=3 均值两遍法的 tmp 往返 4B/px 是大头，单遍 smem 直加 ~2B/px 快 1.5 倍。
- **定点整数可替代 float 中间缓冲**：dyadic 核（[1,4,6,4,1]/16）下 OpenCV 的 CV_32F 管线全程无舍入，
  整数路径（u16 tmp）逐位一致且带宽减半。
- warp 内 ILP 取数变体（16B load stride 32B）会破坏合并访存，慢 7 倍；smem 原子冲突在随机数据下不是瓶颈。
