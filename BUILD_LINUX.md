# Linux 服务器构建（vcpkg 管理 OpenCV-CUDA）

Windows 笔记本构建见 README 主文档。本文件记录服务器（无预装 OpenCV-CUDA）的构建路径。
方向：**继续用 OpenCV-CUDA 作为唯一对标基线**（与笔记本一致），缺失时用 **vcpkg** 安装，
不做 NPP 兜底。

## 已验证环境

- Ubuntu 24.04，CUDA Toolkit 12.8（`/usr/local/cuda`），2× RTX 5090 D (sm_120)
- vcpkg（`~/vcpkg`）+ OpenCV 4.14.0（`opencv4[cuda]`，含 cudaarithm/cudafilters/
  cudaimgproc/cudawarping/cudev）

## 一键脚本

```bash
./scripts/build_linux.sh
```

脚本完成：装 vcpkg → `vcpkg install`（用仓库根 `vcpkg.json` manifest，精简只编
`opencv4[cuda]`）→ 配置 CMake → 构建 → `./smoke` 逐像素比对。

## 关键参数（实测必需）

1. **`VCPKG_MAX_CONCURRENCY=4`**：服务器默认 32 核并行编译时，并发子进程会被环境
   SIGTERM（约 2 分钟内被杀，非 OOM——123GB 内存充足）。限制到 4 后编译稳定
   （1.2 小时完整跑完）。
2. **`-DCUDACXX=/usr/local/cuda/bin/nvcc`**：nvcc 不在默认 PATH，CMake 找不到
   CUDA 编译器会报 `No CMAKE_CUDA_COMPILER`。
3. **`-DOpenCV_DIR=<vcpkg_installed>/x64-linux/share/opencv4`**：项目 CMakeLists
   用 `set(OpenCV_DIR "" CACHE)` 强制要求显式指定（FATAL_ERROR），须手动传 vcpkg
   安装的 OpenCVConfig.cmake 目录。
4. **`-DCMAKE_CUDA_ARCHITECTURES=native`**：5090 是 sm_120；Pascal 笔记本默认 61。
   用 `-DCMAKE_CUDA_ARCHITECTURES="61;120"` 可同时出两套 SASS。

## 验证结果（本机实测，OpenCV-CUDA 4.14 基线）

- `smoke`：**all pass (0 flagged)**。integral/erode/dilate/gaussian/histogram 与
  OpenCV-CUDA 逐位一致（边界语义完全一致，无 REPLICATE 差异）；mean u8 有 ±1
  舍入差（OpenCV 自身累加舍入，README 已注明容差内）；mean/gauss f32 ≤1.5e-5
  （MAC 次序差）。
- `benchmark`：**104 WIN / 16 lose** vs OpenCV-CUDA。lose 集中在 integral 与
  gaussian 大尺寸/大核（库为 Pascal sm_61 优化，OpenCV 4.14 对 sm_120 优化更好，
  属预期性能差异，非正确性问题）。

## 维护注意

- 若换机器/路径，改 `scripts/build_linux.sh` 里的 `VCPKG`、`OpenCV_DIR_VCPKG`、
  `CUDACXX` 三处。
- vcpkg manifest `vcpkg.json` 用 `default-features:false` 关掉 gtk/highgui/dnn/
  gapi/v4l 等图形模块（服务器无 X 环境，避免编译失败），只编 core/imgproc+cuda。