#!/usr/bin/env bash
# Linux 服务器：用 vcpkg 安装 OpenCV-CUDA 并构建/验证 cuda-kernel
# 用法：在服务器终端执行  ./scripts/build_linux.sh
# 说明：OpenCV-CUDA 全量编译需 1~1.5 小时；建议在你的终端运行（若用 Hermes
#       后台跑，务必设 VCPKG_MAX_CONCURRENCY=4，否则并发子进程会被环境清理）。
set -euo pipefail

export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
export VCPKG_FORCE_SYSTEM_BINARIES=1
export VCPKG_DISABLE_METRICS=1
export VCPKG_MAX_CONCURRENCY="${VCPKG_MAX_CONCURRENCY:-4}"   # 关键：限制并行，规避环境清理
VCPKG="$HOME/vcpkg/vcpkg"

echo "==> [1/4] 准备 vcpkg（如已存在则复用）"
if [ ! -x "$VCPKG" ]; then
  if [ ! -d "$HOME/vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
  fi
  (cd "$HOME/vcpkg" && ./bootstrap-vcpkg.sh -disableMetrics)
fi

echo "==> [2/4] 安装 OpenCV-CUDA（manifest 精简：opencv4[cuda]，约 1~1.5 小时）"
cd "$(dirname "$0")/.."     # 回到仓库根（manifest vcpkg.json 所在）
"$VCPKG" install

echo "==> [3/4] 配置并构建 cuda-kernel（OpenCV 走 vcpkg + 显式 nvcc/OpenCV_DIR）"
OpenCV_DIR_VCPKG="$HOME/cuda-kernel/vcpkg_installed/x64-linux/share/opencv4"
rm -rf build-vcpkg
cmake -B build-vcpkg -DCMAKE_BUILD_TYPE=Release \
      -DCUDACXX=/usr/local/cuda/bin/nvcc \
      -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
      -DOpenCV_DIR="$OpenCV_DIR_VCPKG" \
      -DCMAKE_CUDA_ARCHITECTURES="native"
cmake --build build-vcpkg -j 8

echo "==> [4/4] 冒烟测试（vs OpenCV-CUDA 逐像素比对）"
cd build-vcpkg && ./smoke

echo "==> 完成。跑分：cd build-vcpkg && ./benchmark"