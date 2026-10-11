#!/usr/bin/env bash
# Linux 服务器：用 vcpkg 安装 OpenCV-CUDA 并构建/验证 cuda-kernel
# 用法： 在服务器终端执行  ./scripts/build_linux.sh
# 说明： 编译 OpenCV-CUDA 需 30~60 分钟；请在你的终端（非 Hermes 后台）运行，
#        因为 Hermes 的后台进程管理会对长时多进程编译发 SIGTERM。
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export VCPKG_FORCE_SYSTEM_BINARIES=1
export VCPKG_DISABLE_METRICS=1
VCPKG="$HOME/vcpkg/vcpkg"

echo "==> [1/4] 准备 vcpkg（如已存在则复用）"
if [ ! -x "$VCPKG" ]; then
  if [ ! -d "$HOME/vcpkg" ]; then
    git clone https://github.com/microsoft/vcpkg.git "$HOME/vcpkg"
  fi
  (cd "$HOME/vcpkg" && ./bootstrap-vcpkg.sh -disableMetrics)
fi

echo "==> [2/4] 安装 OpenCV-CUDA（manifest 精简：opencv4[cuda]，约 30~60 分钟）"
cd "$(dirname "$0")/.."     # 回到仓库根（manifest vcpkg.json 所在）
"$VCPKG" install

echo "==> [3/4] 配置并构建 cuda-kernel（OpenCV 走 vcpkg toolchain）"
rm -rf build-vcpkg
cmake -B build-vcpkg -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE="$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" \
      -DCMAKE_CUDA_ARCHITECTURES="native"
cmake --build build-vcpkg -j

echo "==> [4/4] 冒烟测试（vs OpenCV-CUDA 逐像素比对）"
cd build-vcpkg && ./smoke

echo "==> 完成。跑分：cd build-vcpkg && ./benchmark"