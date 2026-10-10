cmake_minimum_required(VERSION 3.18)
project(cuda_kernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 61)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xptxas -O3,-allow-expensive-optimizations=true")

find_package(CUDAToolkit REQUIRED)
# OpenCV-CUDA 安装位置：用 -DOpenCV_DIR=<path> 指定你的构建
# （需含 core/imgproc/cudaarithm/cudawarping/cudaimgproc/cudafilters，构建方法见 scripts/build_opencv_cuda.bat）
set(OpenCV_DIR "" CACHE PATH "OpenCV-CUDA install prefix")
if(NOT OpenCV_DIR)
    message(FATAL_ERROR "请用 -DOpenCV_DIR=<OpenCV-CUDA 安装目录> 指定（例：cmake -DOpenCV_DIR=D:/opencv-cuda-build/install ..）")
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc cudaarithm cudawarping cudaimgproc cudafilters)

add_library(gpuops STATIC
    src/integral.cu
    src/mean_filter.cu
    src/morphology.cu
    src/gaussian.cu
    src/histogram.cu
)
target_include_directories(gpuops PUBLIC src)
target_link_libraries(gpuops PUBLIC CUDA::cudart)

add_executable(benchmark bench/benchmark.cu)
target_link_libraries(benchmark PRIVATE gpuops ${OpenCV_LIBS})

# 冒烟测试：每算子小尺寸正确性
add_executable(smoke bench/smoke.cu)
target_link_libraries(smoke PRIVATE gpuops ${OpenCV_LIBS})
