cmake_minimum_required(VERSION 3.10)

# 设置项目名称
project(CUDAKernel LANGUAGES CXX CUDA)

# 查找 CUDA 工具包
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置 CUDA 架构
set(CMAKE_CUDA_ARCHITECTURES 61)

# 可选：添加调试标志
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
endif()

# 启用测试支持
include(CTest)
enable_testing()

# 添加子目录
add_subdirectory(src)
add_subdirectory(test)
