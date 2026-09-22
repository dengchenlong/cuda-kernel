#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
#include <vector>
#include <device_launch_parameters.h>

#include <iostream>
#include "helper_timer.h"

// 啥也不干
__global__ void dilationFrameKernel1(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
}

// 只有赋值
__global__ void dilationFrameKernel2(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < width && y < height)
    {
        d_dst[step * y + x] = expand;
    }
}

// 只有循环
__global__ void dilationFrameKernel3(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int dx = (x > rw ? (x - rw) : 0); dx <= (x < width - rw ? (x + rw) : width - 1); ++dx)
    {
    }
}

// 条件赋值
__global__ void dilationFrameKernel4(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int dx = (x > rw ? (x - rw) : 0); dx <= (x < width - rw ? (x + rw) : width - 1); ++dx)
    {
        if (dx == (x < width - rw ? (x + rw) : width - 1))
            d_dst[step * y + x] = expand;
    }
}

// 条件赋值+删除分支
__global__ void dilationFrameKernel5(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    int min = x - rw;
    int max = x + rw;
    bool b = false;
    for (int dx = min; dx <= max; ++dx)
    {
        b = b || true;
    }
    d_dst[step * y + x] = expand;
}

// 基础实现
__global__ void dilationFrameKernel(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int dx = (x > rw ? (x - rw) : 0); dx <= (x < width - rw ? (x + rw) : width - 1); ++dx)
    {
        if (checkVal == d_src[step * y + dx])
        {
            d_dst[step * y + x] = expand;
            return;
        }
    }
}

// 基础实现+删除分支
__global__ void dilationFrameKernel7(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = 128 + threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x - 128 >= width || y >= height)
    {
        return;
    }

    for (int dx = x - rw; dx <= x + rw; ++dx)
    {
        if (checkVal == d_src[step * y + dx])
        {
            d_dst[step * y + x] = expand;
            return;
        }
    }
}

// 基础实现+删除分支+向量化访问
__global__ void dilationFrameKernel8(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = 128 + threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x - 128 >= width || y >= height)
    {
        return;
    }

    for (int dx = (x - rw) / 4 * 4; dx <= x + rw; dx += 4)
    {
        uchar4 current = *(uchar4*)(d_src + step * y + dx);
        if (checkVal == current.x || checkVal == current.y || checkVal == current.z || checkVal == current.w)
        {
            d_dst[step * y + x] = expand;
            return;
        }
    }
}

// 基础实现+删除赋值
__global__ void dilationFrameKernel9(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int dx = (x > rw ? (x - rw) : 0); dx <= (x < width - rw ? (x + rw) : width - 1); ++dx)
    {
        // (void)d_src[step * y + dx];
        // if (checkVal == d_src[step * y + dx])
        // {
        //     return;
        // }
    }
}

// 基础实现+删除赋值+删除分支
__global__ void dilationFrameKernel10(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    for (int dx = x - rw; dx <= x + rw; ++dx)
    {
        // (void)d_src[step * y + dx];
        // if (checkVal == d_src[step * y + dx])
        // {
        //     return;
        // }
    }
}

// 基础实现+删除赋值+向量化访问
__global__ void dilationFrameKernel11(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    const uchar4* src_vec = reinterpret_cast<const uchar4*>(d_src + step * y + 4 * x);
    // uchar4* dst_vec = reinterpret_cast<uchar4*>(d_dst + step * y + 4 * x);

    for (int dx = (x > rw ? (x - rw) / 4 * 4 : 0); dx <= (x < width - rw ? (x + rw) : width - 1); dx += 4)
    {
        uchar4 current = src_vec[dx / 4];
        // uint8_t* bytes = reinterpret_cast<uint8_t*>(&current);

        // if (*bytes == checkVal)
        // {
        //     return;
        // }
    }
}

cudaError_t dilationFrameCUDA(const uint8_t* d_src, uint8_t* d_dst, const int width, const int height, const int step, const int rw, const int rh, const uint8_t checkVal, const uint8_t expand)
{
    const dim3 blockSize{32, 32};
    const dim3 gridSize{(width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y};

    cudaError_t cudaStatus;

    // Launch a kernel on the GPU.
    dilationFrameKernel << <gridSize, blockSize >> > (d_src, d_dst, width, height, step, rw, rh, checkVal, expand);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "dilationFrameKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching dilationFrameKernel!\n", cudaStatus);
        return cudaStatus;
    }

    return cudaStatus;
}

// 基础实现
int testDilationFrame(const int width, const int height, const int r, const uint8_t checkVal, const uint8_t expand, const int repeat)
{
    int status = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    {
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return 1;
        }
    }

    // 准备数据
    uint8_t* d_src = nullptr, * d_dst = nullptr;
    size_t step;
    {
        std::vector<uint8_t> grayImage(width * height);
        cudaError_t cudaStatus = cudaMallocPitch((void**)(&d_src), &step, width, height);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMallocPitch((void**)(&d_dst), &step, width, height);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy2D(d_src, step, grayImage.data(), width, width, height, cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
    }

    // 预热
    for (size_t i = 0; i < 5; i++)
    {
        cudaError_t cudaStatus = dilationFrameCUDA(d_src, d_dst, width, height, step, r, r, checkVal, expand);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "dilationFrameCUDA failed!");
            status = 1;
            cudaFree(d_src);
            cudaFree(d_dst);
            return status;
        }
    }

    // 测试
    {
        StopWatchInterface* timer;
        sdkCreateTimer(&timer);
        for (size_t i = 0; i < repeat; i++)
        {
            sdkStartTimer(&timer);
            cudaError_t cudaStatus = dilationFrameCUDA(d_src, d_dst, width, height, step, r, r, checkVal, expand);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "dilationFrameCUDA failed!");
                status = 1;
                cudaFree(d_src);
                cudaFree(d_dst);
                return status;
            }
            sdkStopTimer(&timer);
        }
        const auto time = sdkGetAverageTimerValue(&timer);
        std::cout << "平均耗时: " << time << "ms\n";
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        status = 1;
        cudaFree(d_src);
        cudaFree(d_dst);
        return status;
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return status;
}

// 填充边界
int testDilationFrame2(const int width, const int height, const int r, const uint8_t checkVal, const uint8_t expand, const int repeat)
{
    int status = 0;

    // Choose which GPU to run on, change this on a multi-GPU system.
    {
        cudaError_t cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            return 1;
        }
    }

    // 准备数据
    uint8_t* d_src = nullptr, * d_dst = nullptr;
    size_t step;
    {
        std::vector<uint8_t> grayImage(width * height);
        cudaError_t cudaStatus = cudaMallocPitch((void**)(&d_src), &step, 128 + width + 128, height);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMallocPitch((void**)(&d_dst), &step, width, height);
        assert(cudaStatus == cudaSuccess);
        cudaStatus = cudaMemcpy2D(d_src + 128, step, grayImage.data(), width, width, height, cudaMemcpyHostToDevice);
        assert(cudaStatus == cudaSuccess);
    }

    // 预热
    for (size_t i = 0; i < 5; i++)
    {
        cudaError_t cudaStatus = dilationFrameCUDA(d_src, d_dst, width, height, step, r, r, checkVal, expand);
        if (cudaStatus != cudaSuccess)
        {
            fprintf(stderr, "dilationFrameCUDA failed!");
            status = 1;
            cudaFree(d_src);
            cudaFree(d_dst);
            return status;
        }
    }

    // 测试
    {
        StopWatchInterface* timer;
        sdkCreateTimer(&timer);
        for (size_t i = 0; i < repeat; i++)
        {
            sdkStartTimer(&timer);
            cudaError_t cudaStatus = dilationFrameCUDA(d_src, d_dst, width, height, step, r, r, checkVal, expand);
            if (cudaStatus != cudaSuccess)
            {
                fprintf(stderr, "dilationFrameCUDA failed!");
                status = 1;
                cudaFree(d_src);
                cudaFree(d_dst);
                return status;
            }
            sdkStopTimer(&timer);
        }
        const auto time = sdkGetAverageTimerValue(&timer);
        std::cout << "平均耗时: " << time << "ms\n";
    }

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaError_t cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceReset failed!");
        status = 1;
        cudaFree(d_src);
        cudaFree(d_dst);
        return status;
    }

    cudaFree(d_src);
    cudaFree(d_dst);

    return status;
}

int main()
{
    testDilationFrame(800, 2500, 15, 255, 255, 100);
}
