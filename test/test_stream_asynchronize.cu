#include <iostream>
#include <cuda_runtime.h>

// 简单的 CUDA 核函数
__global__ void simpleKernel(int *d_data, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        d_data[idx] *= 2;
    }
}

int main() {
    const int N = 102400000;
    int *h_data = new int[N];
    int *d_data;

    // 初始化主机数据
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // 分配设备内存
    cudaMalloc((void**)&d_data, N * sizeof(int));

    // 创建 CUDA 流
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 异步将数据从主机复制到设备
    cudaMemcpyAsync(d_data, h_data, N * sizeof(int), cudaMemcpyHostToDevice, stream);

    // 定义网格和线程块维度
    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // 记录开始事件
    cudaEventRecord(start, stream);

    // 异步调用核函数
    simpleKernel<<<dimGrid, dimBlock, 0, stream>>>(d_data, N);

    // 记录结束事件
    cudaEventRecord(stop, stream);

    // CPU 可以继续执行其他任务，这里简单打印信息
    std::cout << "CPU is doing other tasks while the kernel is running..." << std::endl;

    // 可以在这里添加 CPU 执行的其他任务代码

    // 检查核函数是否完成
    cudaError_t err = cudaEventQuery(stop);
    if (err == cudaErrorNotReady) {
        std::cout << "Kernel is still running." << std::endl;
    } else if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "Kernel has finished." << std::endl;
    }

    // 异步将数据从设备复制回主机
    cudaMemcpyAsync(h_data, d_data, N * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // 等待流中的所有操作完成
    cudaStreamSynchronize(stream);

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // 销毁流
    cudaStreamDestroy(stream);

    // 释放设备内存
    cudaFree(d_data);

    // 释放主机内存
    delete[] h_data;

    return 0;
}