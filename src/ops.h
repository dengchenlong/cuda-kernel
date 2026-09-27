#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                        \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============ 积分图 ============
// src: CV_8UC1 (H x W, step bytes), dst: CV_32SC1 (H x W, dstep bytes), 原地 inclusive
void gpu_integral(const uint8_t* d_src, size_t srcStep, uint32_t* d_dst, size_t dstStep,
                  int width, int height, cudaStream_t stream = 0);

// ============ 均值滤波 (box, 核 kw x kh, anchor=中心, 边界 replicate) ============
// src/dst: CV_8UC1。内部用分离滑动窗口，O(1)/像素，与核大小无关
void gpu_mean(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
              int width, int height, int kw, int kh, cudaStream_t stream = 0);

// ============ 形态学腐蚀/膨胀 (任意核矩形, anchor=中心, 边界=OpenCV morphologyDefaultBorderValue) ============
// src/dst: CV_8UC1（二值图 0/255 同样走这条路）
void gpu_erode(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
               int width, int height, int kw, int kh, cudaStream_t stream = 0);
void gpu_dilate(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                int width, int height, int kw, int kh, cudaStream_t stream = 0);

// ============ 高斯滤波 (核尺寸 ksize 必须为奇数, sigma<=0 时按 OpenCV 规则推算) ============
// src/dst: CV_8UC1, 边界 BORDER_REFLECT_101 (OpenCV 默认)
void gpu_gaussian(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int ksize, float sigma, cudaStream_t stream = 0);

// ============ 灰度直方图 (256 bins, 输出 int[256]) ============
// last-block 归约实现：全 256 bin 完全覆写，调用前无需清零（stream 内多次调用安全）
void gpu_histogram(const uint8_t* d_src, size_t srcStep,
                   int width, int height, int* d_hist, cudaStream_t stream = 0);

// 二值图(0/255)专用直方图：全 256 bin 完全覆写（bin0=0 的个数、bin255=255 的个数、其余恒 0），
// 调用前无需清零
void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream = 0);
