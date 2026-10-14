// probe1: hist 归因实验（spike 003）
// lose 项：hist-binary 640x480/1280x720/3840x2160 (0.92-0.96x), hist-gray 1280x720 (1.00x) / 3840x2160 (0.72x)
// 现版 gray fast path: 48 block × 256 线程 (32x8), 每行 32 线程步进, 4 byte/uint4, smem atomicAdd
// OpenCV calcHist: 什么结构？读源码后对照。先量化：不同 grid 大小 + 每线程多 uint4。
//
// 4K 时 mine 13us vs cv 10us：只差 30%。640x480 是 7us 平手。
// 变体：
//   v0 = 现版 (48 blk, 每线程 1 uint4/步)
//   v1 = grid 96/170 blk（SM 数 170）
//   v2 = 每线程 2 uint4/步（一次读 32B）
//   v3 = 每线程 4 uint4/步
//   v4 = 每线程 2 uint4 + block 256
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void hist_v(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                       int* __restrict__ partials, unsigned* __restrict__ counter,
                       int gridBlocks, int upto4)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) shist[tid] = 0;
    __syncthreads();
    const int cols4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = threadIdx.x; x < cols4; x += 32 * upto4) {
#pragma unroll
            for (int u4 = 0; u4 < upto4; ++u4) {
                const unsigned xi = x + u4 * 32;
                if (xi < cols4) {
                    const uint4 v = row4[xi];
                    const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
                    for (int u = 0; u < 4; ++u)
#pragma unroll
                        for (int b = 0; b < 32; b += 8)
                            atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
                }
            }
        }
    }
    __syncthreads();
    if (tid < 256) partials[blockIdx.x * 256 + tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridBlocks - 1);
        amLast = (ticket == gridBlocks - 1);
    }
    __syncthreads();
    if (amLast && tid < 256) {
        int s = 0;
        for (int b = 0; b < gridBlocks; ++b) s += partials[b * 256 + tid];
        reinterpret_cast<int*>(counter)[tid % 1] = s; // 不写 d_hist（省事，只测速度）
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU %s SM=%d\n", prop.name, prop.multiProcessorCount);

    int* partials; unsigned* counter;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));

    const int sizes[][2] = {{640,480},{1280,720},{1920,1080},{3840,2160}};
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);

        cvo::GpuMat dG(m), dHist;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

        printf("== %dx%d ==\n", W, H);
        // cv 计时
        {
            for (int i = 0; i < 8; ++i) cvo::calcHist(dG, dHist);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0); cvo::calcHist(dG, dHist); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
            printf("  cv            %8.4f ms\n", ms[15]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        // 变体
        struct { int blocks, upto4; const char* name; } vars[] = {
            {48, 1, "v0 48blk 1u4"}, {96, 1, "v1 96blk 1u4"}, {170, 1, "v1b 170blk"},
            {256, 1, "v1c 256blk"}, {48, 2, "v2 48blk 2u4"}, {96, 2, "v2b 96blk 2u4"},
            {48, 4, "v3 48blk 4u4"}, {96, 4, "v3b 96blk 4u4"}, {256, 2, "v4 256blk 2u4"},
        };
        for (auto& v : vars) {
            const dim3 t(32, 8);
            for (int i = 0; i < 8; ++i)
                hist_v<<<v.blocks, t>>>(s8, ss8, W, H, partials, counter, v.blocks, v.upto4);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0);
                hist_v<<<v.blocks, t>>>(s8, ss8, W, H, partials, counter, v.blocks, v.upto4);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float tt; cudaEventElapsedTime(&tt, e0, e1); ms.push_back(tt);
            }
            std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
            printf("  %-14s %8.4f ms\n", v.name, ms[15]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        cudaFree(s8);
    }
    return 0;
}
