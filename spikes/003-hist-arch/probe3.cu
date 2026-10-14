// probe3: 复刻主 benchmark 的 hist 口径（含前面 f32 op 序列），扫描 grid 上限
// 找 4K hist-gray 在真实 benchmark 环境里的最优 grid。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err\n"); exit(1); } } while (0)

__global__ void hist_v(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                       int* __restrict__ partials, unsigned* __restrict__ counter,
                       int gridBlocks, int* __restrict__ d_hist)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) shist[tid] = 0;
    __syncthreads();
    const int cols4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
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
        d_hist[tid] = s;
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int W = 3840, H = 2160;
    cv::Mat m(H, W, CV_8UC1);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dd(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
    cvo::GpuMat dG(m), dOut;
    int* partials; unsigned* counter; int* d_hist;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMalloc(&d_hist, 256 * 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));

    uint8_t* s8; size_t ss8;
    CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
    CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

    // 主 benchmark 前奏：integral + mean + morph + f32 组 + gaussian（全部跑一遍）
    cvo::integral(dG, dOut);
    for (int k : {3, 15, 31}) {
        auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
        for (int i = 0; i < 30; ++i) f->apply(dG, dOut);
    }
    for (int k : {3, 15, 31}) {
        cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker);
        auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, ker);
        for (int i = 0; i < 30; ++i) { fe->apply(dG, dOut); fd->apply(dG, dOut); }
    }
    for (int k : {5, 7, 15}) {
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
        for (int i = 0; i < 30; ++i) f->apply(dG, dOut);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计时：hist 各 grid（复刻主口径：每轮 30 次取中位）
    for (int blocks : {48, 96, 128, 170, 216, 256}) {
        const dim3 t(32, 8);
        for (int i = 0; i < 6; ++i)
            hist_v<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks, d_hist);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms;
        for (int i = 0; i < 30; ++i) {
            cudaEventRecord(e0);
            hist_v<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks, d_hist);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float tt; cudaEventElapsedTime(&tt, e0, e1); ms.push_back(tt);
        }
        std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
        printf("hist4K(主口径) %3dblk: med=%.4f\n", blocks, ms[15]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    // cv 对照（主口径同环境）
    {
        for (int i = 0; i < 6; ++i) cvo::calcHist(dG, dOut);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms;
        for (int i = 0; i < 30; ++i) {
            cudaEventRecord(e0); cvo::calcHist(dG, dOut); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float tt; cudaEventElapsedTime(&tt, e0, e1); ms.push_back(tt);
        }
        std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
        printf("hist4K(主口径) cv: med=%.4f\n", ms[15]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}
