// probe2: hist 4K 复测 + integral 1080p 复测
// 1) hist-gray 4K：block 48/96/170/256 在"benchmark 同口径"（前面有其它 op）下的时间
// 2) integral 1080p：v2 launch (H+7)/8 vs OpenCV 快路径（本轮先热身拿 cv 快值）
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void hist_v(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                       int* __restrict__ partials, unsigned* __restrict__ counter, int gridBlocks)
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
    }
}

// integral v2 pass1 (落地版)
__global__ void p1_kernel(const uint8_t* __restrict__ src, size_t sstep,
                          uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 512;
    const bool vecOK = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;
    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;
        uint32_t vals[16]; uint32_t partial = 0;
        if (my0 + 16 <= W && vecOK) {
            union { uint4 v; uint8_t b[16]; } u;
            u.v = *reinterpret_cast<const uint4*>(srow + my0);
#pragma unroll
            for (int k = 0; k < 16; ++k) { vals[k] = u.b[k]; partial += vals[k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int x = my0 + k; uint32_t v = 0;
                if (x < W) v = srow[x];
                vals[k] = v; partial += v;
            }
        }
        uint32_t inc = partial;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        const uint32_t excl = inc - partial + carry;
        carry += __shfl_sync(0xffffffffu, inc, 31);
        uint32_t run = excl;
#pragma unroll
        for (int k = 0; k < 16; ++k) { run += vals[k]; const int x = my0 + k; if (x < W) drow[x] = run; }
    }
}
constexpr int CHUNK_H = 512;
__global__ void p2_kernel(const uint32_t* __restrict__ in, size_t istep,
                          uint32_t* __restrict__ out, size_t ostep,
                          uint32_t* __restrict__ carry, int W, int H, int nChunks)
{
    const int chunk = blockIdx.y;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;
    const uint4* ip = (const uint4*)((const uint8_t*)in  + (size_t)r0 * istep) + c0 / 4;
    uint4*       op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;
    uint4 car = make_uint4(0,0,0,0);
    for (int r = r0; r < r1; ++r) {
        uint4 v = *ip;
        v.x += car.x; v.y += car.y; v.z += car.z; v.w += car.w;
        *op = v; car = v;
        ip = (const uint4*)((const uint8_t*)ip + istep);
        op = (uint4*)((uint8_t*)op + ostep);
    }
    if (nChunks > 1) {
        uint32_t* cr = carry + (size_t)chunk * W + c0;
        cr[0] = car.x; cr[1] = car.y; cr[2] = car.z; cr[3] = car.w;
    }
}
__global__ void p2c_kernel(uint32_t* __restrict__ out, size_t ostep,
                           const uint32_t* __restrict__ carry, int W, int H)
{
    const int chunk = blockIdx.y + 1;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;
    uint4 exc = make_uint4(0,0,0,0);
    for (int j = 0; j < chunk; ++j) {
        const uint32_t* cr = carry + (size_t)j * W + c0;
        exc.x += cr[0]; exc.y += cr[1]; exc.z += cr[2]; exc.w += cr[3];
    }
    uint4* op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;
    for (int r = r0; r < r1; ++r) {
        uint4 v = *op;
        v.x += exc.x; v.y += exc.y; v.z += exc.z; v.w += exc.w;
        *op = v;
        op = (uint4*)((uint8_t*)op + ostep);
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    int* partials; unsigned* counter;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));

    // ====== 1) integral 1080p / 4K ======
    for (auto& sz : {std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
        cvo::GpuMat dG(m), dOut;

        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        uint32_t* dst; size_t ds;
        CUDA_CHECK(cudaMallocPitch(&dst, &ds, (size_t)W * 4, H));
        const int nChunks = (H + 511) / 512;
        uint32_t* carry = nullptr;
        if (nChunks > 1) CUDA_CHECK(cudaMalloc(&carry, (size_t)nChunks * W * 4));

        auto mine = [&] {
            p1_kernel<<<(H + 7) / 8, 256>>>(s8, ss8, dst, ds, W, H);
            if (nChunks == 1)
                p2_kernel<<<dim3((W + 511) / 512), 128>>>(dst, ds, dst, ds, nullptr, W, H, 1);
            else {
                p2_kernel<<<dim3((W + 511) / 512, nChunks), 128>>>(dst, ds, dst, ds, carry, W, H, nChunks);
                p2c_kernel<<<dim3((W + 511) / 512, nChunks - 1), 128>>>(dst, ds, carry, W, H);
            }
        };
        for (int i = 0; i < 8; ++i) { mine(); cvo::integral(dG, dOut); }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> msv, cvv;
        for (int i = 0; i < 40; ++i) {
            cudaEventRecord(e0); mine(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); msv.push_back(t);
            cudaEventRecord(e0); cvo::integral(dG, dOut); cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t, e0, e1); cvv.push_back(t);
        }
        std::sort(msv.begin(), msv.end()); std::sort(cvv.begin(), cvv.end());
        printf("integral %dx%d: mine med=%.4f min=%.4f | cv med=%.4f min=%.4f\n",
               W, H, msv[20], msv[0], cvv[20], cvv[0]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaFree(s8); cudaFree(dst); if (carry) cudaFree(carry);
    }

    // ====== 2) hist-gray 4K grid 扫描 ======
    {
        const int W = 3840, H = 2160;
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
        cvo::GpuMat dG(m), dHist;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

        // 状态模拟：先跑一轮 f32 morph 前奏? 简化——只测 cv+hist 交错
        for (int i = 0; i < 8; ++i) cvo::calcHist(dG, dHist);
        const dim3 t(32, 8);
        for (int blocks : {48, 96, 128, 170, 216, 256}) {
            for (int i = 0; i < 8; ++i)
                hist_v<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 40; ++i) {
                cudaEventRecord(e0);
                hist_v<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float tt; cudaEventElapsedTime(&tt, e0, e1); ms.push_back(tt);
            }
            std::sort(ms.begin(), ms.end());
            printf("hist4K %3dblk: med=%.4f min=%.4f\n", blocks, ms[20], ms[0]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        // cv 对照（多轮找快路径）
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> cvv;
        for (int r = 0; r < 5; ++r) {
            float best = 1e9;
            for (int i = 0; i < 20; ++i) {
                cudaEventRecord(e0); cvo::calcHist(dG, dHist); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float tt; cudaEventElapsedTime(&tt, e0, e1); best = std::min(best, tt);
            }
            cvv.push_back(best);
        }
        std::sort(cvv.begin(), cvv.end());
        printf("hist4K cv best-of-5-min: %.4f\n", cvv[0]);
        cudaFree(s8);
    }
    return 0;
}
