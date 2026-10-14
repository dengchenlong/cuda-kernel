// probe4: hist-gray warp 级方案对比（spike 003b）
// v0 = 现版 per-byte smem atomicAdd（32x8 block, grid-stride 行, uint4）
// v1 = warp-aggregated: __match_any_sync 同 bin lane 合并, leader 单次 atomicAdd
// v2 = per-lane 私有 u16 计数 + 归约（128 线程/块, 64KB 动态 smem）
// 对照 cv（多轮取最快）+ 主口径前奏。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

// ---------- v0: 现版 ----------
__global__ void hist_v0(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
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
        // 对照探针：d_hist 覆写省略（各方案相同代价）
    }
}

// ---------- v1: warp-aggregated ----------
__global__ void hist_v1(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                        int* __restrict__ partials, unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    if (tid < 256) shist[tid] = 0;
    __syncthreads();
    const int cols4 = W >> 4;
    const int lane = threadIdx.x & 31;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int b = 0; b < 32; b += 8) {
                    const unsigned bin = (w[u] >> b) & 0xff;
                    // warp-aggregated（NVIDIA 官方写法）：__activemask 而非 0xffffffff，
                    // 行尾部分活跃的 warp 不会死锁（0xffffffff 会等 inactive lane → hang）
                    const unsigned mask = __match_any_sync(__activemask(), bin);
                    const int leader = __ffs(mask) - 1;
                    const int lane = threadIdx.x & 31;
                    if (lane == leader) atomicAdd(&shist[bin], __popc(mask));
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
    }
}

// ---------- v2: per-lane 私有 u16 计数 ----------
// 128 线程/块, 每 lane 256 u16, smem 128*256*2 = 64KB（需动态 smem opt-in）
__global__ void hist_v2(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                        int* __restrict__ partials, unsigned* __restrict__ counter, int gridBlocks)
{
    extern __shared__ uint16_t laneHist[];   // 128 * 256 u16
    __shared__ bool amLast;
    const int tid = threadIdx.x;             // 128 线程
    uint16_t* my = laneHist + tid * 256;     // 注：lane 间同 bin 相差 128 words → 4 路 bank 冲突
    for (int i = 0; i < 256; ++i) my[i] = 0;
    __syncthreads();

    const int cols4 = W >> 4;
    const int laneInWarp = tid & 31;
    const int warp = tid >> 5;
    for (int y = blockIdx.x * 4 + warp; y < H; y += gridBlocks * 4) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = laneInWarp; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    ++my[(w[u] >> b) & 0xff];
            }
        }
    }
    __syncthreads();
    if (tid < 128) {
        for (int bin = tid; bin < 256; bin += 128) {
            int s = 0;
            for (int l = 0; l < 128; ++l) s += laneHist[l * 256 + bin];
            partials[blockIdx.x * 256 + bin] = s;
        }
    }
    __syncthreads();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridBlocks - 1);
        amLast = (ticket == gridBlocks - 1);
    }
    __syncthreads();
    if (amLast && tid == 0) {
        int total = 0;
        for (int b = 0; b < 256; ++b) {
            int s = 0;
            for (int blk = 0; blk < gridBlocks; ++blk) s += partials[blk * 256 + b];
            total += s;
        }
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaFuncSetAttribute(hist_v2, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024));

    int* partials; unsigned* counter;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));

    for (auto& sz : {std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        printf("== preparing %dx%d ==\n", W, H); fflush(stdout);
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
        cvo::GpuMat dG(m), dHist;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

        // 主口径前奏
        cvo::integral(dG, dHist);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  prelude: integral done\n"); fflush(stdout);
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dHist);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  prelude: box done\n"); fflush(stdout);
        for (int k : {5, 7, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 20; ++i) f->apply(dG, dHist);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("  prelude: gaussian done\n"); fflush(stdout);

        printf("== %dx%d ==\n", W, H);
        fflush(stdout);
        const int blocks = std::min(170, std::max(48, (H + 7) / 8));
        const dim3 t(32, 8);

        auto bench = [&](const char* name, auto fn) {
            for (int i = 0; i < 6; ++i) fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0); fn(); cudaEventRecord(e1); cudaEventSynchronize(e1);
                float tt; cudaEventElapsedTime(&tt, e0, e1); ms.push_back(tt);
            }
            std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
            printf("  %-22s med=%.4f\n", name, ms[15]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        };

        // cv 多轮取最快（快路径口径）；独立输出（calcHist 输出 256x1，勿复用 integral 的大 buffer）
        cvo::GpuMat dHistOut;
        float cvBest = 1e9f;
        for (int round = 0; round < 3; ++round) {
            for (int i = 0; i < 20; ++i) cvo::calcHist(dG, dHistOut);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0); cvo::calcHist(dG, dHist); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float tt; cudaEventElapsedTime(&tt, e0, e1); cvBest = std::min(cvBest, tt);
            }
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        printf("  %-22s best=%.4f\n", "cv(best-of-rounds)", cvBest);

        bench("v0 atomic", [&] {
            hist_v0<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks);
        });
        printf("  (v0 timed)\n"); fflush(stdout);
        bench("v1 match_any agg", [&] {
            hist_v1<<<blocks, t>>>(s8, ss8, W, H, partials, counter, blocks);
        });
        printf("  (v1 timed)\n"); fflush(stdout);
        bench("v2 per-lane priv", [&] {
            hist_v2<<<blocks, 128, 128 * 256 * 2>>>(s8, ss8, W, H, partials, counter, blocks);
        });
        printf("  (v2 timed)\n"); fflush(stdout);
        cudaFree(s8);
    }
    return 0;
}
