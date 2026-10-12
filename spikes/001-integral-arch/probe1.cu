// probe1: integral 归因实验（spike 001）
// 三种实现 vs OpenCV，1 个文件全包含：
//   v0 = 项目现版（integral.cu 的 96-warp 行扫描 + 列 carry）
//   v1 = rows 固定 launch shape 探针：只改 Pass1 的 grid/warp 数（96 -> 512/1024 warps）
//   v2 = 单 kernel fused：每 block 负责一个 (H/warps) 行带，行前缀 + 列前缀一次做完（W ≤ 4096 用一行 smem carry 不现实，故 v2 采用 “每 block 一行” 的纯行内扫描 + 列扫描由后续 chunk 内核完成的现版结构）
// 实际可归因的对照是 v0 vs v1（launch shape）和 v0 vs cv（吞吐差）。
// 输出各尺寸 mine(v0)/mine(v1)/opencv ms。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include "ops.h"
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;

// ---- 现版 Pass1（原样拷自 integral.cu，隔离编译避免改名）----
__global__ void p1_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                               uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    constexpr int SEG_BYTES = 512;
    const bool vecOK = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);

    for (int y = warp; y < H; y += nwarps) {
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
}

// ---- v2: 行遍改“每 block 一行 + 全部 128 线程扫一行”的宽行版本 ----
// 现 Pass1 每 warp 串行扫 W/512 段（4K 时 8 段，串行依赖 carry）；
// v2 一个 block(256 线程=8 warp) 协作扫一行：每 warp 仍 SEG 512B，但 warp 数随 block 走，
// 段间 carry 经 smem 传递 —— 只是把“warp 网格”收紧为“block 网格”，减少全局调度。
__global__ void p1_rows_v2_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    __shared__ uint32_t scarry[8];          // 8 warp 的段间 carry（warp0..7 各自行内同列段共享？否——每 warp 处理不同段）
    // 修正设计：一行由一个 block 处理，8 个 warp 每人一段一轮，段内依赖前段 carry。
    // 为保序，仍然 warp0 串行处理段 0,8,16...；warp1 处理段 1,9,16... —— 段间依赖跨 warp，
    // 需要全局顺序。改成：每 warp 独占一行（block 8 行），保留 carry 串行 —— 与现版相同但
    // 以 block 为调度单位，grid = H/8。
    const int rowBase = blockIdx.x * 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int SEG_BYTES = 512;
    const bool vecOK = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);

    const int y = rowBase + warp;
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

// ---- 现版 Pass2（原样）----
__global__ void p2_cols_kernel(const uint32_t* __restrict__ in, size_t istep,
                               uint32_t* __restrict__ out, size_t ostep,
                               uint32_t* __restrict__ carry, int W, int H, int nChunks)
{
    constexpr int CHUNK_H = 512;
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
__global__ void p2_carry_kernel(uint32_t* __restrict__ out, size_t ostep,
                                const uint32_t* __restrict__ carry, int W, int H)
{
    constexpr int CHUNK_H = 512;
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

static void run_v0(const uint8_t* s, size_t ss, uint32_t* d, size_t ds, int W, int H, uint32_t* carry)
{
    (void)s; (void)ss; (void)d; (void)ds; (void)W; (void)H; (void)carry;   // 未用（见 run_v0_probe）
}
// 用 probe 自己的 p1 内核当 v0（逐指令一致），避免链接 ops 库的符号耦合
static void run_v0_probe(const uint8_t* s, size_t ss, uint32_t* d, size_t ds, int W, int H, uint32_t* carry, cudaStream_t st)
{
    p1_rows_kernel<<<12, 256, 0, st>>>(s, ss, d, ds, W, H);
    const int nChunks = (H + 511) / 512;
    if (nChunks == 1) {
        p2_cols_kernel<<<dim3((W + 511) / 512), 128, 0, st>>>(d, ds, d, ds, nullptr, W, H, 1);
    } else {
        p2_cols_kernel<<<dim3((W + 511) / 512, nChunks), 128, 0, st>>>(d, ds, d, ds, carry, W, H, nChunks);
        p2_carry_kernel<<<dim3((W + 511) / 512, nChunks - 1), 128, 0, st>>>(d, ds, carry, W, H);
    }
}
static void run_v1(const uint8_t* s, size_t ss, uint32_t* d, size_t ds, int W, int H, uint32_t* carry, cudaStream_t st, int warps)
{
    p1_rows_kernel<<<(warps * 32) / 256, 256, 0, st>>>(s, ss, d, ds, W, H);
    const int nChunks = (H + 511) / 512;
    if (nChunks == 1) {
        p2_cols_kernel<<<dim3((W + 511) / 512), 128, 0, st>>>(d, ds, d, ds, nullptr, W, H, 1);
    } else {
        p2_cols_kernel<<<dim3((W + 511) / 512, nChunks), 128, 0, st>>>(d, ds, d, ds, carry, W, H, nChunks);
        p2_carry_kernel<<<dim3((W + 511) / 512, nChunks - 1), 128, 0, st>>>(d, ds, carry, W, H);
    }
}
static void run_v2(const uint8_t* s, size_t ss, uint32_t* d, size_t ds, int W, int H, uint32_t* carry, cudaStream_t st)
{
    p1_rows_v2_kernel<<<(H + 7) / 8, 256, 0, st>>>(s, ss, d, ds, W, H);
    const int nChunks = (H + 511) / 512;
    if (nChunks == 1) {
        p2_cols_kernel<<<dim3((W + 511) / 512), 128, 0, st>>>(d, ds, d, ds, nullptr, W, H, 1);
    } else {
        p2_cols_kernel<<<dim3((W + 511) / 512, nChunks), 128, 0, st>>>(d, ds, d, ds, carry, W, H, nChunks);
        p2_carry_kernel<<<dim3((W + 511) / 512, nChunks - 1), 128, 0, st>>>(d, ds, carry, W, H);
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s SM=%d CC=%d.%d\n", prop.name, prop.multiProcessorCount, prop.major, prop.minor);

    const int sizes[][2] = {{1280,720},{1920,1080},{3840,2160}};
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> d(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)d(rng);

        cvo::GpuMat dG(m), dOut;
        cvo::integral(dG, dOut);
        cv::Mat cvr; dOut.download(cvr);

        uint8_t* s; size_t ss;
        CUDA_CHECK(cudaMallocPitch(&s, &ss, W, H));
        CUDA_CHECK(cudaMemcpy2D(s, ss, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        uint32_t* dst; size_t ds;
        CUDA_CHECK(cudaMallocPitch(&dst, &ds, (size_t)W * 4, H));
        const int nChunks = (H + 511) / 512;
        uint32_t* carry = nullptr;
        if (nChunks > 1) CUDA_CHECK(cudaMalloc(&carry, (size_t)nChunks * W * 4));

        auto bench = [&](const char* name, auto fn) {
            for (int i = 0; i < 5; ++i) fn();
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0); fn(); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
            printf("  %-12s %8.4f ms\n", name, ms[15]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        };

        printf("%dx%d:\n", W, H);
        bench("v0(12blk256)", [&]{ run_v0_probe(s, ss, dst, ds, W, H, carry, 0); });
        bench("v1(64blk256)", [&]{ run_v1(s, ss, dst, ds, W, H, carry, 0, 64); });
        bench("v1(128blk256)",[&]{ run_v1(s, ss, dst, ds, W, H, carry, 0, 128); });
        bench("v1(256blk256)",[&]{ run_v1(s, ss, dst, ds, W, H, carry, 0, 256); });
        bench("v2(1blk=8row)",[&]{ run_v2(s, ss, dst, ds, W, H, carry, 0); });

        // 正确性：v2 vs OpenCV
        CUDA_CHECK(cudaMemset2D(dst, ds, 0, W * 4, H));
        run_v2(s, ss, dst, ds, W, H, carry, 0);
        cv::Mat mine(H, W, CV_32SC1);
        CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dst, ds, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        cv::Mat cv32; cvr.convertTo(cv32, CV_32SC1);
        int bad = 0;
        for (int y = 0; y < H && bad < 5; ++y)
            for (int x = 0; x < W; ++x)
                if (mine.at<int>(y, x) != cv32.at<int>(y, x)) { ++bad; break; }
        printf("  v2-vs-cv bad rows: %d\n", bad);
        printf("  opencv: "); // OpenCV 单独计时
        {
            for (int i = 0; i < 5; ++i) cvo::integral(dG, dOut);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 30; ++i) {
                cudaEventRecord(e0); cvo::integral(dG, dOut); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
            printf("%8.4f ms\n", ms[15]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        cudaFree(s); cudaFree(dst); if (carry) cudaFree(carry);
    }
    return 0;
}
