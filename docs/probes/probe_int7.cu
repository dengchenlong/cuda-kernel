// probe_int7 (r52): ROWS(每屏障行数) × TH(chunk 高度) 全尺寸扫参 + 每实例寄存器数。
// probe_int6 结果：4K ROWS=2 0.4710 vs ROWS=1 0.5448（-13.5%），但 1080p ROWS=2 反而慢 19%。
// 本探针就是要搞清「1080p 回归」是寄存器/占用率导致的，还是 TH 与块数的耦合。
// 附带：每个实例打印 numRegs / maxThreadsPerBlock，便于直接判断占用率。
#include "ops.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

struct FSm { uint32_t wsum[4][8]; uint32_t wsum_init[8]; };

template <int ROWS>
__global__ void fused_v(const uint8_t* __restrict__ src, size_t sstep,
                        uint32_t* __restrict__ dst, size_t dstep,
                        const uint32_t* __restrict__ pre, int W, int H, int TH)
{
    __shared__ FSm sm;
    const int chunk = blockIdx.x;
    const int r0 = chunk * TH;
    if (r0 >= H) return;
    const int r1 = min(H, r0 + TH);
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c0 = tid * 16;
    const bool valid = (c0 < W);
    const unsigned ones = 0x01010101u;

    uint32_t acc[16];
    {
        uint32_t pv[16];
        uint32_t seg = 0;
        if (pre && valid) {
            const uint4* pp = (const uint4*)(pre + (size_t)chunk * W + c0);
            const uint4 a0 = pp[0], a1 = pp[1], a2 = pp[2], a3 = pp[3];
            pv[0] = a0.x;  pv[1] = a0.y;  pv[2] = a0.z;  pv[3] = a0.w;
            pv[4] = a1.x;  pv[5] = a1.y;  pv[6] = a1.z;  pv[7] = a1.w;
            pv[8] = a2.x;  pv[9] = a2.y;  pv[10] = a2.z; pv[11] = a2.w;
            pv[12] = a3.x; pv[13] = a3.y; pv[14] = a3.z; pv[15] = a3.w;
            const uint32_t q0 = pv[0] + pv[1] + pv[2] + pv[3];
            const uint32_t q1 = pv[4] + pv[5] + pv[6] + pv[7];
            const uint32_t q2 = pv[8] + pv[9] + pv[10] + pv[11];
            const uint32_t q3 = pv[12] + pv[13] + pv[14] + pv[15];
            seg = (q0 + q1) + (q2 + q3);
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) pv[k] = 0;
        }
        uint32_t inc = seg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        if (lane == 31) sm.wsum_init[warp] = inc;
        __syncthreads();
        uint32_t off = 0;
        for (int w = 0; w < warp; ++w) off += sm.wsum_init[w];
        uint32_t run = off + (inc - seg);
#pragma unroll
        for (int k = 0; k < 16; ++k) { run += pv[k]; acc[k] = run; }
        __syncthreads();
    }

    const uint8_t* p = src + (size_t)r0 * sstep + c0;
    uint32_t* q = (uint32_t*)((uint8_t*)dst + (size_t)r0 * dstep) + c0;

    for (int r = r0; r < r1; r += ROWS) {
        uint32_t o[ROWS][16];
        uint32_t seg[ROWS], inc[ROWS];
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            const bool ok = ((r + j) < r1) && valid;
            uint32_t s = 0;
            if (ok) {
                const uint4 v = *reinterpret_cast<const uint4*>(p + (size_t)j * sstep);
                o[j][0] = v.x & 0xffu;         o[j][1] = (v.x >> 8) & 0xffu;
                o[j][2] = (v.x >> 16) & 0xffu; o[j][3] = v.x >> 24;
                o[j][4] = v.y & 0xffu;         o[j][5] = (v.y >> 8) & 0xffu;
                o[j][6] = (v.y >> 16) & 0xffu; o[j][7] = v.y >> 24;
                o[j][8] = v.z & 0xffu;         o[j][9] = (v.z >> 8) & 0xffu;
                o[j][10] = (v.z >> 16) & 0xffu; o[j][11] = v.z >> 24;
                o[j][12] = v.w & 0xffu;        o[j][13] = (v.w >> 8) & 0xffu;
                o[j][14] = (v.w >> 16) & 0xffu; o[j][15] = v.w >> 24;
                s = __dp4a(v.x, ones, __dp4a(v.y, ones, __dp4a(v.z, ones, __dp4a(v.w, ones, 0u))));
            } else {
#pragma unroll
                for (int k = 0; k < 16; ++k) o[j][k] = 0;
            }
            seg[j] = s;
            uint32_t t = s;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                const uint32_t u = __shfl_up_sync(0xffffffffu, t, d);
                if (lane >= d) t += u;
            }
            inc[j] = t;
        }
#pragma unroll
        for (int j = 0; j < ROWS; ++j)
            if (lane == 31) sm.wsum[j][warp] = inc[j];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            uint32_t off = 0;
            for (int w = 0; w < warp; ++w) off += sm.wsum[j][w];
            uint32_t run = off + (inc[j] - seg[j]);
            if ((r + j) < r1 && valid) {
#pragma unroll
                for (int k = 0; k < 16; ++k) { run += o[j][k]; acc[k] += run; }
                uint32_t* qq = (uint32_t*)((uint8_t*)q + (size_t)j * dstep);
                *reinterpret_cast<uint4*>(qq)      = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
                *reinterpret_cast<uint4*>(qq + 4)  = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
                *reinterpret_cast<uint4*>(qq + 8)  = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
                *reinterpret_cast<uint4*>(qq + 12) = make_uint4(acc[12], acc[13], acc[14], acc[15]);
            }
        }
        p += (size_t)ROWS * sstep;
        q = (uint32_t*)((uint8_t*)q + (size_t)ROWS * dstep);
    }
}

__global__ void chunk_sum_k(const uint8_t* __restrict__ src, size_t sstep,
                            uint32_t* __restrict__ csum, int W, int H, int TH)
{
    const int ngroups = W >> 4;
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ((H + TH - 1) / TH) * ngroups;
    if (gid >= total) return;
    const int chunk = gid / ngroups;
    const int c0 = (gid - chunk * ngroups) * 16;
    const int r0 = chunk * TH;
    const int r1 = min(H, r0 + TH);
    uint32_t acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0;
    const uint8_t* p = src + (size_t)r0 * sstep + c0;
    for (int r = r0; r < r1; ++r) {
        union { uint4 v; uint8_t b[16]; } u;
        u.v = *reinterpret_cast<const uint4*>(p);
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] += u.b[k];
        p += sstep;
    }
    uint4* op = (uint4*)(csum + (size_t)chunk * W + c0);
    op[0] = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
    op[1] = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
    op[2] = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
    op[3] = make_uint4(acc[12], acc[13], acc[14], acc[15]);
}

__global__ void chunk_scan_k(uint32_t* __restrict__ csum, int W, int nChunks)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    uint32_t s = 0;
    for (int k = 0; k < nChunks; ++k) {
        uint32_t* pp = csum + (size_t)k * W + c;
        const uint32_t v = *pp;
        *pp = s;
        s += v;
    }
}

inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

double median_run(const std::function<void()>& f, int warmup, int reps)
{
    for (int i = 0; i < warmup; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms(reps);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(s); f(); cudaEventRecord(e);
        cudaEventSynchronize(e); cudaEventElapsedTime(&ms[i], s, e);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    std::nth_element(ms.begin(), ms.begin() + reps / 2, ms.end());
    return ms[reps / 2];
}

} // namespace

int main()
{
    CUDA_CHECK(cudaFree(0));
    // 寄存器/占用率事实（与运行结果对照用）
    cudaFuncAttributes at{};
    cudaFuncGetAttributes(&at, fused_v<1>); printf("fused R1: regs=%d maxThreads=%d\n", at.numRegs, at.maxThreadsPerBlock);
    cudaFuncGetAttributes(&at, fused_v<2>); printf("fused R2: regs=%d maxThreads=%d\n", at.numRegs, at.maxThreadsPerBlock);
    cudaFuncGetAttributes(&at, fused_v<4>); printf("fused R4: regs=%d maxThreads=%d\n", at.numRegs, at.maxThreadsPerBlock);

    const int sizes[][2] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int THs[] = {0, 64, 128, 256};      // 0 = 自适应 fused_th
    const int ROWSs[] = {1, 2, 4};

    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t sstep = 0;
        uint32_t* dst0 = nullptr; size_t dstep = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &sstep, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&dst0, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(11);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, sstep, host.data(), W, W, H, cudaMemcpyHostToDevice));

        const int maxtc = idiv_up(H, 32) + 2;
        uint32_t* pre = nullptr;
        CUDA_CHECK(cudaMalloc(&pre, (size_t)maxtc * W * sizeof(uint32_t)));

        const int tw = (W + 15) >> 4;
        const int fth = ((tw + 31) >> 5) << 5;
        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);

        printf("\n=== %dx%d (fth=%d) ===\n", W, H, fth);
        for (int th : THs) {
            const int TH = th ? th : ([] (int h) { int t = (h + 63) / 64; t = ((t + 31) / 32) * 32; return t < 32 ? 32 : (t > 256 ? 256 : t); }(H));
            const int nChunks = idiv_up(H, TH);
            auto stage = [&]() {
                const int total = nChunks * (W >> 4);
                chunk_sum_k<<<idiv_up(total, 256), 256>>>(src, sstep, pre, W, H, TH);
                chunk_scan_k<<<idiv_up(W, 256), 256>>>(pre, W, nChunks);
            };
            for (int r : ROWSs) {
                auto launch = [&]() {
                    if (r == 1)      fused_v<1><<<nChunks, fth>>>(src, sstep, dst0, dstep, pre, W, H, TH);
                    else if (r == 2) fused_v<2><<<nChunks, fth>>>(src, sstep, dst0, dstep, pre, W, H, TH);
                    else             fused_v<4><<<nChunks, fth>>>(src, sstep, dst0, dstep, pre, W, H, TH);
                };
                stage(); launch(); CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaGetLastError());
                double best = 1e30;
                for (int round = 0; round < 3; ++round) {
                    stage();
                    best = std::min(best, median_run([&] { launch(); }, warmup, reps));
                }
                printf("  TH=%-4d nC=%-4d ROWS=%d  kernel=%7.4f ms  %6.1f GB/s(5N)\n",
                       TH, nChunks, r, best, 5.0 * W * H / 1e6 / (best * 1e-3) / 1000.0);
            }
        }
        cudaFree(src); cudaFree(dst0); cudaFree(pre);
    }
    return 0;
}
