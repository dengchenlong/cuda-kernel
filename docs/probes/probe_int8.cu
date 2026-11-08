// probe_int8 (r52): 定位 integral fused 内核 65~76 GB/s -> 93 GB/s 的缺口。
// probe_int6/7 已确认：4K TH=64 ROWS=2 比 ROWS=1 快 ~12.7%（两个独立进程复现），但 1080p 变慢。
// 本探针回答「障碍到底是什么」，用 5 轮交错取 min（r49 时钟漂移教训）。
//   v0 R1       : 现状（每行 1 次屏障 + 跨 warp smem 广播）
//   v1 R2       : 两行共享一次屏障（84 regs）
//   v2 NOBAR-R1 : 只做 warp 局部扫描，不写 smem / 不屏障 / 不加跨 warp 偏移（**结果故意错**，纯计时）
//   v3 NOBAR-R2 : 同上 + 两行
//   v4 PF-R1    : 预取下一行 uint4 到寄存器（+4 regs），屏障后不再等内存
//   v5 STCS-R1  : dst 写用 __stcs（流式，不占 L2）
//   v6 R1-TH64  : 1080p 上把 TH 从 32 改成 64（对照）
// 正确性：v0/v1/v6 三次运行互相逐位比对（v2~v5 输出无效，不比对）。
#include "ops.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

struct FSm { uint32_t wsum[4][8]; uint32_t wsum_init[8]; };

__device__ __forceinline__ void ex16(const uint4& v, uint32_t* o)
{
    o[0] = v.x & 0xffu;         o[1] = (v.x >> 8) & 0xffu;
    o[2] = (v.x >> 16) & 0xffu; o[3] = v.x >> 24;
    o[4] = v.y & 0xffu;         o[5] = (v.y >> 8) & 0xffu;
    o[6] = (v.y >> 16) & 0xffu; o[7] = v.y >> 24;
    o[8] = v.z & 0xffu;         o[9] = (v.z >> 8) & 0xffu;
    o[10] = (v.z >> 16) & 0xffu; o[11] = v.z >> 24;
    o[12] = v.w & 0xffu;        o[13] = (v.w >> 8) & 0xffu;
    o[14] = (v.w >> 16) & 0xffu; o[15] = v.w >> 24;
}

template <int ROWS, int NOBAR, int PF, int STCS>
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
            seg = (pv[0]+pv[1]+pv[2]+pv[3]) + (pv[4]+pv[5]+pv[6]+pv[7])
                + (pv[8]+pv[9]+pv[10]+pv[11]) + (pv[12]+pv[13]+pv[14]+pv[15]);
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

    uint4 pfl;                       // 预取缓冲
    if (PF && valid) pfl = *reinterpret_cast<const uint4*>(p);

    for (int r = r0; r < r1; r += ROWS) {
        uint32_t o[ROWS][16];
        uint32_t seg[ROWS], inc[ROWS];
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            const bool ok = ((r + j) < r1) && valid;
            uint32_t s = 0;
            if (ok) {
                uint4 v;
                if (PF && j == 0) v = pfl;
                else               v = *reinterpret_cast<const uint4*>(p + (size_t)j * sstep);
                ex16(v, o[j]);
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
        if (!NOBAR) {
#pragma unroll
            for (int j = 0; j < ROWS; ++j)
                if (lane == 31) sm.wsum[j][warp] = inc[j];
            __syncthreads();
        }
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            uint32_t off = 0;
            if (!NOBAR)
                for (int w = 0; w < warp; ++w) off += sm.wsum[j][w];
            uint32_t run = off + (inc[j] - seg[j]);
            if ((r + j) < r1 && valid) {
#pragma unroll
                for (int k = 0; k < 16; ++k) { run += o[j][k]; acc[k] += run; }
                uint32_t* qq = (uint32_t*)((uint8_t*)q + (size_t)j * dstep);
                if (STCS) {
                    __stcs(reinterpret_cast<uint4*>(qq),      make_uint4(acc[0],  acc[1],  acc[2],  acc[3]));
                    __stcs(reinterpret_cast<uint4*>(qq + 4),  make_uint4(acc[4],  acc[5],  acc[6],  acc[7]));
                    __stcs(reinterpret_cast<uint4*>(qq + 8),  make_uint4(acc[8],  acc[9],  acc[10], acc[11]));
                    __stcs(reinterpret_cast<uint4*>(qq + 12), make_uint4(acc[12], acc[13], acc[14], acc[15]));
                } else {
                    *reinterpret_cast<uint4*>(qq)      = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
                    *reinterpret_cast<uint4*>(qq + 4)  = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
                    *reinterpret_cast<uint4*>(qq + 8)  = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
                    *reinterpret_cast<uint4*>(qq + 12) = make_uint4(acc[12], acc[13], acc[14], acc[15]);
                }
            }
        }
        p += (size_t)ROWS * sstep;
        q = (uint32_t*)((uint8_t*)q + (size_t)ROWS * dstep);
        if (PF && ((r + ROWS) < r1) && valid)
            pfl = *reinterpret_cast<const uint4*>(p);
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
inline int adapt_th(int H)
{
    int t = (H + 63) / 64;
    t = ((t + 31) / 32) * 32;
    return t < 32 ? 32 : (t > 256 ? 256 : t);
}

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
    cudaFuncAttributes at{};
    cudaFuncGetAttributes(&at, fused_v<1,0,0,0>); printf("R1:      regs=%d\n", at.numRegs);
    cudaFuncGetAttributes(&at, fused_v<2,0,0,0>); printf("R2:      regs=%d\n", at.numRegs);
    cudaFuncGetAttributes(&at, fused_v<1,0,1,0>); printf("R1+PF:   regs=%d\n", at.numRegs);
    cudaFuncGetAttributes(&at, fused_v<1,0,0,1>); printf("R1+STCS: regs=%d\n", at.numRegs);

    const int sizes[][2] = {{1280, 720}, {1920, 1080}, {3840, 2160}};
    const int NV = 7;
    const char* vn[NV] = {"R1", "R2", "NOBAR-R1", "NOBAR-R2", "PF-R1", "STCS-R1", "R1-TH64"};

    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t sstep = 0;
        uint32_t* dA = nullptr, *dB = nullptr, *dC = nullptr; size_t dstep = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &sstep, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&dA, &dstep, (size_t)W * 4, H));
        CUDA_CHECK(cudaMallocPitch(&dB, &dstep, (size_t)W * 4, H));
        CUDA_CHECK(cudaMallocPitch(&dC, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(11);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, sstep, host.data(), W, W, H, cudaMemcpyHostToDevice));

        const int THa = adapt_th(H);
        uint32_t* pre = nullptr;
        CUDA_CHECK(cudaMalloc(&pre, (size_t)idiv_up(H, 16) * W * sizeof(uint32_t)));

        const int tw = (W + 15) >> 4;
        const int fth = ((tw + 31) >> 5) << 5;
        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);

        auto stage = [&](int TH) {
            const int nC = idiv_up(H, TH);
            const int total = nC * (W >> 4);
            chunk_sum_k<<<idiv_up(total, 256), 256>>>(src, sstep, pre, W, H, TH);
            chunk_scan_k<<<idiv_up(W, 256), 256>>>(pre, W, nC);
        };
        auto launch = [&](int v, uint32_t* d, int TH) {
            const int nC = idiv_up(H, TH);
            switch (v) {
            case 0: fused_v<1,0,0,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 1: fused_v<2,0,0,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 2: fused_v<1,1,0,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 3: fused_v<2,1,0,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 4: fused_v<1,0,1,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 5: fused_v<1,0,0,1><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            default: fused_v<1,0,0,0><<<nC, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            }
        };

        // 正确性：v0 / v1 / v6(TH=64 的 R1) 互相比对
        std::vector<uint32_t> hA((size_t)W * H), hB((size_t)W * H), hC((size_t)W * H);
        stage(THa); launch(0, dA, THa); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy2D(hA.data(), (size_t)W * 4, dA, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        stage(THa); launch(1, dB, THa); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy2D(hB.data(), (size_t)W * 4, dB, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        const int TH64 = 64;
        stage(TH64); launch(6, dC, TH64); CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy2D(hC.data(), (size_t)W * 4, dC, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        long d10 = 0, d16 = 0;
        for (size_t i = 0; i < hA.size(); ++i) { if (hA[i] != hB[i]) ++d10; if (hA[i] != hC[i]) ++d16; }
        printf("\n=== %dx%d TH_adapt=%d fth=%d ===  diff(v1-v0)=%ld diff(v6-v0)=%ld\n",
               W, H, THa, fth, d10, d16);

        double best[NV];
        for (int v = 0; v < NV; ++v) best[v] = 1e30;
        for (int round = 0; round < 5; ++round) {
            for (int v = 0; v < NV; ++v) {
                const int TH = (v == 6) ? 64 : THa;
                stage(TH);
                best[v] = std::min(best[v], median_run([&] { launch(v, dA, TH); }, warmup, reps));
            }
        }
        for (int v = 0; v < NV; ++v)
            printf("  %-9s TH=%-4d %7.4f ms  %6.1f GB/s(5N)\n", vn[v], (v == 6) ? 64 : THa,
                   best[v], 5.0 * W * H / 1e6 / (best[v] * 1e-3) / 1000.0);
        cudaFree(src); cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(pre);
    }
    return 0;
}
