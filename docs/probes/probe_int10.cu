// probe_int10 (r52): 确认 4K 的 TH 甜点与 1080p 的 TH 上限（r49 教训：改默认前 ≥3 轮交错 A/B）。
// 只用 ROWS=1（不改内核形状，仅调 chunk 高度）—— 这样改动最小。
// 同时报告「stage(A+B) + fused」整串时间，用来预测 bench 数字。
#include "ops.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

struct FSm { uint32_t wsum[2][8]; uint32_t wsum_init[8]; };

// 与 src/integral.cu 的 integral_fused_kernel 逐行同构（ROWS=1）
__global__ void fused_k(const uint8_t* __restrict__ src, size_t sstep,
                        uint32_t* __restrict__ dst, size_t dstep,
                        const uint32_t* __restrict__ pre, int W, int H, int TH)
{
    __shared__ FSm sm;
    const int chunk = blockIdx.x;
    const int r0 = chunk * TH;
    if (r0 >= H) return;
    const int r1 = min(H, r0 + TH);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c0 = tid * 16;
    const bool valid = (c0 < W);
    const unsigned ones = 0x01010101u;
    uint32_t acc[16];
    {
        uint32_t pv[16]; uint32_t seg = 0;
        if (pre && valid) {
            const uint4* pp = (const uint4*)(pre + (size_t)chunk * W + c0);
            const uint4 a0 = pp[0], a1 = pp[1], a2 = pp[2], a3 = pp[3];
            pv[0] = a0.x; pv[1] = a0.y; pv[2] = a0.z; pv[3] = a0.w;
            pv[4] = a1.x; pv[5] = a1.y; pv[6] = a1.z; pv[7] = a1.w;
            pv[8] = a2.x; pv[9] = a2.y; pv[10] = a2.z; pv[11] = a2.w;
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
    for (int r = r0; r < r1; ++r) {
        uint32_t o[16]; uint32_t seg = 0;
        if (valid) {
            const uint4 v = *reinterpret_cast<const uint4*>(p);
            o[0] = v.x & 0xffu;         o[1] = (v.x >> 8) & 0xffu;
            o[2] = (v.x >> 16) & 0xffu; o[3] = v.x >> 24;
            o[4] = v.y & 0xffu;         o[5] = (v.y >> 8) & 0xffu;
            o[6] = (v.y >> 16) & 0xffu; o[7] = v.y >> 24;
            o[8] = v.z & 0xffu;         o[9] = (v.z >> 8) & 0xffu;
            o[10] = (v.z >> 16) & 0xffu; o[11] = v.z >> 24;
            o[12] = v.w & 0xffu;        o[13] = (v.w >> 8) & 0xffu;
            o[14] = (v.w >> 16) & 0xffu; o[15] = v.w >> 24;
            seg = __dp4a(v.x, ones, __dp4a(v.y, ones, __dp4a(v.z, ones, __dp4a(v.w, ones, 0u))));
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) o[k] = 0;
        }
        uint32_t inc = seg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        if (lane == 31) sm.wsum[r & 1][warp] = inc;
        __syncthreads();
        uint32_t off = 0;
        for (int w = 0; w < warp; ++w) off += sm.wsum[r & 1][w];
        uint32_t run = off + (inc - seg);
        if (valid) {
#pragma unroll
            for (int k = 0; k < 16; ++k) { run += o[k]; acc[k] += run; }
            uint4* qp = (uint4*)q;
            qp[0] = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
            qp[1] = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
            qp[2] = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
            qp[3] = make_uint4(acc[12], acc[13], acc[14], acc[15]);
            p += sstep; q = (uint32_t*)((uint8_t*)q + dstep);
        }
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
        const uint32_t v = *pp; *pp = s; s += v;
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
    const struct { int W, H; int THs[5]; int n; } cases[] = {
        {3840, 2160, {64, 96, 128, 192, 256}, 5},
        {1920, 1080, {32, 48, 64, 96, 128}, 5},
        {1280,  720, {32, 48, 64, 96, 128}, 5},
        { 640,  480, {32, 48, 64, 96, 128}, 5},
    };
    for (auto& cs : cases) {
        const int W = cs.W, H = cs.H;
        uint8_t* src = nullptr; size_t sstep = 0;
        uint32_t* dA = nullptr, *dB = nullptr; size_t dstep = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &sstep, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&dA, &dstep, (size_t)W * 4, H));
        CUDA_CHECK(cudaMallocPitch(&dB, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(11);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, sstep, host.data(), W, W, H, cudaMemcpyHostToDevice));
        uint32_t* pre = nullptr;
        CUDA_CHECK(cudaMalloc(&pre, (size_t)idiv_up(H, 32) * W * sizeof(uint32_t)));
        const int tw = (W + 15) >> 4;
        const int fth = ((tw + 31) >> 5) << 5;
        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);

        printf("\n=== %dx%d fth=%d ===\n", W, H, fth);
        // 参照输出（TH=64）
        auto stage = [&](int TH) {
            const int nC = idiv_up(H, TH);
            chunk_sum_k<<<idiv_up(nC * (W >> 4), 256), 256>>>(src, sstep, pre, W, H, TH);
            chunk_scan_k<<<idiv_up(W, 256), 256>>>(pre, W, nC);
        };
        stage(64); fused_k<<<idiv_up(H, 64), fth>>>(src, sstep, dA, dstep, pre, W, H, 64);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint32_t> hA((size_t)W * H), hB((size_t)W * H);
        CUDA_CHECK(cudaMemcpy2D(hA.data(), (size_t)W * 4, dA, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));

        double bk[5], bp[5];
        for (int i = 0; i < 5; ++i) { bk[i] = 1e30; bp[i] = 1e30; }
        for (int round = 0; round < 5; ++round) {
            for (int i = 0; i < cs.n; ++i) {
                const int TH = cs.THs[i];
                const int nC = idiv_up(H, TH);
                stage(TH);
                bk[i] = std::min(bk[i], median_run([&] { fused_k<<<nC, fth>>>(src, sstep, dA, dstep, pre, W, H, TH); }, warmup, reps));
                bp[i] = std::min(bp[i], median_run([&] {
                    stage(TH);
                    fused_k<<<nC, fth>>>(src, sstep, dA, dstep, pre, W, H, TH);
                }, warmup, reps));
            }
        }
        for (int i = 0; i < cs.n; ++i) {
            const int TH = cs.THs[i];
            const int nC = idiv_up(H, TH);
            // 正确性
            stage(TH); fused_k<<<nC, fth>>>(src, sstep, dB, dstep, pre, W, H, TH);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy2D(hB.data(), (size_t)W * 4, dB, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            long diff = 0;
            for (size_t j = 0; j < hA.size(); ++j) if (hA[j] != hB[j]) ++diff;
            printf("  TH=%-4d nC=%-3d kernel=%7.4f (%5.1f GB/s)  pipeline=%7.4f  diff=%ld\n",
                   TH, nC, bk[i], 5.0 * W * H / 1e6 / (bk[i] * 1e-3) / 1000.0, bp[i], diff);
        }
        cudaFree(src); cudaFree(dA); cudaFree(dB); cudaFree(pre);
    }
    return 0;
}
