// probe_int6 (r52): integral fused kernel variants, aiming at the 4K 65 GB/s -> 93 GB/s gap.
// r51 结论：4K 0.639ms 纯内核 / 5N 字节 = 65 GB/s（天花板 93），气泡在「每行一次 __syncthreads」。
// 变体：
//   ROWS = 每屏障处理的行数（1 = r51 原版；2/4 = 一次屏障摊到 2/4 行）
//   ST   = dst 用 __stcs 流式写（不污染 L2，dst 只写不读）
//   LD   = src 用 __ldcs 流式读
// 每行组内的跨 warp 偏移靠 wsum[ROWS][8] 的不同槽位隔离，屏障保证「读旧槽 < 屏障 < 写新槽」。
// 计时 3 轮交错取 min；每个变体与 R1S0L0 基线逐位比对（要求 diff=0）。
#include "ops.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

constexpr int kVGrid = 0;   // 不使用

struct FSm {
    uint32_t wsum[4][8];
    uint32_t wsum_init[8];
};

__device__ __forceinline__ uint4 ld_u4(const uint8_t* p, int LD)
{
    if (LD) return __ldcs(reinterpret_cast<const uint4*>(p));
    return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void st_u4(uint32_t* q, const uint4& v, int ST)
{
    if (ST) __stcs(reinterpret_cast<uint4*>(q), v);
    else    *reinterpret_cast<uint4*>(q) = v;
}

template <int ROWS, int ST, int LD>
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
        uint32_t seg[ROWS];
        uint32_t inc[ROWS];
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            const bool rowok = ((r + j) < r1) && valid;
            uint32_t s = 0;
            if (rowok) {
                const uint4 v = ld_u4(p + (size_t)j * sstep, LD);
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
        __syncthreads();                    // 每 ROWS 行仅此一次屏障
#pragma unroll
        for (int j = 0; j < ROWS; ++j) {
            uint32_t off = 0;
            for (int w = 0; w < warp; ++w) off += sm.wsum[j][w];
            uint32_t run = off + (inc[j] - seg[j]);
            if ((r + j) < r1 && valid) {
#pragma unroll
                for (int k = 0; k < 16; ++k) { run += o[j][k]; acc[k] += run; }
                uint32_t* qq = (uint32_t*)((uint8_t*)q + (size_t)j * dstep);
                st_u4(qq,      make_uint4(acc[0],  acc[1],  acc[2],  acc[3]),  ST);
                st_u4(qq + 4,  make_uint4(acc[4],  acc[5],  acc[6],  acc[7]),  ST);
                st_u4(qq + 8,  make_uint4(acc[8],  acc[9],  acc[10], acc[11]), ST);
                st_u4(qq + 12, make_uint4(acc[12], acc[13], acc[14], acc[15]), ST);
            }
        }
        p += (size_t)ROWS * sstep;
        q = (uint32_t*)((uint8_t*)q + (size_t)ROWS * dstep);
    }
}

// ---- A/B 阶段（B：chunk 列和 + 跨 chunk 前缀）原样照抄 src 版，用于计时整串 ----
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
    uint32_t* o = csum + (size_t)chunk * W + c0;
    uint4* op = (uint4*)o;
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
inline int fused_th(int H)
{
    int th = (H + 63) / 64;
    th = ((th + 31) / 32) * 32;
    if (th < 32) th = 32;
    if (th > 256) th = 256;
    return th;
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
    const int sizes[][2] = {{1280, 720}, {1920, 1080}, {3840, 2160}};
    const int NV = 7;
    const char* vn[NV] = {"R1S0L0", "R2S0L0", "R4S0L0", "R1S1L0", "R2S1L0", "R2S0L1", "R2S1L1"};

    printf("%-12s", "size");
    for (int v = 0; v < NV; ++v) printf("%10s", vn[v]);
    printf("%10s%10s%10s\n", "sum+scan", "R1seq", "R2seq");

    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t sstep = 0;
        uint32_t* dst0 = nullptr, *dst1 = nullptr; size_t dstep = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &sstep, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&dst0, &dstep, (size_t)W * 4, H));
        CUDA_CHECK(cudaMallocPitch(&dst1, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(11);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, sstep, host.data(), W, W, H, cudaMemcpyHostToDevice));

        const int TH = fused_th(H);
        const int nChunks = idiv_up(H, TH);
        uint32_t* pre = nullptr;
        CUDA_CHECK(cudaMalloc(&pre, (size_t)nChunks * W * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(pre, 0, (size_t)nChunks * W * sizeof(uint32_t)));

        const int tw = (W + 15) >> 4;
        const int fth = ((tw + 31) >> 5) << 5;

        auto stage = [&]() {   // B 阶段：列和 + 前缀
            const int groups = (W >> 4);
            const int total = nChunks * groups;
            chunk_sum_k<<<idiv_up(total, 256), 256>>>(src, sstep, pre, W, H, TH);
            chunk_scan_k<<<idiv_up(W, 256), 256>>>(pre, W, nChunks);
        };
        auto fused = [&](int v, uint32_t* d) {
            switch (v) {
            case 0: fused_v<1,0,0><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 1: fused_v<2,0,0><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 2: fused_v<4,0,0><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 3: fused_v<1,1,0><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 4: fused_v<2,1,0><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            case 5: fused_v<2,0,1><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            default: fused_v<2,1,1><<<nChunks, fth>>>(src, sstep, d, dstep, pre, W, H, TH); break;
            }
        };

        // ---- 正确性：每个变体 vs 基线（R1S0L0）逐位比对 ----
        stage();
        fused(0, dst0);
        CUDA_CHECK(cudaDeviceSynchronize());
        int bad[NV] = {0};
        std::vector<uint32_t> h0((size_t)W * H), h1((size_t)W * H);
        CUDA_CHECK(cudaMemcpy2D(h0.data(), (size_t)W * 4, dst0, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        for (int v = 1; v < NV; ++v) {
            CUDA_CHECK(cudaMemset(dst1, 0, (size_t)dstep * H));
            stage();
            fused(v, dst1);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy2D(h1.data(), (size_t)W * 4, dst1, dstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < h0.size(); ++i) if (h0[i] != h1[i]) { ++bad[v]; }
        }

        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);
        double best[NV];
        for (int v = 0; v < NV; ++v) best[v] = 1e30;
        double best_stage = 1e30, best_seq1 = 1e30, best_seq2 = 1e30;
        for (int round = 0; round < 3; ++round) {
            for (int v = 0; v < NV; ++v) {
                stage();
                best[v] = std::min(best[v], median_run([&] { fused(v, dst0); }, warmup, reps));
            }
            best_stage = std::min(best_stage, median_run([&] { stage(); }, warmup, reps));
            best_seq1 = std::min(best_seq1, median_run([&] { stage(); fused(0, dst0); }, warmup, reps));
            best_seq2 = std::min(best_seq2, median_run([&] { stage(); fused(1, dst0); }, warmup, reps));
        }
        printf("%5dx%-6d", W, H);
        for (int v = 0; v < NV; ++v) printf("%10.4f", best[v]);
        printf("%10.4f%10.4f%10.4f\n", best_stage, best_seq1, best_seq2);
        printf("%-12s", "  GB/s(5N)");
        for (int v = 0; v < NV; ++v)
            printf("%10.1f", 5.0 * W * H / 1e6 / (best[v] * 1e-3) / 1000.0);
        printf("\n%-12s", "  diff");
        for (int v = 0; v < NV; ++v) printf("%10d", bad[v]);
        printf("\n");
        cudaFree(src); cudaFree(dst0); cudaFree(dst1); cudaFree(pre);
    }
    printf("\nTH/nChunks 由 fused_th 决定；GB/s(5N) 按读 1N + 写 4N 折算\n");
    return 0;
}
