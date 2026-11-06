// probe_int1：积分图分段耗时 + 候选内核实测（r50）
// 动机：4K integral 2.42ms = 1.76x（全库最低的真实大内核）。r50 首次分段实测：
//   pass1 1.189ms (34.7 GB/s) / pass2a 0.707 (93 GB/s) / pass2b 0.536 (98 GB/s)，合计 2.42
//   → pass2a+pass2b 已贴带宽天花板，唯一大空间在 ① pass1 的写放大 ② pass2 的「回加前缀」多一遍全图
// 本探针：
//   rows     现有实现（每 lane 16B、16 条标量 u32 store）
//   rows_vec 同算法，16 个前缀合 4 条 uint4 store（验证写放大假设）
//   rows_seg 每 lane 只负责 4 字节（32 lane 连续 128B，load/store 天然全合并）
//   cols     现有 pass2a + pass2b（chunk 内列扫描 + 回加 carry）
//   cols1    不分 chunk，每线程 1 列走完 H 行（无回加，并行度只剩 W 线程）
//   cols4    不分 chunk，每线程 4 列（uint4）走完 H 行（并行度 W/4）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

typedef unsigned int uint;

namespace rep {

constexpr int CHUNK_H    = 512;
constexpr int CT_THREADS = 128;

// ---------------- pass1 三个变体 ----------------
__global__ void rows_scalar_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 512;
    const bool vecOK = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;
    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;
        uint32_t vals[16];
        uint32_t partial = 0;
        if (my0 + 16 <= W && vecOK) {
            union { uint4 v; uint8_t b[16]; } u;
            u.v = *reinterpret_cast<const uint4*>(srow + my0);
#pragma unroll
            for (int k = 0; k < 16; ++k) { vals[k] = u.b[k]; partial += vals[k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int x = my0 + k;
                uint32_t v = 0;
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
        for (int k = 0; k < 16; ++k) {
            run += vals[k];
            const int x = my0 + k;
            if (x < W) drow[x] = run;
        }
    }
}

__global__ void rows_vec_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 512;
    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;
    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;
        uint32_t vals[16];
        uint32_t partial = 0;
        const bool vec = (my0 + 16 <= W);
        if (vec) {
            union { uint4 v; uint8_t b[16]; } u;
            u.v = *reinterpret_cast<const uint4*>(srow + my0);
#pragma unroll
            for (int k = 0; k < 16; ++k) { vals[k] = u.b[k]; partial += vals[k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int x = my0 + k;
                uint32_t v = 0;
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
        if (vec) {
            uint32_t o[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) { run += vals[k]; o[k] = run; }
            uint4* op = (uint4*)(drow + my0);
            op[0] = make_uint4(o[0],  o[1],  o[2],  o[3]);
            op[1] = make_uint4(o[4],  o[5],  o[6],  o[7]);
            op[2] = make_uint4(o[8],  o[9],  o[10], o[11]);
            op[3] = make_uint4(o[12], o[13], o[14], o[15]);
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                run += vals[k];
                const int x = my0 + k;
                if (x < W) drow[x] = run;
            }
        }
    }
}

// 每 lane 4 字节：32 lane 覆盖 128 连续字节 → load/store 一条指令覆盖 128B
__global__ void rows_seg_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 128;
    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;
    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 4;
        uint32_t vals[4];
        uint32_t partial = 0;
        const bool vec = (my0 + 4 <= W);
        if (vec) {
            const uint32_t v4 = *(const uint*)(srow + my0);
            vals[0] = v4 & 0xffu;
            vals[1] = (v4 >> 8) & 0xffu;
            vals[2] = (v4 >> 16) & 0xffu;
            vals[3] = (v4 >> 24) & 0xffu;
            partial = vals[0] + vals[1] + vals[2] + vals[3];
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int x = my0 + k;
                const uint32_t t = (x < W) ? (uint32_t)srow[x] : 0u;
                vals[k] = t; partial += t;
            }
        }
        uint32_t inc = partial;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        const uint32_t excl = inc - partial + carry;
        carry += __shfl_sync(0xffffffffu, inc, 31);
        uint32_t run = excl;
        uint32_t o[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) { run += vals[k]; o[k] = run; }
        if (vec) {
            *(uint4*)(drow + my0) = make_uint4(o[0], o[1], o[2], o[3]);
        } else {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int x = my0 + k;
                if (x < W) drow[x] = o[k];
            }
        }
    }
}

// ---------------- pass2 变体 ----------------
__global__ void cols_kernel(const uint32_t* __restrict__ in, size_t istep,
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
    uint4 car = make_uint4(0, 0, 0, 0);
    for (int r = r0; r < r1; ++r) {
        uint4 v = *ip;
        v.x += car.x; v.y += car.y; v.z += car.z; v.w += car.w;
        *op = v;
        car = v;
        ip = (const uint4*)((const uint8_t*)ip + istep);
        op = (uint4*)((const uint8_t*)op + ostep);
    }
    if (nChunks > 1) {
        uint32_t* cr = carry + (size_t)chunk * W + c0;
        cr[0] = car.x; cr[1] = car.y; cr[2] = car.z; cr[3] = car.w;
    }
}

__global__ void carry_add_kernel(uint32_t* __restrict__ out, size_t ostep,
                                 const uint32_t* __restrict__ carry, int W, int H)
{
    const int chunk = blockIdx.y + 1;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;
    uint4 exc = make_uint4(0, 0, 0, 0);
    for (int j = 0; j < chunk; ++j) {
        const uint32_t* cr = carry + (size_t)j * W + c0;
        exc.x += cr[0]; exc.y += cr[1]; exc.z += cr[2]; exc.w += cr[3];
    }
    uint4* op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;
    for (int r = r0; r < r1; ++r) {
        uint4 v = *op;
        v.x += exc.x; v.y += exc.y; v.z += exc.z; v.w += exc.w;
        *op = v;
        op = (uint4*)((const uint8_t*)op + ostep);
    }
}

// 无 chunk：每线程 1 列走完 H 行
__global__ void cols1_kernel(const uint32_t* __restrict__ in, size_t istep,
                             uint32_t* __restrict__ out, size_t ostep, int W, int H)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    const uint32_t* ip = (const uint32_t*)((const uint8_t*)in) + c;
    uint32_t* op = (uint32_t*)((uint8_t*)out) + c;
    uint32_t car = 0;
    for (int r = 0; r < H; ++r) {
        car += *ip;
        *op = car;
        ip = (const uint32_t*)((const uint8_t*)ip + istep);
        op = (uint32_t*)((const uint8_t*)op + ostep);
    }
}

// 无 chunk：每线程 4 列（uint4）走完 H 行
__global__ void cols4_kernel(const uint32_t* __restrict__ in, size_t istep,
                             uint32_t* __restrict__ out, size_t ostep, int W, int H)
{
    const int c0 = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c0 >= W) return;
    const uint4* ip = (const uint4*)((const uint8_t*)in) + c0 / 4;
    uint4* op = (uint4*)((uint8_t*)out) + c0 / 4;
    uint4 car = make_uint4(0, 0, 0, 0);
    for (int r = 0; r < H; ++r) {
        uint4 v = *ip;
        v.x += car.x; v.y += car.y; v.z += car.z; v.w += car.w;
        *op = v;
        car = v;
        ip = (const uint4*)((const uint8_t*)ip + istep);
        op = (uint4*)((const uint8_t*)op + ostep);
    }
}

struct Buf { void* p = nullptr; size_t sz = 0; };
Buf g_buf;
uint32_t* ws(size_t need) {
    if (g_buf.sz < need) {
        if (g_buf.p) cudaFree(g_buf.p);
        cudaMalloc(&g_buf.p, need); g_buf.sz = need;
    }
    return (uint32_t*)g_buf.p;
}

inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

} // namespace rep

template <typename F>
static double timeitMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    cudaDeviceSynchronize();
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    struct Case { int W, H; };
    const Case cases[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int reps = (argc > 1) ? atoi(argv[1]) : 40;

    for (const Case& c : cases) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        uint8_t* d_src = nullptr; size_t sstep = 0;
        uint32_t* d_dst = nullptr; size_t dstep = 0;
        CK(cudaMallocPitch(&d_src, &sstep, (size_t)W, H));
        CK(cudaMallocPitch(&d_dst, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> h(N);
        for (size_t i = 0; i < N; ++i) h[i] = (uint8_t)(i * 37 + 11);
        CK(cudaMemcpy2D(d_src, sstep, h.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));

        const int nChunks = rep::idiv_up(H, rep::CHUNK_H);
        const int colBlocks = rep::idiv_up(W, rep::CT_THREADS * 4);
        uint32_t* carry = (nChunks > 1) ? rep::ws((size_t)nChunks * W * 4) : nullptr;
        const int blocks1 = (H + 7) / 8;

        auto p1_scalar = [&] { rep::rows_scalar_kernel<<<blocks1, 256>>>(d_src, sstep, d_dst, dstep, W, H); };
        auto p1_vec    = [&] { rep::rows_vec_kernel<<<blocks1, 256>>>(d_src, sstep, d_dst, dstep, W, H); };
        auto p1_seg    = [&] { rep::rows_seg_kernel<<<blocks1, 256>>>(d_src, sstep, d_dst, dstep, W, H); };
        auto p2_chunk  = [&] {
            dim3 g(colBlocks, nChunks), t(rep::CT_THREADS);
            rep::cols_kernel<<<g, t>>>(d_dst, dstep, d_dst, dstep, carry, W, H, nChunks);
            if (nChunks > 1) {
                dim3 g2(colBlocks, nChunks - 1);
                rep::carry_add_kernel<<<g2, t>>>(d_dst, dstep, carry, W, H);
            }
        };
        auto p2_c1 = [&] {
            const int cb = rep::idiv_up(W, 256);
            rep::cols1_kernel<<<cb, 256>>>(d_dst, dstep, d_dst, dstep, W, H);
        };
        auto p2_c4 = [&] {
            const int cb = rep::idiv_up(W, 256 * 4);
            rep::cols4_kernel<<<cb, 256>>>(d_dst, dstep, d_dst, dstep, W, H);
        };
        auto api = [&] { gpu_integral(d_src, sstep, d_dst, dstep, W, H); };

        const double t1s = timeitMin(p1_scalar, 5, reps);
        const double t1v = timeitMin(p1_vec, 5, reps);
        const double t1g = timeitMin(p1_seg, 5, reps);
        const double t2c = timeitMin(p2_chunk, 5, reps);
        const double t2a = timeitMin(p2_c1, 5, reps);
        const double t2b = timeitMin(p2_c4, 5, reps);
        const double tApi = timeitMin(api, 5, reps);
        const double mb = (double)N * 5.0 / 1e6;   // pass1 流量 MB（读 1B + 写 4B）

        printf("%4dx%-4d nCh=%d | p1 scalar %.4f (%.0f GB/s) | p1_vec %.4f (%.0f) | p1_seg %.4f (%.0f)\n",
               W, H, nChunks, t1s, mb / t1s, t1v, mb / t1v, t1g, mb / t1g);
        printf("            | p2 chunk %.4f | p2 1col %.4f | p2 4col %.4f | api %.4f\n", t2c, t2a, t2b, tApi);
        const double b1 = std::min(t1s, std::min(t1v, t1g));
        const double b2 = std::min(t2c, std::min(t2a, t2b));
        printf("            | best combo %.4f (now %.4f, %.2fx)\n", b1 + b2, tApi, tApi / (b1 + b2));
    }
    return 0;
}