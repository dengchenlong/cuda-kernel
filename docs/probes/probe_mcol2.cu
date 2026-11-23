// probe_mcol2 (r65): mean 大核**列遍**重写 —— 从「每线程一列沿 y 走」到「每线程 C 个连续列 × rpt 行分块」
//
// 归因（probe_meanbig asym，r64 已量）：
//   4K    (1,1) 1.506 | (31,1) 4.893（行遍 +3.39）| (1,31) 1.861（列遍 +0.36）| (31,31) 4.461
//   r64 修完行遍后（4K k=31 行遍 3.54→0.48），列遍成为主因：
//     4K k=15 总 3.74ms = 行遍 ~1.37 + 列遍 ~2.36；k=31 总 1.50 = 行遍 0.48 + 列遍 ~1.02
//   两个病灶：
//     A) mean_cols_blocked_kernel（k<=15 用）**每输出行 O(k) 直加** → 15 倍 tmp 重读；
//     B) mean_cols_kernel（k>15 用）**每线程一列沿 y 走** → 4K 只有 W=3840 线程（6 SM 上
//        30 个 warp），并行度严重不足；折 41 GB/s vs u16->u8 地板 95.3。
//
// 本探针要回答的核心问题（PROGRESS 明确要求先量清，别直接改）：
//   列方向「每线程 C 个连续列」时，**上下一行的 16B 读/写是否真的合并**？
//   ——答案在结构上应该是「合并的」：同一 warp 的 32 个 lane 在同一行上覆盖 32*C 个连续列，
//     读 32*C*2B、写 32*C*1B 都是连续大段；与行遍的坑（lane 相隔 32B）**不同源**。
//     探针用实测定论，并顺带扫出 C / rpt / 线程数的最佳档。
//
// 内核族：
//   old_blocked : 照抄 src 的 mean_cols_blocked_kernel（k<=15 现状）
//   old_slide   : 照抄 src 的 mean_cols_kernel（k>15 现状）
//   vec<C>      : 新内核 —— 每线程 C 个连续列，行分块 rpt，初始窗口 O(k) 直加后滑动 O(1)/px；
//                 读 uint4（16 字节 = 8 个 u16 列）×2、写 uint4（16 字节 = C 个 u8）
//   floor       : 同形状 u16->u8 纯搬运（3N 字节）作地板
// 模式：check（新 vs 旧 vs CPU int64，逐值）/ sweep（尺寸 × k × C × rpt 交错 min）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include "ops.h"

#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

typedef unsigned int uint;

namespace {

static inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

static __device__ __forceinline__ int mir(int i, int len)
{
    if (i < 0) i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// ---------------- 旧内核 A：mean_cols_blocked_kernel（每输出行 O(k) 直加）----------------
__global__ void old_blocked(const uint16_t* __restrict__ tmp, size_t tstep,
                            uint8_t* __restrict__ dst, size_t dstep,
                            int W, int H, int kh, float inv_area, int rpt)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;
    const int half = kh >> 1;
    const uint16_t* tcol = (const uint16_t*)((const uint8_t*)tmp + x * 2);
    const size_t trows = tstep >> 1;
    auto fetch = [=](int r) -> uint32_t { return tcol[(size_t)mir(r, H) * trows]; };
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    for (int y = y0; y < y1; ++y) {
        uint32_t sum = 0;
#pragma unroll 4
        for (int j = -half; j <= half; ++j) sum += fetch(y + j);
        const int v = __float2int_rn((float)sum * inv_area);
        dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
    }
}

// ---------------- 旧内核 B：mean_cols_kernel（每线程一列，全程滑动）----------------
__global__ void old_slide(const uint16_t* __restrict__ tmp, size_t tstep,
                          uint8_t* __restrict__ dst, size_t dstep,
                          int W, int H, int kh, float inv_area)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= W) return;
    const int half = kh >> 1;
    const uint16_t* tcol = (const uint16_t*)((const uint8_t*)tmp + x * 2);
    const size_t trows = tstep >> 1;
    auto fetch = [=](int r) -> uint32_t { return tcol[(size_t)mir(r, H) * trows]; };
    uint32_t sum = 0;
    for (int j = 0; j < kh; ++j) sum += fetch(-half + j);
    for (int y = 0; y < H; ++y) {
        const int v = __float2int_rn((float)sum * inv_area);
        dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
        if (y + 1 < H) sum += fetch(y + 1 + half) - fetch(y - half);
    }
}

// ---------------- 地板：u16 -> u8 同形状纯搬运 ----------------
__global__ void floor_copy(const uint16_t* __restrict__ s, size_t sstep,
                           uint8_t* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int y = blockIdx.y;
    if (c >= W) return;
    const uint4* p = (const uint4*)((const uint8_t*)s + (size_t)y * sstep + (size_t)c * 2);
    const uint4 a = p[0], b = p[1];
    const uint4 o = make_uint4(
        (a.x & 0xffu) | ((a.x >> 8) & 0xff00u) | ((a.y & 0xffu) << 16) | (((a.y >> 8) & 0xffu) << 24),
        (a.z & 0xffu) | ((a.z >> 8) & 0xff00u) | ((a.w & 0xffu) << 16) | (((a.w >> 8) & 0xffu) << 24),
        (b.x & 0xffu) | ((b.x >> 8) & 0xff00u) | ((b.y & 0xffu) << 16) | (((b.y >> 8) & 0xffu) << 24),
        (b.z & 0xffu) | ((b.z >> 8) & 0xff00u) | ((b.w & 0xffu) << 16) | (((b.w >> 8) & 0xffu) << 24));
    *(uint4*)(d + (size_t)y * dstep + c) = o;
}

// ---------------- 新内核：每线程 C 个连续列 × rpt 行分块 + 滑动 ----------------
// C ∈ {4,8,16}；读/写都按 16B 或 8B 粒度（C=16: 读 32B 写 16B；C=8: 读 16B 写 8B；C=4: 读 8B 写 4B）
template <int C>
__device__ __forceinline__ void col_ld(const uint16_t* __restrict__ p, uint32_t v[C])
{
    static_assert(C % 4 == 0, "C must be multiple of 4");
    if (C >= 8) {
#pragma unroll
        for (int i = 0; i < C / 8; ++i) {
            const uint4 w = *(const uint4*)(p + 8 * i);
            v[8*i+0] = w.x & 0xffffu;  v[8*i+1] = w.x >> 16;
            v[8*i+2] = w.y & 0xffffu;  v[8*i+3] = w.y >> 16;
            v[8*i+4] = w.z & 0xffffu;  v[8*i+5] = w.z >> 16;
            v[8*i+6] = w.w & 0xffffu;  v[8*i+7] = w.w >> 16;
        }
    } else {
        const uint2 w = *(const uint2*)p;          // 4 个 u16
        v[0] = w.x & 0xffffu; v[1] = w.x >> 16;
        v[2] = w.y & 0xffffu; v[3] = w.y >> 16;
    }
}

template <int C>
__device__ __forceinline__ void col_st(uint8_t* __restrict__ p, const uint32_t acc[C], float inv)
{
    uint32_t o[C / 4];
#pragma unroll
    for (int i = 0; i < C / 4; ++i) {
        uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = __float2int_rn((float)acc[4*i+j] * inv);
            v = max(0, min(255, v));
            w |= ((uint32_t)v) << (8 * j);
        }
        o[i] = w;
    }
    if (C == 16) {
        *(uint4*)p = make_uint4(o[0], o[1], o[2], o[3]);
    } else if (C == 8) {
        *(uint2*)p = make_uint2(o[0], o[1]);
    } else {
        *(uint*)p = o[0];
    }
}

template <int C>
__global__ void vec_cols_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
                                uint8_t* __restrict__ dst, size_t dstep,
                                int W, int H, int kh, float inv_area, int rpt)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * C;
    if (c >= W) return;
    const int y0 = blockIdx.y * rpt;
    if (y0 >= H) return;
    const int y1 = min(H, y0 + rpt);
    const int half = kh >> 1;
    const size_t trows = tstep >> 1;

    if (c + C > W) {                     // 尾块：标量
        for (int y = y0; y < y1; ++y) {
            for (int i = c; i < W; ++i) {
                uint32_t s = 0;
                for (int j = -half; j <= half; ++j)
                    s += tmp[(size_t)mir(y + j, H) * trows + i];
                int v = __float2int_rn((float)s * inv_area);
                dst[(size_t)y * dstep + i] = (uint8_t)max(0, min(255, v));
            }
        }
        return;
    }

    uint32_t acc[C];
#pragma unroll
    for (int i = 0; i < C; ++i) acc[i] = 0;
    {
        uint32_t v[C];
        for (int j = -half; j <= half; ++j) {
            col_ld<C>(tmp + (size_t)mir(y0 + j, H) * trows + c, v);
#pragma unroll
            for (int i = 0; i < C; ++i) acc[i] += v[i];
        }
    }

    for (int y = y0; y < y1; ++y) {
        col_st<C>(dst + (size_t)y * dstep + c, acc, inv_area);
        if (y + 1 < y1) {
            uint32_t vn[C], vo[C];
            col_ld<C>(tmp + (size_t)mir(y + 1 + half, H) * trows + c, vn);
            col_ld<C>(tmp + (size_t)mir(y - half, H) * trows + c, vo);
#pragma unroll
            for (int i = 0; i < C; ++i) acc[i] += vn[i] - vo[i];
        }
    }
}

// host 侧启动
template <int C>
static void launch_vec(const uint16_t* tmp, size_t tstep, uint8_t* dst, size_t dstep,
                       int W, int H, int kh, float inv, int rpt)
{
    const int ng = idiv_up(W, C);
    int th = idiv_up(ng, 32) * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(ng, th), idiv_up(H, rpt));
    vec_cols_kernel<C><<<g, th>>>(tmp, tstep, dst, dstep, W, H, kh, inv, rpt);
}

template <typename F>
static double timeMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

template <typename FA, typename FB>
static void abMin(FA fa, FB fb, int warm, int reps, double& oa, double& ob)
{
    for (int i = 0; i < warm; ++i) { fa(); fb(); }
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ba = 1e9f, bb = 1e9f;
    for (int i = 0; i < reps; ++i) {
        float ms = 0.f;
        cudaEventRecord(a); fa(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < ba) ba = ms;
        cudaEventRecord(a); fb(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < bb) bb = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    oa = ba; ob = bb;
}

struct C2 { const char* n; int W, H; };
static const C2 kCs[] = {{"640x480", 640, 480}, {"1280x720", 1280, 720},
                         {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};

// 分配 tmp（u16，tstep = W*2 或补到 16B 对齐两种都给）+ 输出
struct Buf {
    uint16_t* tmp = nullptr; size_t tstep = 0;
    uint8_t* dst = nullptr;  size_t dstep = 0;
    int W = 0, H = 0;
    void make(int w, int hh, int pad16 = 0) {
        W = w; H = hh;
        tstep = (size_t)w * 2;
        if (pad16) tstep = (tstep + 15) & ~(size_t)15;
        CKV(cudaMalloc(&tmp, tstep * hh + 64));
        CKV(cudaMallocPitch(&dst, &dstep, (size_t)w, hh));
        std::vector<uint16_t> v(tstep / 2 * hh / 1 + 1);
        unsigned st = 91u;
        for (size_t i = 0; i < v.size(); ++i) {
            st = st * 1664525u + 1013904223u;
            v[i] = (uint16_t)((st >> 16) & 0x3fff);
        }
        CKV(cudaMemcpy(tmp, v.data(), tstep * hh, cudaMemcpyHostToDevice));
    }
    ~Buf() { if (tmp) cudaFree(tmp); if (dst) cudaFree(dst); }
};

} // namespace

int main(int argc, char** argv)
{
    CKV(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "check";
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    if (mode == "check") {
        // 正确性：vec<C> vs old_slide vs old_blocked vs CPU int64，全逐位
        struct S { int W, H; };
        const S shapes[] = {{16,1},{16,2},{16,3},{17,3},{31,3},{32,2},{33,17},{64,64},{100,100},
                            {127,129},{333,257},{640,480},{720,479},{1280,720},{1920,1080},
                            {4096,64},{7,400},{1281,721},{512,512},{3840,2160},{12,5},{8,1}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            for (int k : {1, 3, 15, 31, 33, 63}) {
                if (k > 2 * s.H + 1 && s.H > 1) { /* 镜像折返多次，CPU 参照同式即可 */ }
                Buf bf; bf.make(s.W, s.H, 1);
                const int npx = s.W * s.H;
                std::vector<uint8_t> ref(npx), got(npx), oldb(npx);
                // CPU 参照（int64 精确窗口和，float 乘 inv_area 后 rn）
                const int half = k >> 1;
                const float inv = 1.0f / (float)k;
                std::vector<uint16_t> host(bf.tstep / 2 * s.H);
                CKV(cudaMemcpy(host.data(), bf.tmp, bf.tstep * s.H, cudaMemcpyDeviceToHost));
                for (int y = 0; y < s.H; ++y) {
                    for (int x = 0; x < s.W; ++x) {
                        long long acc = 0;
                        for (int j = -half; j <= half; ++j) {
                            int yy = y + j;
                            if (yy < 0) yy = -yy;
                            if (yy >= s.H) yy = 2 * s.H - 2 - yy;
                            yy = std::max(0, std::min(s.H - 1, yy));
                            acc += host[(size_t)yy * (bf.tstep / 2) + x];
                        }
                        float f = (float)(uint32_t)acc * inv;
                        int v = (int)nearbyintf(f);
                        ref[(size_t)y * s.W + x] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
                    }
                }
                auto run = [&](int which, int C, int rpt, uint8_t* out) {
                    if (which == 0) {
                        dim3 g(idiv_up(s.W, 256), idiv_up(s.H, rpt));
                        old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rpt);
                    } else if (which == 1) {
                        dim3 g(idiv_up(s.W, 256));
                        old_slide<<<g, 256>>>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv);
                    } else if (C == 4)  launch_vec<4>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rpt);
                    else if (C == 8)    launch_vec<8>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rpt);
                    else                launch_vec<16>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rpt);
                    CKV(cudaDeviceSynchronize());
                };
                auto fetch_out = [&](uint8_t* d) {
                    std::vector<uint8_t> v(npx);
                    CKV(cudaMemcpy2D(v.data(), (size_t)s.W, d, bf.dstep, (size_t)s.W, s.H,
                                     cudaMemcpyDeviceToHost));
                    return v;
                };
                // 注意：必须按 dstep*(H-1)+W 分配（cudaMemcpy2D 的契约），按 W*H 分在 H>=2 时会
                // 报 invalid argument（本轮踩坑）
                uint8_t* out = nullptr;
                CKV(cudaMalloc(&out, bf.dstep * (size_t)(s.H > 0 ? s.H - 1 : 0) + (size_t)s.W + 64));
                run(0, 0, 16, out); oldb = fetch_out(out);
                size_t badOld = 0, firstBad = (size_t)-1;
                for (size_t i = 0; i < (size_t)npx; ++i) {
                    if (oldb[i] != ref[i]) { ++badOld; if (firstBad == (size_t)-1) firstBad = i; }
                }
                size_t badNew = 0;
                for (int C : {4, 8, 16}) {
                    for (int rpt : {7, 16, 64, 1000000}) {
                        run(2, C, rpt, out); got = fetch_out(out);
                        for (size_t i = 0; i < (size_t)npx; ++i)
                            if (got[i] != oldb[i]) { ++badNew; break; }
                    }
                }
                ++total;
                const bool ok = (badNew == 0 && badOld == 0);
                printf("  %4dx%-4d k=%-2d %s%s\n", s.W, s.H, k, ok ? "[OK]" : "[FAIL]",
                       ok ? "" : " (new!=old or old!=cpu)");
                if (!ok) {
                    printf("     old_vs_cpu bad=%zu first=%zu | new_vs_old bad=%zu\n", badOld, firstBad, badNew);
                    ++fails;
                }
                cudaFree(out);
            }
        }
        // 两种 tstep（紧排 vs 补 16）都要过
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MCOL2_FAIL" : "PROBE_MCOL2_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "sweep") {
        const int rpts[] = {7, 16, 32, 64, 128, 256, 1000000};
        for (const C2& c : kCs) {
            Buf bf; bf.make(c.W, c.H, 1);
            const int npx = c.W * c.H;
            const double fl = timeMin([&] {
                dim3 g(idiv_up(c.W, 16 * 256), c.H);
                floor_copy<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W);
            }, 5, reps);
            printf("== %s  floor(u16->u8) %.4f ms (%.1f GB/s)\n", c.n, fl, 3.0 * npx / fl / 1e6);
            for (int k : {15, 31}) {
                const float inv = 1.0f / (float)k;
                double ob = 0, os = 0;
                abMin([&] { dim3 g(idiv_up(c.W, 256), idiv_up(c.H, c.H > 1080 ? 64 : 16));
                            old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv,
                                                    c.H > 1080 ? 64 : 16); },
                      [&] { dim3 g(idiv_up(c.W, 256));
                            old_slide<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv); },
                      3, reps, ob, os);
                printf("   k=%-2d old_blocked %8.4f (%.1f GB/s) | old_slide %8.4f (%.1f GB/s)\n",
                       k, ob, 3.0 * npx / ob / 1e6, os, 3.0 * npx / os / 1e6);
                for (int C : {4, 8, 16}) {
                    printf("   k=%-2d  ", k);
                    for (int rpt : rpts) {
                        double t = timeMin([&] {
                            if (C == 4)  launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, rpt);
                            else if (C == 8) launch_vec<8>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, rpt);
                            else launch_vec<16>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, rpt);
                        }, 3, reps);
                        printf("C%d/rpt%-7d %7.4f(%4.1fG) ", C, rpt == 1000000 ? 999999 : rpt, t, 3.0 * npx / t / 1e6);
                    }
                    printf("\n");
                }
            }
        }
        return 0;
    }

    printf("usage: probe_mcol2 [check|sweep] [reps]\n");
    return 0;
}