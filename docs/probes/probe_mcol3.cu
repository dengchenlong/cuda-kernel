// probe_mcol3 (r65): mean 列遍 —— 寄存器 ring 版（每行 tmp 只读一次）vs 无 ring 版（读两次）
//
// 由来（probe_mcol2 sweep 实测，4K，u16->u8 地板 0.2611ms / 95.3 GB/s）：
//   old_blocked k=15 3.1242（每输出行 O(15) 直加 = 45B/px）| old_slide k=31 0.9308（5B/px）
//   vec C4/rpt7  k=15 0.3318（75.0 GB/s）| k=31 0.4413（56.4 GB/s）
//   ⇒ 两个残差：
//     ① vec 的滑动每行要读 **两次** tmp（vn=y+1+half 与 vo=y-half 是同一物理行的两次访问，
//        相隔 half 次迭代，超出寄存/L1 复用距离）⇒ 流量 2+2+1 = 5B/px，理论上限 = 0.261*5/3 = 0.435ms。
//        4K k=31 实测 0.4413 正好贴这条 5B/px 线（不是随机噪声）。
//     ② 因此唯一再降的办法是「每行只读一次」= 3B/px。用**寄存器 ring** 保存 k 个窗口值：
//        滑动时「+新 − 旧」且旧值直接从 ring 取，不再回读 tmp。
//
// 关键设计（否则 ring 会落 local memory，r54 的坑）：
//   不让 ring 旋转（旋转 = 动态下标 → local memory），而是**外层按 K 行一组展开**，
//   内层 j 是编译期常量 ⇒ v[j] 下标恒为常量，全部留寄存器、零 MOVE 指令。
//   j 轮转一圈回到同一位置时，v[j] 恰好是 K 行前的值 = 本轮最旧的元素，语义自洽。
//
// 对比：ring<C=1,K>、ring<C=2,K>（C=2 把 store 指令数减半）vs vec C4/rpt7 vs 旧内核
// 模式：check（vs CPU int64 逐值）/ sweep（四尺寸 × k=15/31 × {C,rpt} 交错 min）
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

static __device__ __forceinline__ int q8(uint32_t s, float inv)
{
    int v = __float2int_rn((float)s * inv);
    return max(0, min(255, v));
}

// ---------------- 旧内核（基线） ----------------
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
        dst[(size_t)y * dstep + x] = (uint8_t)q8(sum, inv_area);
    }
}

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
        dst[(size_t)y * dstep + x] = (uint8_t)q8(sum, inv_area);
        if (y + 1 < H) sum += fetch(y + 1 + half) - fetch(y - half);
    }
}

// ---------------- vec（probe_mcol2 定案形态：C 连续列，滑动读两次） ----------------
template <int C>
__device__ __forceinline__ void col_ld(const uint16_t* __restrict__ p, uint32_t v[C])
{
    if (C < 4) {              // C=1/2：标量载入（4B 对齐仍在，只是不满 8B）
        const uint16_t* q = p;
#pragma unroll
        for (int i = 0; i < C; ++i) v[i] = q[i];
    } else if (C >= 8) {
#pragma unroll
        for (int i = 0; i < C / 8; ++i) {
            const uint4 w = *(const uint4*)(p + 8 * i);
            v[8*i+0] = w.x & 0xffffu;  v[8*i+1] = w.x >> 16;
            v[8*i+2] = w.y & 0xffffu;  v[8*i+3] = w.y >> 16;
            v[8*i+4] = w.z & 0xffffu;  v[8*i+5] = w.z >> 16;
            v[8*i+6] = w.w & 0xffffu;  v[8*i+7] = w.w >> 16;
        }
    } else {
        const uint2 w = *(const uint2*)p;
        v[0] = w.x & 0xffffu; v[1] = w.x >> 16;
        v[2] = w.y & 0xffffu; v[3] = w.y >> 16;
    }
}

template <int C>
__device__ __forceinline__ void col_st(uint8_t* __restrict__ p, const uint32_t acc[C], float inv)
{
    if (C == 1) {
        *p = (uint8_t)q8(acc[0], inv);
        return;
    } else if (C == 2) {
        p[0] = (uint8_t)q8(acc[0], inv);
        p[1] = (uint8_t)q8(acc[1], inv);
        return;
    }
    uint32_t o[(C / 4) ? (C / 4) : 1];   // C<4 时分支已 return，这里只为让模板实例化合法
#pragma unroll
    for (int i = 0; i < C / 4; ++i) {
        uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) w |= ((uint32_t)q8(acc[4*i+j], inv)) << (8 * j);
        o[i] = w;
    }
    if (C == 16)      *(uint4*)p = make_uint4(o[0], o[1], o[2], o[3]);
    else if (C == 8)  *(uint2*)p = make_uint2(o[0], o[1]);
    else              *(uint*)p  = o[0];
}

template <int C>
__global__ void vec_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
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
    if (c + C > W) return;      // sweep 模式下尾块由 ring/CLAMP 版本单独覆盖，这里只测主体
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

// ---------------- ring：每行只读一次，K 编译期常量，外层按 K 行展开 ----------------
template <int C, int K>
__global__ void ring_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
                            uint8_t* __restrict__ dst, size_t dstep,
                            int W, int H, float inv_area, int rpt)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * C;
    if (c >= W) return;
    const int y0 = blockIdx.y * rpt;
    if (y0 >= H) return;
    const int y1 = min(H, y0 + rpt);
    constexpr int half = K >> 1;
    const size_t trows = tstep >> 1;

    if (c + C > W) {    // 尾列：标量回退（保证 check 模式也覆盖到）
        for (int y = y0; y < y1; ++y) {
            for (int i = c; i < W; ++i) {
                uint32_t s = 0;
                for (int j = -half; j <= half; ++j)
                    s += tmp[(size_t)mir(y + j, H) * trows + i];
                dst[(size_t)y * dstep + i] = (uint8_t)q8(s, inv_area);
            }
        }
        return;
    }

    uint32_t v[K][C];
#pragma unroll
    for (int j = 0; j < K; ++j)
        col_ld<C>(tmp + (size_t)mir(y0 - half + j, H) * trows + c, v[j]);

    uint32_t s[C];
#pragma unroll
    for (int i = 0; i < C; ++i) s[i] = 0;
#pragma unroll
    for (int j = 0; j < K; ++j)
#pragma unroll
        for (int i = 0; i < C; ++i) s[i] += v[j][i];

    // 外层按 K 行一组展开：内层 j 编译期常量 ⇒ v[j] 恒为静态下标（不进 local memory）
    for (int yb = y0; yb < y1; yb += K) {
#pragma unroll
        for (int j = 0; j < K; ++j) {
            const int y = yb + j;
            if (y >= y1) break;
            col_st<C>(dst + (size_t)y * dstep + c, s, inv_area);
            const int yn = y + 1;
            if (yn < y1) {
                uint32_t nv[C];
                col_ld<C>(tmp + (size_t)mir(yn + half, H) * trows + c, nv);
#pragma unroll
                for (int i = 0; i < C; ++i) { s[i] += nv[i] - v[j][i]; v[j][i] = nv[i]; }
            }
        }
    }
}

// ---------------- 地板 ----------------
__global__ void floor_copy(const uint16_t* __restrict__ s, size_t sstep,
                           uint8_t* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (c >= W) return;
    const int y = blockIdx.y;
    const uint4* p = (const uint4*)((const uint8_t*)s + (size_t)y * sstep + (size_t)c * 2);
    const uint4 a = p[0], b = p[1];
    *(uint4*)(d + (size_t)y * dstep + c) = make_uint4(
        (a.x & 0xffu) | ((a.x >> 8) & 0xff00u) | ((a.y & 0xffu) << 16) | (((a.y >> 8) & 0xffu) << 24),
        (a.z & 0xffu) | ((a.z >> 8) & 0xff00u) | ((a.w & 0xffu) << 16) | (((a.w >> 8) & 0xffu) << 24),
        (b.x & 0xffu) | ((b.x >> 8) & 0xff00u) | ((b.y & 0xffu) << 16) | (((b.y >> 8) & 0xffu) << 24),
        (b.z & 0xffu) | ((b.z >> 8) & 0xff00u) | ((b.w & 0xffu) << 16) | (((b.w >> 8) & 0xffu) << 24));
}

// ---------------- host 启动 ----------------
template <int C>
static void launch_vec(const uint16_t* t, size_t ts, uint8_t* d, size_t ds,
                       int W, int H, int k, float inv, int rpt)
{
    const int ng = idiv_up(W, C);
    int th = idiv_up(ng, 32) * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(ng, th), idiv_up(H, rpt));
    if (C == 4) vec_kernel<4><<<g, th>>>(t, ts, d, ds, W, H, k, inv, rpt);
    else if (C == 8) vec_kernel<8><<<g, th>>>(t, ts, d, ds, W, H, k, inv, rpt);
    else vec_kernel<16><<<g, th>>>(t, ts, d, ds, W, H, k, inv, rpt);
}

template <int C, int K>
static void launch_ring(const uint16_t* t, size_t ts, uint8_t* d, size_t ds,
                        int W, int H, float inv, int rpt)
{
    const int ng = idiv_up(W, C);
    int th = idiv_up(ng, 32) * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(ng, th), idiv_up(H, rpt));
    ring_kernel<C, K><<<g, th>>>(t, ts, d, ds, W, H, inv, rpt);
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

struct Buf {
    uint16_t* tmp = nullptr; size_t tstep = 0;
    uint8_t* dst = nullptr;  size_t dstep = 0;
    int W = 0, H = 0;
    void make(int w, int hh) {
        W = w; H = hh;
        tstep = ((size_t)w * 2 + 15) & ~(size_t)15;
        CKV(cudaMalloc(&tmp, tstep * hh + 64));
        CKV(cudaMallocPitch(&dst, &dstep, (size_t)w, hh));
        std::vector<uint16_t> v(tstep / 2 * hh);
        unsigned st = 91u;
        for (auto& x : v) { st = st * 1664525u + 1013904223u; x = (uint16_t)((st >> 16) & 0x3fff); }
        CKV(cudaMemcpy(tmp, v.data(), tstep * hh, cudaMemcpyHostToDevice));
    }
    ~Buf() { if (tmp) cudaFree(tmp); if (dst) cudaFree(dst); }
};

static void run_kind(int kind, int C, int K, const Buf& bf, uint8_t* out, size_t ostep,
                     int W, int H, int k, float inv, int rpt)
{
    if (kind == 0) {
        dim3 g(idiv_up(W, 256), idiv_up(H, rpt));
        old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, out, ostep, W, H, k, inv, rpt);
    } else if (kind == 1) {
        dim3 g(idiv_up(W, 256));
        old_slide<<<g, 256>>>(bf.tmp, bf.tstep, out, ostep, W, H, k, inv);
    } else if (kind == 2) {
        if (C == 4) launch_vec<4>(bf.tmp, bf.tstep, out, ostep, W, H, k, inv, rpt);
        else if (C == 8) launch_vec<8>(bf.tmp, bf.tstep, out, ostep, W, H, k, inv, rpt);
        else launch_vec<16>(bf.tmp, bf.tstep, out, ostep, W, H, k, inv, rpt);
    } else {
        if (C == 1) { if (K == 15) launch_ring<1,15>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt);
                      else           launch_ring<1,31>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt); }
        else if (C == 2) { if (K == 15) launch_ring<2,15>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt);
                           else           launch_ring<2,31>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt); }
        else { if (K == 15) launch_ring<4,15>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt);
               else           launch_ring<4,31>(bf.tmp, bf.tstep, out, ostep, W, H, inv, rpt); }
    }
    CKV(cudaDeviceSynchronize());
}

} // namespace

int main(int argc, char** argv)
{
    CKV(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "check";
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    if (mode == "check") {
        struct S { int W, H; };
        const S shapes[] = {{16,1},{16,2},{16,3},{17,3},{31,3},{32,2},{33,17},{64,64},{100,100},
                            {127,129},{333,257},{640,480},{720,479},{1280,720},{1920,1080},
                            {4096,64},{7,400},{1281,721},{512,512},{3840,2160},{12,5},{8,1},{1,1},{1,7}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            for (int k : {15, 31}) {
                Buf bf; bf.make(s.W, s.H);
                const int npx = s.W * s.H;
                std::vector<uint16_t> host(bf.tstep / 2 * s.H);
                CKV(cudaMemcpy(host.data(), bf.tmp, bf.tstep * s.H, cudaMemcpyDeviceToHost));
                std::vector<uint8_t> ref(npx);
                const int half = k >> 1;
                const float inv = 1.0f / (float)k;
                for (int y = 0; y < s.H; ++y)
                    for (int x = 0; x < s.W; ++x) {
                        long long acc = 0;
                        for (int j = -half; j <= half; ++j) {
                            int yy = y + j;
                            if (yy < 0) yy = -yy;
                            if (yy >= s.H) yy = 2 * s.H - 2 - yy;
                            yy = std::max(0, std::min(s.H - 1, yy));
                            acc += host[(size_t)yy * (bf.tstep / 2) + x];
                        }
                        // CPU 参照必须**逐位复刻** GPU 的 float 路径（(float)acc*inv 单次 float 乘 +
                        // round-half-even），否则 .5 附近会假 FAIL —— probe_mcol2 已用同式全过
                        const float f = (float)(uint32_t)acc * inv;
                        const int vq = (int)nearbyintf(f);
                        ref[(size_t)y * s.W + x] = (uint8_t)(vq < 0 ? 0 : (vq > 255 ? 255 : vq));
                    }
                uint8_t* out = nullptr;
                CKV(cudaMalloc(&out, bf.dstep * (size_t)(s.H - 1) + s.W + 64));
                auto fetch = [&] {
                    std::vector<uint8_t> v(npx);
                    CKV(cudaMemcpy2D(v.data(), (size_t)s.W, out, bf.dstep, (size_t)s.W, s.H,
                                     cudaMemcpyDeviceToHost));
                    return v;
                };
                size_t badOld = 0, badVec = 0, badRing = 0;
                run_kind(1, 0, 0, bf, out, bf.dstep, s.W, s.H, k, inv, 16);
                { auto g = fetch(); for (int i = 0; i < npx; ++i) if (g[i] != ref[i]) ++badOld; }
                const int rpt = (s.H > 1080) ? 7 : 16;
                run_kind(2, 4, 0, bf, out, bf.dstep, s.W, s.H, k, inv, rpt);
                { auto g = fetch(); for (int i = 0; i < npx; ++i) if (g[i] != ref[i]) ++badVec; }
                for (int C : {1, 2, 4}) {
                    for (int r : {7, 16, 64, 1000000}) {
                        run_kind(3, C, k, bf, out, bf.dstep, s.W, s.H, k, inv, r);
                        auto g = fetch();
                        for (int i = 0; i < npx; ++i) if (g[i] != ref[i]) { ++badRing; break; }
                    }
                }
                ++total;
                const bool ok = (badOld == 0 && badVec == 0 && badRing == 0);
                printf("  %4dx%-4d k=%-2d %s", s.W, s.H, k, ok ? "[OK]" : "[FAIL]");
                if (!ok) printf("  old=%zu vec=%zu ring=%zu", badOld, badVec, badRing);
                printf("\n");
                if (!ok) ++fails;
                cudaFree(out);
            }
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MCOL3_FAIL" : "PROBE_MCOL3_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "sweep") {
        const int rptv[] = {7, 16, 32, 64, 128};
        for (const C2& c : kCs) {
            Buf bf; bf.make(c.W, c.H);
            const int npx = c.W * c.H;
            double fl = timeMin([&] {
                dim3 g(idiv_up(c.W, 16 * 256), c.H);
                floor_copy<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W);
            }, 5, reps);
            printf("== %-10s floor %.4f ms (%.1f GB/s)\n", c.n, fl, 3.0 * npx / fl / 1e6);
            for (int k : {15, 31}) {
                const float inv = 1.0f / (float)k;
                const int rptOld = c.H > 1080 ? 64 : 16;
                double ob = 0, os = 0;
                abMin([&] { dim3 g(idiv_up(c.W, 256), idiv_up(c.H, rptOld));
                            old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, rptOld); },
                      [&] { dim3 g(idiv_up(c.W, 256));
                            old_slide<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv); },
                      3, reps, ob, os);
                printf("  k=%-2d old_blocked %7.4f(%4.1fG) | old_slide %7.4f(%4.1fG)\n",
                       k, ob, 3.0 * npx / ob / 1e6, os, 3.0 * npx / os / 1e6);
                // vec C4 × 全部 rpt（对照）
                printf("  k=%-2d  vecC4 ", k);
                for (int r : rptv) {
                    double t = timeMin([&] { launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, r); }, 3, reps);
                    printf("r%-4d %7.4f(%4.1fG) ", r, t, 3.0 * npx / t / 1e6);
                }
                printf("\n");
                for (int C : {1, 2, 4}) {
                    printf("  k=%-2d  ringC%d ", k, C);
                    for (int r : rptv) {
                        double t = timeMin([&] {
                            if (C == 1) { if (k == 15) launch_ring<1,15>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r);
                                          else         launch_ring<1,31>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r); }
                            else if (C == 2) { if (k == 15) launch_ring<2,15>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r);
                                               else         launch_ring<2,31>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r); }
                            else { if (k == 15) launch_ring<4,15>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r);
                                   else         launch_ring<4,31>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, inv, r); }
                        }, 3, reps);
                        printf("r%-4d %7.4f(%4.1fG) ", r, t, 3.0 * npx / t / 1e6);
                    }
                    printf("\n");
                }
            }
        }
        return 0;
    }

    if (mode == "rpt") {
        // 精细扫 vecC4 的 rpt（交错多轮 min），只测 1080p / 4K —— 小图由同一规则外推
        const C2 cs[] = {{"1280x720", 1280, 720}, {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160},
                         {"640x480", 640, 480}};
        const int rp[] = {3, 4, 6, 8, 12, 16, 24, 32, 48};
        const int NR = (int)(sizeof(rp) / sizeof(rp[0]));
        for (const C2& c : cs) {
            Buf bf; bf.make(c.W, c.H);
            const int npx = c.W * c.H;
            for (int k : {15, 31}) {
                const float inv = 1.0f / (float)k;
                double best[16];
                for (int i = 0; i < NR; ++i) best[i] = 1e9;
                // 轮转起点交错：每轮从不同 rpt 起步，压掉时钟漂移的排序偏置
                for (int round = 0; round < 3; ++round) {
                    for (int s = 0; s < NR; ++s) {
                        const int i = (s + round * 3) % NR;
                        const int r = rp[i];
                        double t = timeMin([&] { launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, r); }, 2, reps);
                        if (t < best[i]) best[i] = t;
                    }
                }
                printf("%-10s k=%-2d ", c.n, k);
                int bi = 0;
                for (int i = 0; i < NR; ++i) {
                    printf("r%-3d %6.4f(%4.1fG) ", rp[i], best[i], 3.0 * npx / best[i] / 1e6);
                    if (best[i] < best[bi]) bi = i;
                }
                printf(" => rpt=%d\n", rp[bi]);
            }
        }
        return 0;
    }

    if (mode == "bigk") {
        // 大核（k=63/127/255）对比：vec 的初始 O(k) 直加要靠 rpt 摊薄，rpt 必须随 k 放大
        const C2 cs[] = {{"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};
        for (const C2& c : cs) {
            Buf bf; bf.make(c.W, c.H);
            const int npx = c.W * c.H;
            for (int k : {33, 63, 127, 255}) {
                const float inv = 1.0f / (float)k;
                double ob = 0, os = 0;
                abMin([&] { dim3 g(idiv_up(c.W, 256), idiv_up(c.H, 64));
                            old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, 64); },
                      [&] { dim3 g(idiv_up(c.W, 256));
                            old_slide<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv); },
                      3, reps, ob, os);
                printf("%-10s k=%-3d old_blocked %7.4f(%4.1fG) | old_slide %7.4f(%4.1fG)\n",
                       c.n, k, ob, 3.0 * npx / ob / 1e6, os, 3.0 * npx / os / 1e6);
                printf("%-10s k=%-3d vecC4 ", c.n, k);
                for (int r : {8, 12, 16, 32, 64, 128, 256}) {
                    double t = timeMin([&] { launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, r); }, 2, reps);
                    printf("r%-3d %7.4f(%4.1fG) ", r, t, 3.0 * npx / t / 1e6);
                }
                printf("\n");
            }
        }
        return 0;
    }

    printf("usage: probe_mcol3 [check|sweep|rpt|bigk] [reps]\n");
    return 0;
}