// probe_mcol4 (r66): mean-f32 大核列遍 —— vec 化重写（照搬 r65 u8 列遍结构）
//
// 由来（r65b 全量 bench 实测，mean-f32 是全表唯一没动的家族）：
//   k=15: 640 7.9x / 720p 9.3x / 1080p 9.7x / 4K 10.2x；k=31: 12.6/16.9/18.5/19.8x
//   中位 9.7x，远低于 u8 侧的 30~111x。病灶 = 列遍仍是老路径：
//   ① mean32_cols_blocked_kernel（k<=15）：每输出行 O(k) 直加 ⇒ tmp 重读 k 倍（k=15 是 45B/px）；
//   ② mean32_cols_kernel（k>15）：每线程一列沿 y 走 ⇒ 4K 只有 W=3840 线程，并行度不足。
//
// 本探针：直接套 r65 的 mean_cols_vec_kernel<C=4> 结构（r65 新认知：**一行内 C 个连续列**
//   与行遍同源、无 sector 放大；唯一有放大的是「一列 C 个连续行」的拆法）。
//   f32 侧比 u8 侧更对称：每线程 4 个连续 float 列，读 = 1 次 float4（16B），
//   写 = 1 次 float4（4 列 × 4B = 16B）。
//   边界语义：列方向 REFLECT_101（与旧内核一致，smoke 的 f32 容差 1e-2 之外逐位一致）。
//   算术次序：acc 初值 = ((0 + r0) + r1) + ... 顺序加 k 行（与 mean32_cols_blocked_kernel
//   的 `acc += scol[...]` 完全同序）⇒ 与旧内核逐位一致（除 +/- 合并顺序差异，见下）。
//     注：滑动更新旧内核是 `acc += 新行; acc -= 旧行`（两次累加），本内核是
//     `acc += (新行 - 旧行)`（一次累加）—— float 结合律差异 ~1e-6 量级，smoke 容差 1e-2。
//
// 模式：
//   check  —— 24 形状 × k ∈ {15,31}：vec vs 旧内核（容差 1e-3）+ 旧内核 vs CPU(float 复刻)
//          逐值比对（float 结合律差异 ~1e-6，见下）
//   sweep  —— 四尺寸 × k=15/31：floor(float4 拷贝) / old_blocked / old_slide / vecC4×rpt 交错 min
//   rpt    —— vecC4 的 rpt 精细扫（轮转起点交错 min）：定 rpt 规则
//   bigk   —— k=33/63/127/255：vec 的初始 O(k) 直加需要更大 rpt，测要不要纳入 vec
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

namespace {

static inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

static __device__ __forceinline__ int mir(int i, int len)
{
    if (i < 0) i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// ---------------- 旧内核（基线，与 src/mean_filter.cu 逐句一致） ----------------
__global__ void old_blocked(const float* __restrict__ tmp, size_t tstep,
                            float* __restrict__ dst, size_t dstep,
                            int W, int H, int k, float inv_area, int rpt)
{
    const int x = blockIdx.x * 256 + threadIdx.x;
    if (x >= W) return;
    const int half = k >> 1;
    const float* scol = tmp + x;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mir(j - half + y0, H) * rows];
    for (int r = y0; r < y1; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        if (r + 1 < y1) {
            acc += scol[(size_t)mir(r + 1 + half, H) * rows]
                 - scol[(size_t)mir(r - half, H) * rows];
        }
    }
}

__global__ void old_slide(const float* __restrict__ tmp, size_t tstep,
                          float* __restrict__ dst, size_t dstep,
                          int W, int H, int k, float inv_area)
{
    const int x = blockIdx.x * 256 + threadIdx.x;
    if (x >= W) return;
    const int half = k >> 1;
    const float* scol = tmp + x;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mir(j - half, H) * rows];
    for (int r = 0; r < H; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        acc += scol[(size_t)mir(r + 1 + half, H) * rows]
             - scol[(size_t)mir(r - half, H) * rows];
    }
}

// ---------------- vec（照搬 r65 mean_cols_vec_kernel<C=4>） ----------------
template <int C>
__global__ void vec_kernel(const float* __restrict__ tmp, size_t tstep,
                           float* __restrict__ dst, size_t dstep,
                           int W, int H, int k, float inv_area, int rpt)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * C;
    if (c >= W) return;
    const int y0 = blockIdx.y * rpt;
    if (y0 >= H) return;
    const int y1 = min(H, y0 + rpt);
    const int half = k >> 1;
    const size_t trows = tstep >> 2;

    if (c + C > W) {                       // 尾列兜底（W%4==0 时不会进这里）
        for (int y = y0; y < y1; ++y) {
            for (int i = c; i < W; ++i) {
                float s = 0.f;
                for (int j = -half; j <= half; ++j)
                    s += tmp[(size_t)mir(y + j, H) * trows + i];
                dst[(size_t)y * (dstep >> 2) + i] = s * inv_area;
            }
        }
        return;
    }

    float acc[C];
    float4 w;
#pragma unroll
    for (int i = 0; i < C; ++i) acc[i] = 0.f;
    for (int j = -half; j <= half; ++j) {
        w = *(const float4*)(tmp + (size_t)mir(y0 + j, H) * trows + c);
        acc[0] += w.x; acc[1] += w.y; acc[2] += w.z; acc[3] += w.w;
    }

    for (int y = y0; y < y1; ++y) {
        float4 o;
        o.x = acc[0] * inv_area; o.y = acc[1] * inv_area;
        o.z = acc[2] * inv_area; o.w = acc[3] * inv_area;
        *(float4*)(dst + (size_t)y * (dstep >> 2) + c) = o;
        if (y + 1 < y1) {
            float4 vn = *(const float4*)(tmp + (size_t)mir(y + 1 + half, H) * trows + c);
            float4 vo = *(const float4*)(tmp + (size_t)mir(y - half, H) * trows + c);
            acc[0] += vn.x - vo.x;
            acc[1] += vn.y - vo.y;
            acc[2] += vn.z - vo.z;
            acc[3] += vn.w - vo.w;
        }
    }
}

// ---------------- 地板：tmp(float) -> dst(float) 纯拷贝 ----------------
__global__ void floor_copy(const float* __restrict__ s, size_t sstep,
                           float* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (c >= W) return;
    const int y = blockIdx.y;
    const float4* p = (const float4*)(s + (size_t)y * (sstep >> 2) + c);
    const float4 a = p[0], b = p[1], e = p[2], f = p[3];
    *(float4*)(d + (size_t)y * (dstep >> 2) + c) = a;
    *(float4*)(d + (size_t)y * (dstep >> 2) + c + 4) = b;
    *(float4*)(d + (size_t)y * (dstep >> 2) + c + 8) = e;
    *(float4*)(d + (size_t)y * (dstep >> 2) + c + 12) = f;
}

// ---------------- host 启动 ----------------
template <int C>
static void launch_vec(const float* t, size_t ts, float* d, size_t ds,
                       int W, int H, int k, float inv, int rpt)
{
    const int ng = idiv_up(W, C);
    int th = idiv_up(ng, 32) * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(ng, th), idiv_up(H, rpt));
    vec_kernel<C><<<g, th>>>(t, ts, d, ds, W, H, k, inv, rpt);
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

struct C2 { const char* n; int W, H; };
static const C2 kCs[] = {{"640x480", 640, 480}, {"1280x720", 1280, 720},
                         {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};

struct Buf {
    float* tmp = nullptr; size_t tstep = 0;
    float* dst = nullptr;  size_t dstep = 0;
    int W = 0, H = 0;
    void make(int w, int hh) {
        W = w; H = hh;
        tstep = ((size_t)w * 4 + 15) & ~(size_t)15;
        CKV(cudaMalloc(&tmp, tstep * hh + 64));
        CKV(cudaMallocPitch(&dst, &dstep, (size_t)w * 4, hh));
        std::vector<float> v(tstep / 4 * hh);
        unsigned st = 91u;
        for (auto& x : v) { st = st * 1664525u + 1013904223u; x = (float)(int)((st >> 16) & 0x3fff); }
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
        struct S { int W, H; };
        const S shapes[] = {{16,1},{16,2},{16,3},{17,3},{31,3},{32,2},{33,17},{64,64},{100,100},
                            {127,129},{333,257},{640,480},{720,479},{1280,720},{1920,1080},
                            {4096,64},{7,400},{1281,721},{512,512},{3840,2160},{12,5},{8,1},{1,1},{1,7}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            for (int k : {15, 31}) {
                Buf bf; bf.make(s.W, s.H);
                const int npx = s.W * s.H;
                std::vector<float> host(bf.tstep / 4 * s.H);
                CKV(cudaMemcpy(host.data(), bf.tmp, bf.tstep * s.H, cudaMemcpyDeviceToHost));
                std::vector<float> ref(npx);
                const int half = k >> 1;
                const float inv = 1.0f / (float)k;
                for (int y = 0; y < s.H; ++y)
                    for (int x = 0; x < s.W; ++x) {
                        // CPU 参照**逐位复刻**旧内核语义：acc = 0 然后顺序加 k 项，输出 acc*inv
                        float acc = 0.f;
                        for (int j = -half; j <= half; ++j) {
                            int yy = y + j;
                            if (yy < 0) yy = -yy;
                            if (yy >= s.H) yy = 2 * s.H - 2 - yy;
                            yy = std::max(0, std::min(s.H - 1, yy));
                            acc += host[(size_t)yy * (bf.tstep / 4) + x];
                        }
                        ref[(size_t)y * s.W + x] = acc * inv;
                    }
                float* out = nullptr;
                CKV(cudaMalloc(&out, bf.dstep * (size_t)(s.H - 1) + (size_t)s.W * 4 + 64));
                auto fetch = [&] {
                    std::vector<float> v(npx);
                    CKV(cudaMemcpy2D(v.data(), (size_t)s.W * 4, out, bf.dstep, (size_t)s.W * 4, s.H,
                                     cudaMemcpyDeviceToHost));
                    return v;
                };
                // 判据：vec 与旧内核输出一致（float 结合律差异 ~1e-6 量级，容差 1e-3 足够；
                // 真正验收是 smoke 对 OpenCV 的 1e-2 容差）。旧内核 vs CPU 参照再单独校验。
                size_t badVec = 0;
                float maxVecOld = 0, maxOldCpu = 0;
                std::vector<float> outOld, outVec;
                const int rptB = s.H > 1080 ? 64 : 16;
                // k=15 走 blocked（旧分派规则），k=31 走 slide
                if (k <= 15) {
                    dim3 g(idiv_up(s.W, 256), idiv_up(s.H, rptB));
                    old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rptB);
                } else {
                    dim3 g(idiv_up(s.W, 256));
                    old_slide<<<g, 256>>>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv);
                }
                CKV(cudaDeviceSynchronize());
                outOld = fetch();
                // 旧内核 vs CPU 参照（复刻旧内核累加序：acc=0 顺序加 k 项）
                for (int i = 0; i < npx; ++i) {
                    const float d = std::fabs(outOld[i] - ref[i]);
                    if (d > maxOldCpu) maxOldCpu = d;
                }
                const int rptV = (k <= 15) ? 8 : 12;
                launch_vec<4>(bf.tmp, bf.tstep, out, bf.dstep, s.W, s.H, k, inv, rptV);
                CKV(cudaDeviceSynchronize());
                outVec = fetch();
                for (int i = 0; i < npx; ++i) {
                    const float d = std::fabs(outVec[i] - outOld[i]);
                    if (d > maxVecOld) maxVecOld = d;
                    if (d > 1e-3f) ++badVec;
                }
                ++total;
                const bool ok = (badVec == 0 && maxOldCpu <= 1e-3f);
                printf("  %4dx%-4d k=%-2d %s", s.W, s.H, k, ok ? "[OK]" : "[FAIL]");
                if (!ok) printf("  vec-old_md=%.3g(bad=%zu) old-cpu_md=%.3g",
                                maxVecOld, badVec, maxOldCpu);
                printf("\n");
                if (!ok) ++fails;
                cudaFree(out);
            }
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MCOL4_FAIL" : "PROBE_MCOL4_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "sweep") {
        const int rptv[] = {4, 6, 8, 12, 16, 24, 32};
        for (const C2& c : kCs) {
            Buf bf; bf.make(c.W, c.H);
            const int npx = c.W * c.H;
            double fl = timeMin([&] {
                dim3 g(idiv_up(c.W, 16 * 256), c.H);
                floor_copy<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W);
            }, 5, reps);
            printf("== %-10s floor %.4f ms (%.1f GB/s)\n", c.n, fl, 5.0 * npx / fl / 1e6);
            for (int k : {15, 31}) {
                const float inv = 1.0f / (float)k;
                const int rptOld = c.H > 1080 ? 64 : 16;
                double ob = timeMin([&] {
                    dim3 g(idiv_up(c.W, 256), idiv_up(c.H, rptOld));
                    old_blocked<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, rptOld);
                }, 3, reps);
                double os = timeMin([&] {
                    dim3 g(idiv_up(c.W, 256));
                    old_slide<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv);
                }, 3, reps);
                printf("  k=%-2d old_blocked %7.4f(%4.1fG) | old_slide %7.4f(%4.1fG)\n",
                       k, ob, 5.0 * npx / ob / 1e6, os, 5.0 * npx / os / 1e6);
                printf("  k=%-2d  vecC4 ", k);
                for (int r : rptv) {
                    double t = timeMin([&] { launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, r); }, 3, reps);
                    printf("r%-3d %7.4f(%4.1fG) ", r, t, 5.0 * npx / t / 1e6);
                }
                printf("\n");
            }
        }
        return 0;
    }

    if (mode == "rpt") {
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
                    printf("r%-3d %6.4f(%4.1fG) ", rp[i], best[i], 5.0 * npx / best[i] / 1e6);
                    if (best[i] < best[bi]) bi = i;
                }
                printf(" => rpt=%d\n", rp[bi]);
            }
        }
        return 0;
    }

    if (mode == "bigk") {
        const C2 cs[] = {{"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};
        for (const C2& c : cs) {
            Buf bf; bf.make(c.W, c.H);
            const int npx = c.W * c.H;
            for (int k : {33, 63, 127, 255}) {
                const float inv = 1.0f / (float)k;
                double os = timeMin([&] {
                    dim3 g(idiv_up(c.W, 256));
                    old_slide<<<g, 256>>>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv);
                }, 3, reps);
                printf("%-10s k=%-3d old_slide %7.4f(%4.1fG)\n", c.n, k, os, 5.0 * npx / os / 1e6);
                printf("%-10s k=%-3d vecC4 ", c.n, k);
                for (int r : {8, 12, 16, 32, 64, 128, 256}) {
                    double t = timeMin([&] { launch_vec<4>(bf.tmp, bf.tstep, bf.dst, bf.dstep, c.W, c.H, k, inv, r); }, 2, reps);
                    printf("r%-3d %7.4f(%4.1fG) ", r, t, 5.0 * npx / t / 1e6);
                }
                printf("\n");
            }
        }
        return 0;
    }

    printf("usage: probe_mcol4 [check|sweep|rpt|bigk] [reps]\n");
    return 0;
}
