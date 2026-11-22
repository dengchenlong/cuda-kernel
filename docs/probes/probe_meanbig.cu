// probe_meanbig (r64):
// 两个问题一次测完（r59 教训：地板必须分尺寸实测，不能拿 4K 的地板套小图）。
//  A) mean 大核（k=15/31）两遍路径的分段归因。手法同 probe_mcol/probe_mf32：
//     用非对称核 (1,1) 基线 / (31,1) 只放大行遍 / (1,31) 只放大列遍 / (31,31) 全体。
//     背景：4K k=31 = 4.72ms，6N 流量（读1N写2N + 读2N写1N = 49.8MB）折算只有 10.6 GB/s，
//     而 k=3 融合版 0.28ms。差 16 倍 —— 到底是行遍（Hillis-Steele 512 元扫描 18 屏障/块，
//     4K 有 32400 个块）还是列遍（每线程一列，全场只有 W=3840 线程）？
//  B) 各尺寸搬运地板（为候选①「gaussian-f32 k=5 小图 / mean k=3 小图」定案用）：
//     noop（纯启动固定成本）、u8->u16（行遍形状 1N:2N）、u16->u8（列遍形状 2N:1N）、
//     f32->f32（融合形状 8B/px）。
// 计时：交错 A/B 取 min，多轮。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <functional>
#include "ops.h"

#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

namespace {

typedef unsigned int uint;

static inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

__global__ void k_noop() {}

// u8 -> u16（行遍形状：每线程读 16B，写 32B）
__global__ void k_u8_u16(const uint8_t* __restrict__ s, size_t sstep,
                         uint16_t* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (c >= W) return;
    const int y = blockIdx.y;
    const uint4 v = *(const uint4*)(s + (size_t)y * sstep + c);
    const uint t[8] = {
        (v.x & 0xffu)        | (((v.x >> 8)  & 0xffu) << 16),
        ((v.x >> 16) & 0xffu) | ((v.x >> 24)          << 16),
        (v.y & 0xffu)        | (((v.y >> 8)  & 0xffu) << 16),
        ((v.y >> 16) & 0xffu) | ((v.y >> 24)          << 16),
        (v.z & 0xffu)        | (((v.z >> 8)  & 0xffu) << 16),
        ((v.z >> 16) & 0xffu) | ((v.z >> 24)          << 16),
        (v.w & 0xffu)        | (((v.w >> 8)  & 0xffu) << 16),
        ((v.w >> 16) & 0xffu) | ((v.w >> 24)          << 16),
    };
    uint4* q = (uint4*)((uint8_t*)d + (size_t)y * dstep + (size_t)c * 2);
    q[0] = make_uint4(t[0], t[1], t[2], t[3]);
    q[1] = make_uint4(t[4], t[5], t[6], t[7]);
}

// u16 -> u8（列遍形状：每线程读 32B，写 16B）
__global__ void k_u16_u8(const uint16_t* __restrict__ s, size_t sstep,
                         uint8_t* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (c >= W) return;
    const int y = blockIdx.y;
    const uint4* p = (const uint4*)((const uint8_t*)s + (size_t)y * sstep + (size_t)c * 2);
    const uint4 a = p[0], b = p[1];
    const uint o[4] = {
        (a.x & 0xffu) | ((a.x >> 8) & 0xff00u) | ((a.y & 0xffu) << 16) | (((a.y >> 8) & 0xffu) << 24),
        (a.z & 0xffu) | ((a.z >> 8) & 0xff00u) | ((a.w & 0xffu) << 16) | (((a.w >> 8) & 0xffu) << 24),
        (b.x & 0xffu) | ((b.x >> 8) & 0xff00u) | ((b.y & 0xffu) << 16) | (((b.y >> 8) & 0xffu) << 24),
        (b.z & 0xffu) | ((b.z >> 8) & 0xff00u) | ((b.w & 0xffu) << 16) | (((b.w >> 8) & 0xffu) << 24),
    };
    *(uint4*)(d + (size_t)y * dstep + c) = make_uint4(o[0], o[1], o[2], o[3]);
}

// f32 -> f32（融合内核形状：每线程 float4 读 + float4 写 = 8B/px）
__global__ void k_f32cp(const float* __restrict__ s, size_t sstep,
                        float* __restrict__ d, size_t dstep, int W)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (c >= W) return;
    const int y = blockIdx.y;
    *(float4*)(d + (size_t)y * (dstep >> 2) + c) =
        *(const float4*)(s + (size_t)y * (sstep >> 2) + c);
}

template <typename F>
double timeMin(F f, int warm, int reps)
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

struct Img {
    uint8_t* d = nullptr; size_t step = 0; int W = 0, H = 0;
    void make(int w, int h) {
        W = w; H = h;
        CKV(cudaMallocPitch(&d, &step, (size_t)w, h));
        std::vector<uint8_t> v((size_t)w * h);
        unsigned st = 12345u + (unsigned)(w * 7 + h);
        for (auto& x : v) { st = st * 1664525u + 1013904223u; x = (uint8_t)(st >> 24); }
        CKV(cudaMemcpy2D(d, step, v.data(), (size_t)w, (size_t)w, h, cudaMemcpyHostToDevice));
    }
    ~Img() { if (d) cudaFree(d); }
};

struct FImg {
    float* d = nullptr; size_t step = 0; int W = 0, H = 0;
    void make(int w, int h) {
        W = w; H = h;
        CKV(cudaMallocPitch(&d, &step, (size_t)w * 4, h));
        std::vector<float> v((size_t)w * h);
        unsigned st = 999u + (unsigned)(w * 7 + h);
        for (auto& x : v) { st = st * 1664525u + 1013904223u; x = (float)((st >> 8) & 0xffff) / 7.0f; }
        CKV(cudaMemcpy2D(d, step, v.data(), (size_t)w * 4, (size_t)w * 4, h, cudaMemcpyHostToDevice));
    }
    ~FImg() { if (d) cudaFree(d); }
};

struct C { const char* n; int W, H; };
const C kCs[] = {{"640x480", 640, 480}, {"1280x720", 1280, 720},
                 {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};

} // namespace

int main(int argc, char** argv)
{
    CKV(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "asym";
    const int reps = (argc > 2) ? atoi(argv[2]) : 15;

    if (mode == "asym") {
        // gpu_mean 的 (1,1)/(31,1)/(1,31)/(31,31)：非对称核把行遍/列遍分别放大
        for (const C& c : kCs) {
            Img img; img.make(c.W, c.H);
            double e11 = 0, e31r = 0, e31c = 0, e31 = 0;
            e11  = timeMin([&] { gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 1, 1); }, 3, reps);
            e31r = timeMin([&] { gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 31, 1); }, 3, reps);
            e31c = timeMin([&] { gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 1, 31); }, 3, reps);
            e31  = timeMin([&] { gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 31, 31); }, 3, reps);
            printf("%-10s (1,1) %8.4f | (31,1) %8.4f (rows %+6.2fx) | (1,31) %8.4f (cols %+6.2fx) | (31,31) %8.4f\n",
                   c.n, e11, e31r, e31r / e11, e31c, e31c / e11, e31);
        }
        return 0;
    }

    if (mode == "floor") {
        // 各尺寸搬运地板（含启动固定成本）
        for (const C& c : kCs) {
            Img s; s.make(c.W, c.H);
            Img d16; { CKV(cudaMallocPitch(&d16.d, &d16.step, (size_t)c.W * 2, c.H)); }
            Img d8;   { CKV(cudaMallocPitch(&d8.d,  &d8.step,  (size_t)c.W * 2, c.H)); }
            FImg fs; fs.make(c.W, c.H);
            FImg fd; { CKV(cudaMallocPitch(&fd.d, &fd.step, (size_t)c.W * 4, c.H)); }

            const int th = 256;
            dim3 g16(idiv_up(c.W, 16 * th), c.H);
            dim3 g4(idiv_up(c.W, 4 * th), c.H);
            const double nop  = timeMin([&] { k_noop<<<1, 32>>>(); }, 5, reps);
            const double up   = timeMin([&] { k_u8_u16<<<g16, th>>>(s.d, s.step, (uint16_t*)d16.d, d16.step, c.W); }, 5, reps);
            const double dn   = timeMin([&] { k_u16_u8<<<g16, th>>>((const uint16_t*)d16.d, d16.step, d8.d, d8.step, c.W); }, 5, reps);
            const double f32c = timeMin([&] { k_f32cp<<<g4, th>>>(fs.d, fs.step, fd.d, fd.step, c.W); }, 5, reps);
            const double npx  = (double)c.W * c.H;
            printf("%-10s noop %7.4f | u8->u16 %7.4f (%5.1f GB/s) | u16->u8 %7.4f (%5.1f GB/s) | f32cp %7.4f (%5.1f GB/s)\n",
                   c.n, nop, up, 3.0 * npx / up / 1e6, dn, 3.0 * npx / dn / 1e6,
                   f32c, 8.0 * npx / f32c / 1e6);
            cudaFree(d16.d); cudaFree(d8.d); cudaFree(fd.d);
        }
        return 0;
    }

    if (mode == "sweep") {
        // 小图「固定成本」探针：同一算子连续 N 次调用的平均（摊薄启动）vs 单次
        // 用来判断 gaussian-f32 k=5 / mean k=3 小图究竟是启动成本还是内核本体
        for (const C& c : kCs) {
            Img img; img.make(c.W, c.H);
            FImg fs; fs.make(c.W, c.H);
            FImg fd; { CKV(cudaMallocPitch(&fd.d, &fd.step, (size_t)c.W * 4, c.H)); }
            const int N = 20;
            double one = timeMin([&] { gpu_gaussian_f32(fs.d, fs.step, fd.d, fd.step, c.W, c.H, 5, 0.f); }, 5, reps);
            double rep = timeMin([&] {
                for (int i = 0; i < N; ++i) gpu_gaussian_f32(fs.d, fs.step, fd.d, fd.step, c.W, c.H, 5, 0.f);
            }, 3, reps);
            double oneM = timeMin([&] { gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 3, 3); }, 5, reps);
            double repM = timeMin([&] {
                for (int i = 0; i < N; ++i) gpu_mean(img.d, img.step, img.d, img.step, c.W, c.H, 3, 3);
            }, 3, reps);
            printf("%-10s gaussF32k5 single %8.4f amortized %8.4f (launch %.4f) | mean3 single %8.4f amortized %8.4f (launch %.4f)\n",
                   c.n, one, rep / N, one - rep / N, oneM, repM / N, oneM - repM / N);
            cudaFree(fd.d);
        }
        return 0;
    }

    printf("usage: probe_meanbig [asym|floor|sweep] [reps]\n");
    return 0;
}