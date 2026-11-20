// probe_gf32k15：gaussian-f32 k=15 单遍融合原型（r63 候选①的实质）
// 背景：k=5 走 r45 的 gauss32_fuse_kernel 已到 91 GB/s（拷贝天花板，4K 0.73ms）；
//       k=15 仍走 v4 两遍（16B/px + 每像素 30 次 smem 读），4K 1.86ms = 71 GB/s。
// 本探针把 hrow 的 halo 从 |2| 推广到 |HALF|（用 shfl_up/down delta 1/2 + 边界 global 镜像），
// 目标是 8B/px + 全寄存器垂直滚动。参照/计时对象：gpu_gaussian_f32 默认路径（两遍 v4）。
// 约束与 k=5 融合相同：out-of-place、W%4==0、8<=W<=4096、16B 对齐。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include "ops.h"
using uint = unsigned int;

__constant__ float c_w15[15];

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); badTotal++; } } while(0)

// 与 src/gaussian.cu g32f_mirror 同形：单次折叠 + clamp（HALF<=7、W>=8 时与 REFLECT_101 等价）
__host__ __device__ __forceinline__ int mfold(int v, int len)
{
    int r = v;
    if (r < 0) r = -r;
    if (r >= len) r = 2 * len - 2 - r;
    return max(0, min(len - 1, r));
}

template <int K, int TH>
__global__ void g32f_fuse_kernel(const float* __restrict__ src, size_t sstep,
                                 float* __restrict__ dst, size_t dstep, int W, int H)
{
    constexpr int HALF = K / 2;
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 4;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 4) & ~3);   // 非活跃线程给安全地址，仍参与 shuffle
    const int rows = (int)(sstep >> 2), orows = (int)(dstep >> 2);

    float h[K][4];

    auto hrow = [&](int ry, float o[4]) {
        const float* srow = src + (size_t)ry * rows;
        const float4 v = *(const float4*)(srow + xa);
        float L[HALF], Rt[HALF];
        // 水平 halo：delta1 给 |1..4|，delta2 给 |5..7|（无条件执行，避免 shfl 进分支 = UB）
#pragma unroll
        for (int j = 1; j <= HALF; ++j) {
            const int d = (j - 1) / 4 + 1;
            const int c = (j - 1) % 4;
            float lv, rv;
            if (c == 0)      { lv = __shfl_up_sync(0xffffffffu, v.w, d);  rv = __shfl_down_sync(0xffffffffu, v.x, d); }
            else if (c == 1) { lv = __shfl_up_sync(0xffffffffu, v.z, d);  rv = __shfl_down_sync(0xffffffffu, v.y, d); }
            else if (c == 2) { lv = __shfl_up_sync(0xffffffffu, v.y, d);  rv = __shfl_down_sync(0xffffffffu, v.z, d); }
            else             { lv = __shfl_up_sync(0xffffffffu, v.x, d);  rv = __shfl_down_sync(0xffffffffu, v.w, d); }
            L[j - 1]  = lv;
            Rt[j - 1] = rv;
        }
        // 边界覆盖：warp 边界（lane<2 / lane>29）+ 图像左右边缘（镜像）
        if (lane < 2 || xa < HALF) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) L[j - 1] = srow[mfold(xa - j, W)];
        }
        if (lane > 29 || xa + 4 + HALF > W) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) Rt[j - 1] = srow[mfold(xa + 3 + j, W)];
        }
        float pb[4 + 2 * HALF];
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[j] = L[HALF - 1 - j];
        pb[HALF + 0] = v.x; pb[HALF + 1] = v.y; pb[HALF + 2] = v.z; pb[HALF + 3] = v.w;
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[HALF + 4 + j] = Rt[j];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < K; ++q) s = fmaf(c_w15[q], pb[i + q], s);
            o[i] = s;
        }
    };

#pragma unroll
    for (int i = 0; i < K; ++i) hrow(mfold(y0 - HALF + i, H), h[i]);

    for (int y = y0; y < y1; ++y) {
        float hn[4];
        hrow(mfold(y + HALF + 1, H), hn);
        if (act) {
            float vv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < K; ++q) s = fmaf(c_w15[q], h[q][j], s);
                vv[j] = s;
            }
            *(float4*)(dst + (size_t)y * orows + x0) = make_float4(vv[0], vv[1], vv[2], vv[3]);
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) h[i][j] = h[i + 1][j];
#pragma unroll
        for (int j = 0; j < 4; ++j) h[K - 1][j] = hn[j];
    }
}

static int runVariant(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H, int TH)
{
    const int th = std::min(256, std::max(32, (((W + 3) / 4) + 31) / 32 * 32));
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    switch (TH) {
    case 8:   g32f_fuse_kernel<15, 8>  <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 16:  g32f_fuse_kernel<15, 16> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 32:  g32f_fuse_kernel<15, 32> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 48:  g32f_fuse_kernel<15, 48> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 64:  g32f_fuse_kernel<15, 64> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 96:  g32f_fuse_kernel<15, 96> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    case 128: g32f_fuse_kernel<15, 128><<<g, th>>>(dA, astep, dB, bstep, W, H); break;
    default:  return -1;
    }
    return 0;
}

static double timeMs(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                     int reps, int varied)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        if (varied < 0) {                 // 参照：默认两遍路径
            gpu_gaussian_f32(dA, astep, dB, bstep, W, H, 15, 0.f);
        } else {
            runVariant(dA, astep, dB, bstep, W, H, varied);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

int main()
{
    CK(cudaFree(0));
    {
        // 与 make_gauss_kernel(15, 0) 相同：sigmaX = 0.15*15+0.35
        const int ksize = 15;
        const double sigmaX = 0.15 * ksize + 0.35;
        const double scale2X = -0.125 / (sigmaX * sigmaX);
        const int n2 = (ksize - 1) / 2;
        std::vector<double> halfV(n2 + 1);
        halfV[n2] = 1.0;
        double sumHalf = 0.0;
        for (int i = 0, x = 1 - ksize; i < n2; i += 1, x += 2) {
            const double t = std::exp((double)x * x * scale2X);
            halfV[i] = t; sumHalf += t;
        }
        const double sum = 2.0 * sumHalf + 1.0;
        std::vector<float> w(ksize);
        for (int i = 0; i <= n2; ++i) { w[i] = (float)(halfV[i] / sum); w[ksize - 1 - i] = w[i]; }
        CK(cudaMemcpyToSymbol(c_w15, w.data(), sizeof(float) * ksize));
    }

    int badTotal = 0;
    struct Case { int W, H; };
    const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int THs[] = {8, 16, 32, 48, 64, 96, 128};
    const int nTH = 7;

    // ---- 正确性：非 bench 尺寸（含极扁/小图/W=8）----
    const Case chk[] = {{8, 8}, {8, 9}, {12, 5}, {16, 1}, {16, 3}, {36, 4}, {64, 3}, {100, 100},
                        {320, 240}, {1279 + 1, 720}, {720, 479}, {512, 512}, {1024, 768}, {1920, 1080},
                        {4096, 64}, {3840, 64}};
    for (const Case& c : chk) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        std::vector<float> img(N);
        unsigned st = 12345u + (unsigned)W;
        for (size_t i = 0; i < N; ++i) { st = st * 1664525u + 1013904223u; img[i] = (float)((int)((st >> 16) & 255)) * 0.5f - 40.f; }
        float *dA = nullptr, *dB = nullptr, *dR = nullptr;
        size_t astep = 0, bstep = 0, rstep = 0;
        CKV(cudaMallocPitch(&dA, &astep, (size_t)W * 4, H));
        CKV(cudaMallocPitch(&dB, &bstep, (size_t)W * 4, H));
        CKV(cudaMallocPitch(&dR, &rstep, (size_t)W * 4, H));
        CKV(cudaMemcpy2D(dA, astep, img.data(), (size_t)W * 4, (size_t)W * 4, H, cudaMemcpyHostToDevice));
        gpu_gaussian_f32(dA, astep, dR, rstep, W, H, 15, 0.f);   // 参照（两遍 v4）
        CKV(cudaDeviceSynchronize());
        std::vector<float> ref(N);
        CKV(cudaMemcpy2D(ref.data(), (size_t)W * 4, dR, rstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        for (int t = 0; t < nTH; ++t) {
            const int TH = THs[t];
            if (runVariant(dA, astep, dB, bstep, W, H, TH)) continue;
            CKV(cudaDeviceSynchronize());
            std::vector<float> out(N);
            CKV(cudaMemcpy2D(out.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            int bad = 0; float md = 0;
            for (size_t i = 0; i < N; ++i)
                if (out[i] != ref[i]) { ++bad; md = std::max(md, (float)fabs((double)out[i] - ref[i])); }
            if (bad) { printf("  TH%d W=%d H=%d bad=%d maxdiff=%g\n", TH, W, H, bad, md); ++badTotal; }
        }
        printf("  chk %dx%d ok\n", W, H); fflush(stdout);
        cudaFree(dA); cudaFree(dB); cudaFree(dR);
    }
    if (badTotal) { printf("PROBE_GF32K15_CHECK_FAIL bad=%d\n", badTotal); }

    // ---- 性能：6 轮轮转交错 min（V0 = 默认两遍）----
    for (const Case& c : perf) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        std::vector<float> img(N);
        unsigned st = 777u;
        for (size_t i = 0; i < N; ++i) { st = st * 1664525u + 1013904223u; img[i] = (float)((int)((st >> 16) & 255)) * 0.5f; }
        float *dA, *dB; size_t astep, bstep;
        CK(cudaMallocPitch(&dA, &astep, (size_t)W * 4, H));
        CK(cudaMallocPitch(&dB, &bstep, (size_t)W * 4, H));
        CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W * 4, (size_t)W * 4, H, cudaMemcpyHostToDevice));
        const int reps = (N > 4000000) ? 20 : 60;
        double bestV0 = 1e9;
        double best[nTH]; for (int i = 0; i < nTH; ++i) best[i] = 1e9;
        for (int r = 0; r < 6; ++r) {
            const double ms0 = timeMs(dA, astep, dB, bstep, W, H, reps, -1);
            if (ms0 < bestV0) bestV0 = ms0;
            for (int t = 0; t < nTH; ++t) {
                const int idx = (t + r) % nTH;
                const double ms = timeMs(dA, astep, dB, bstep, W, H, reps, THs[idx]);
                if (ms < best[idx]) best[idx] = ms;
            }
        }
        printf("%4dx%-4d V0 %.4f", W, H, bestV0);
        for (int i = 0; i < nTH; ++i) printf("  T%d %.4f", THs[i], best[i]);
        printf("\n"); fflush(stdout);
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    printf("PROBE_GF32K15_DONE\n");
    return 0;
}