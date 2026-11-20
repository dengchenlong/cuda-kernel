// probe_gf32k15b：gaussian-f32 k=15 单遍融合 —— 占用率攻坚（r63）
// 上一版 probe_gf32k15 结论：正确，但 128 regs → 2 块/SM，4K 1.335ms = 49.7 GB/s（天花板 93）。
// 本探针对比两个族：
//   PX4：每线程 4 px（float4），h[15][4] = 60 float 寄存器
//   PX2：每线程 2 px（float2），h[15][2] = 30 float 寄存器 → 期望 3~4 块/SM
// 两族都扫 TH ∈ {8,16,32,48,64,96,128,144,180,240}（4K/1080p 的 grid.x=4/2，甜点应在
// 「总块数 ≈ 12 常驻块的整数倍」附近：4K → nChunks∈{12,15,9}，1080p → nChunks∈{6,12,18}）。
// 正确性：两族 × 全部 TH 与 gpu_gaussian_f32 默认两遍路径逐位比较。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include "ops.h"
using uint = unsigned int;

__constant__ float c_w15[15];

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); badTotal++; } } while(0)

__host__ __device__ __forceinline__ int mfold(int v, int len)
{
    int r = v;
    if (r < 0) r = -r;
    if (r >= len) r = 2 * len - 2 - r;
    return max(0, min(len - 1, r));
}

// ---------------- PX4：每线程 4 px ----------------
template <int K, int TH>
__global__ void g32f_px4_kernel(const float* __restrict__ src, size_t sstep,
                                float* __restrict__ dst, size_t dstep, int W, int H)
{
    constexpr int HALF = K / 2;
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 4;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 4) & ~3);
    const int rows = (int)(sstep >> 2), orows = (int)(dstep >> 2);

    float h[K][4];

    auto hrow = [&](int ry, float o[4]) {
        const float* srow = src + (size_t)ry * rows;
        const float4 v = *(const float4*)(srow + xa);
        float L[HALF], Rt[HALF];
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

// ---------------- PX2：每线程 2 px（float2），寄存器减半 ----------------
template <int K, int TH>
__global__ void g32f_px2_kernel(const float* __restrict__ src, size_t sstep,
                                float* __restrict__ dst, size_t dstep, int W, int H)
{
    constexpr int HALF = K / 2;          // 7 → delta 最大 4
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 2;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 2) & ~1);
    const int rows = (int)(sstep >> 2), orows = (int)(dstep >> 2);

    float h[K][2];

    auto hrow = [&](int ry, float o[2]) {
        const float* srow = src + (size_t)ry * rows;
        const float2 v = *(const float2*)(srow + xa);
        float L[HALF], Rt[HALF];
        // 左 j → px[xa-j]：j 奇 = 前 j/2+1 线程的 y，j 偶 = 前 j/2 线程的 x
#pragma unroll
        for (int j = 1; j <= HALF; ++j) {
            const int d = (j + 1) / 2;
            if (j & 1) L[j - 1] = __shfl_up_sync(0xffffffffu, v.y, d);
            else       L[j - 1] = __shfl_up_sync(0xffffffffu, v.x, d);
            if (j & 1) Rt[j - 1] = __shfl_down_sync(0xffffffffu, v.x, d);
            else       Rt[j - 1] = __shfl_down_sync(0xffffffffu, v.y, d);
        }
        if (lane < 4 || xa < HALF) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) L[j - 1] = srow[mfold(xa - j, W)];
        }
        if (lane > 27 || xa + 2 + HALF > W) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) Rt[j - 1] = srow[mfold(xa + 1 + j, W)];
        }
        float pb[2 + 2 * HALF];
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[j] = L[HALF - 1 - j];
        pb[HALF + 0] = v.x; pb[HALF + 1] = v.y;
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[HALF + 2 + j] = Rt[j];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < K; ++q) s = fmaf(c_w15[q], pb[i + q], s);
            o[i] = s;
        }
    };

#pragma unroll
    for (int i = 0; i < K; ++i) hrow(mfold(y0 - HALF + i, H), h[i]);

    for (int y = y0; y < y1; ++y) {
        float hn[2];
        hrow(mfold(y + HALF + 1, H), hn);
        if (act) {
            float vv[2];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < K; ++q) s = fmaf(c_w15[q], h[q][j], s);
                vv[j] = s;
            }
            *(float2*)(dst + (size_t)y * orows + x0) = make_float2(vv[0], vv[1]);
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i)
#pragma unroll
            for (int j = 0; j < 2; ++j) h[i][j] = h[i + 1][j];
#pragma unroll
        for (int j = 0; j < 2; ++j) h[K - 1][j] = hn[j];
    }
}

static const int THs[] = {8, 16, 32, 48, 64, 96, 128, 144, 180, 240};
static const int nTH = 10;

template <int TH>
static void launchPx4(const float* a, size_t ast, float* b, size_t bst, int W, int H)
{
    const int th = std::min(256, std::max(32, (((W + 3) / 4) + 31) / 32 * 32));
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    g32f_px4_kernel<15, TH><<<g, th>>>(a, ast, b, bst, W, H);
}
template <int TH>
static void launchPx2(const float* a, size_t ast, float* b, size_t bst, int W, int H)
{
    const int th = std::min(256, std::max(32, (((W + 1) / 2) + 31) / 32 * 32));
    dim3 g((unsigned)((W + 2 * th - 1) / (2 * th)), (unsigned)((H + TH - 1) / TH));
    g32f_px2_kernel<15, TH><<<g, th>>>(a, ast, b, bst, W, H);
}

// fam: 0 = PX4, 1 = PX2
static int runVariant(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H, int TH, int fam)
{
    if (W < 8) return -1;
    switch (TH) {
    case 8:   fam ? launchPx2<8>(dA, astep, dB, bstep, W, H)   : launchPx4<8>(dA, astep, dB, bstep, W, H); break;
    case 16:  fam ? launchPx2<16>(dA, astep, dB, bstep, W, H)  : launchPx4<16>(dA, astep, dB, bstep, W, H); break;
    case 32:  fam ? launchPx2<32>(dA, astep, dB, bstep, W, H)  : launchPx4<32>(dA, astep, dB, bstep, W, H); break;
    case 48:  fam ? launchPx2<48>(dA, astep, dB, bstep, W, H)  : launchPx4<48>(dA, astep, dB, bstep, W, H); break;
    case 64:  fam ? launchPx2<64>(dA, astep, dB, bstep, W, H)  : launchPx4<64>(dA, astep, dB, bstep, W, H); break;
    case 96:  fam ? launchPx2<96>(dA, astep, dB, bstep, W, H)  : launchPx4<96>(dA, astep, dB, bstep, W, H); break;
    case 128: fam ? launchPx2<128>(dA, astep, dB, bstep, W, H) : launchPx4<128>(dA, astep, dB, bstep, W, H); break;
    case 144: fam ? launchPx2<144>(dA, astep, dB, bstep, W, H) : launchPx4<144>(dA, astep, dB, bstep, W, H); break;
    case 180: fam ? launchPx2<180>(dA, astep, dB, bstep, W, H) : launchPx4<180>(dA, astep, dB, bstep, W, H); break;
    case 240: fam ? launchPx2<240>(dA, astep, dB, bstep, W, H) : launchPx4<240>(dA, astep, dB, bstep, W, H); break;
    default:  return -1;
    }
    return 0;
}

static double timeMs(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                     int reps, int varied, int fam)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        if (varied < 0) gpu_gaussian_f32(dA, astep, dB, bstep, W, H, 15, 0.f);
        else            runVariant(dA, astep, dB, bstep, W, H, varied, fam);
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
    const Case chk[] = {{8, 8}, {8, 9}, {12, 5}, {16, 1}, {16, 3}, {18, 4}, {36, 4}, {64, 3}, {100, 100},
                        {320, 240}, {720, 479}, {512, 512}, {1024, 768}, {4096, 64}, {3840, 64}};
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
        gpu_gaussian_f32(dA, astep, dR, rstep, W, H, 15, 0.f);
        CKV(cudaDeviceSynchronize());
        std::vector<float> ref(N);
        CKV(cudaMemcpy2D(ref.data(), (size_t)W * 4, dR, rstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        for (int fam = 0; fam < 2; ++fam)
            for (int t = 0; t < nTH; ++t) {
                if (runVariant(dA, astep, dB, bstep, W, H, THs[t], fam)) continue;
                CKV(cudaDeviceSynchronize());
                std::vector<float> out(N);
                CKV(cudaMemcpy2D(out.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
                int bad = 0; float md = 0;
                for (size_t i = 0; i < N; ++i)
                    if (out[i] != ref[i]) { ++bad; md = std::max(md, (float)fabs((double)out[i] - ref[i])); }
                if (bad) { printf("  fam%d TH%d W=%d H=%d bad=%d maxdiff=%g\n", fam, THs[t], W, H, bad, md); ++badTotal; }
            }
        printf("  chk %dx%d ok\n", W, H); fflush(stdout);
        cudaFree(dA); cudaFree(dB); cudaFree(dR);
    }

    const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
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
        for (int fam = 0; fam < 2; ++fam) {
            double bestV0 = 1e9;
            double best[nTH]; for (int i = 0; i < nTH; ++i) best[i] = 1e9;
            for (int r = 0; r < 5; ++r) {
                const double ms0 = timeMs(dA, astep, dB, bstep, W, H, reps, -1, fam);
                if (ms0 < bestV0) bestV0 = ms0;
                for (int t = 0; t < nTH; ++t) {
                    const int idx = (t + r) % nTH;
                    const double ms = timeMs(dA, astep, dB, bstep, W, H, reps, THs[idx], fam);
                    if (ms < best[idx]) best[idx] = ms;
                }
            }
            printf("%4dx%-4d %s V0 %.4f", W, H, fam ? "PX2" : "PX4", bestV0);
            for (int i = 0; i < nTH; ++i) printf("  T%d %.4f", THs[i], best[i]);
            printf("\n"); fflush(stdout);
        }
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    printf(badTotal ? "PROBE_GF32K15B_CHECK_FAIL bad=%d\n" : "PROBE_GF32K15B_OK\n", badTotal);
    return 0;
}