// probe_gf32k15d：k=15 融合内核 —— 依赖链拆分 + launch_bounds + 形状扫描（r63）
// c 轮结论：PF（软件预取）正确，4K 最优 pf192/th128 = 1.2278 vs 默认两遍 1.7183（1.40x），
// 但仍只 ~54 GB/s（8B/px 地板 93）。剩下的怀疑是「垂直 15 级 fmaf 串行链」的延迟：
// 每线程每像素 15 次依赖 FMA，即使有预取也压不住。
// 本探针：
//   · SG=false：垂直和按 c_w[0..14] 顺序链（与两遍路径逐位一致）
//   · SG=true ：拆 3 组各自累加后相加（依赖链 5 级），会改动浮点加法次序 → 记录 maxdiff
//   · LB：launch_bounds 档（256,2 / 128,3 / 128,4）压寄存器提占用率
// 全部走 PF 结构（预取）。
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

template <int K>
__device__ __forceinline__ float4 g15_load(const float* __restrict__ src, int rows, int xa, int ry)
{
    return *(const float4*)(src + (size_t)ry * rows + xa);
}

template <int K>
__device__ __forceinline__ void g15_hrow(const float* __restrict__ src, int rows, int W,
                                         int xa, int lane, int ry, float4 v, float o[4])
{
    constexpr int HALF = K / 2;
    const float* srow = src + (size_t)ry * rows;
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
}

// 垂直和：SG=false 顺序链；SG=true 3 组并行后相加（依赖链 15 → 5+2）
template <int K, bool SG>
__device__ __forceinline__ void g15_vsum(const float h[K][4], float vv[4])
{
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        if (!SG) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < K; ++q) s = fmaf(c_w15[q], h[q][j], s);
            vv[j] = s;
        } else {
            float s0 = 0.f, s1 = 0.f, s2 = 0.f;
#pragma unroll
            for (int q = 0; q < 5; ++q) {
                s0 = fmaf(c_w15[q],     h[q][j],     s0);
                s1 = fmaf(c_w15[q + 5], h[q + 5][j], s1);
                s2 = fmaf(c_w15[q + 10], h[q + 10][j], s2);
            }
            vv[j] = (s0 + s1) + s2;
        }
    }
}

template <int K, int TH, bool SG, int LB_MIN, int LBPB>
__global__ void __launch_bounds__(LB_MIN, LBPB)
g32f_d_kernel(const float* __restrict__ src, size_t sstep,
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
#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int ry = mfold(y0 - HALF + i, H);
        const float4 v = g15_load<K>(src, rows, xa, ry);
        g15_hrow<K>(src, rows, W, xa, lane, ry, v, h[i]);
    }

    int rcur = mfold(y0 + HALF + 1, H);
    float4 vcur = g15_load<K>(src, rows, xa, rcur);
    for (int y = y0; y < y1; ++y) {
        const int rnext = mfold(y + HALF + 2, H);
        const float4 vnext = g15_load<K>(src, rows, xa, rnext);
        float hn[4];
        g15_hrow<K>(src, rows, W, xa, lane, rcur, vcur, hn);
        if (act) {
            float vv[4];
            g15_vsum<K, SG>(h, vv);
            *(float4*)(dst + (size_t)y * orows + x0) = make_float4(vv[0], vv[1], vv[2], vv[3]);
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) h[i][j] = h[i + 1][j];
#pragma unroll
        for (int j = 0; j < 4; ++j) h[K - 1][j] = hn[j];
        vcur = vnext; rcur = rnext;
    }
}

// 启动器（TH × LB 档）
template <int TH, bool SG, int LBM, int LBPB>
static void launchD(const float* a, size_t ast, float* b, size_t bst, int W, int H, int th)
{
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    g32f_d_kernel<15, TH, SG, LBM, LBPB><<<g, th>>>(a, ast, b, bst, W, H);
}

struct Var { int TH; int th; int sg; int lbm; int lbpb; const char* name; };

static int runVar(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H, const Var& v)
{
    int th = v.th; if (th > (W + 3) / 4) th = std::max(32, ((W + 3) / 4 + 31) / 32 * 32);
    if (th < 32) th = 32;
    if (v.lbm == 256 && v.lbpb == 2) {
        switch (v.TH) {
        case 64:  v.sg ? launchD<64, true, 256, 2>(dA, astep, dB, bstep, W, H, th)
                       : launchD<64, false, 256, 2>(dA, astep, dB, bstep, W, H, th); break;
        case 128: v.sg ? launchD<128, true, 256, 2>(dA, astep, dB, bstep, W, H, th)
                       : launchD<128, false, 256, 2>(dA, astep, dB, bstep, W, H, th); break;
        case 192: v.sg ? launchD<192, true, 256, 2>(dA, astep, dB, bstep, W, H, th)
                       : launchD<192, false, 256, 2>(dA, astep, dB, bstep, W, H, th); break;
        case 256: v.sg ? launchD<256, true, 256, 2>(dA, astep, dB, bstep, W, H, th)
                       : launchD<256, false, 256, 2>(dA, astep, dB, bstep, W, H, th); break;
        default: return -1;
        }
    } else if (v.lbm == 128 && v.lbpb == 3) {
        switch (v.TH) {
        case 64:  v.sg ? launchD<64, true, 128, 3>(dA, astep, dB, bstep, W, H, th)
                       : launchD<64, false, 128, 3>(dA, astep, dB, bstep, W, H, th); break;
        case 128: v.sg ? launchD<128, true, 128, 3>(dA, astep, dB, bstep, W, H, th)
                       : launchD<128, false, 128, 3>(dA, astep, dB, bstep, W, H, th); break;
        case 192: v.sg ? launchD<192, true, 128, 3>(dA, astep, dB, bstep, W, H, th)
                       : launchD<192, false, 128, 3>(dA, astep, dB, bstep, W, H, th); break;
        case 256: v.sg ? launchD<256, true, 128, 3>(dA, astep, dB, bstep, W, H, th)
                       : launchD<256, false, 128, 3>(dA, astep, dB, bstep, W, H, th); break;
        default: return -1;
        }
    } else {   // 128,4
        switch (v.TH) {
        case 64:  v.sg ? launchD<64, true, 128, 4>(dA, astep, dB, bstep, W, H, th)
                       : launchD<64, false, 128, 4>(dA, astep, dB, bstep, W, H, th); break;
        case 128: v.sg ? launchD<128, true, 128, 4>(dA, astep, dB, bstep, W, H, th)
                       : launchD<128, false, 128, 4>(dA, astep, dB, bstep, W, H, th); break;
        case 192: v.sg ? launchD<192, true, 128, 4>(dA, astep, dB, bstep, W, H, th)
                       : launchD<192, false, 128, 4>(dA, astep, dB, bstep, W, H, th); break;
        case 256: v.sg ? launchD<256, true, 128, 4>(dA, astep, dB, bstep, W, H, th)
                       : launchD<256, false, 128, 4>(dA, astep, dB, bstep, W, H, th); break;
        default: return -1;
        }
    }
    return 0;
}

static double timeMs(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                     int reps, const Var* v)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        if (!v) gpu_gaussian_f32(dA, astep, dB, bstep, W, H, 15, 0.f);
        else    runVar(dA, astep, dB, bstep, W, H, *v);
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
    // 检查：全部 8 个代表形状 × {SG=false 逐位, SG=true 记录 maxdiff}
    struct Case { int W, H; };
    const Case chk[] = {{16, 3}, {36, 4}, {100, 100}, {320, 240}, {720, 479}, {1280, 720}, {4096, 64}};
    const Var chkSG[] = {{128, 256, 1, 128, 3, "sg"}, {192, 128, 0, 256, 2, "chain"}};
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
        for (const Var& v : chkSG) {
            if (runVar(dA, astep, dB, bstep, W, H, v)) continue;
            CKV(cudaDeviceSynchronize());
            std::vector<float> out(N);
            CKV(cudaMemcpy2D(out.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            int bad = 0; float md = 0;
            for (size_t i = 0; i < N; ++i)
                if (out[i] != ref[i]) { ++bad; md = std::max(md, (float)fabs((double)out[i] - ref[i])); }
            printf("  chk %dx%d %s bad=%d maxdiff=%g\n", W, H, v.name, bad, md);
            if (v.sg == 0 && bad) ++badTotal;
        }
        cudaFree(dA); cudaFree(dB); cudaFree(dR);
    }

    const Var pv[] = {
        {128, 128, 0, 128, 3, "c128"},
        {128, 128, 1, 128, 3, "s128"},
        {192, 128, 0, 256, 2, "c192"},
        {192, 128, 1, 256, 2, "s192"},
        {192, 128, 0, 128, 4, "c192q4"},
        {192, 128, 1, 128, 4, "s192q4"},
        {256, 128, 0, 128, 3, "c256"},
        {256, 128, 1, 128, 3, "s256"},
        {192, 64,  1, 128, 3, "s192t64"},
    };
    const int nV = sizeof(pv) / sizeof(pv[0]);
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
        const int reps = (N > 4000000) ? 25 : 60;
        double bV0 = 1e9;
        double best[16]; for (int i = 0; i < nV; ++i) best[i] = 1e9;
        for (int r = 0; r < 4; ++r) {
            const double m0 = timeMs(dA, astep, dB, bstep, W, H, reps, nullptr);
            if (m0 < bV0) bV0 = m0;
            for (int t = 0; t < nV; ++t) {
                const int idx = (t + r) % nV;
                const double m = timeMs(dA, astep, dB, bstep, W, H, reps, &pv[idx]);
                if (m < best[idx]) best[idx] = m;
            }
        }
        printf("%4dx%-4d V0 %.4f", W, H, bV0);
        for (int i = 0; i < nV; ++i) printf("  %s %.4f", pv[i].name, best[i]);
        printf("\n"); fflush(stdout);
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    printf(badTotal ? "PROBE_GF32K15D_CHECK_FAIL bad=%d\n" : "PROBE_GF32K15D_OK\n", badTotal);
    return 0;
}