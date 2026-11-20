// probe_gf32k15c：gaussian-f32 k=15 融合内核 —— 加跨迭代预取（软件流水）打占用率/延迟
// 上一轮（probe_gf32k15b）结论：
//   · PX2（2px/线程）全线更慢，弃；PX4（4px/线程）+ TH=128 是唯一赢的形状
//     （4K 1.338 vs 默认两遍 1.717 = 1.28x；但只 49.6 GB/s，天花板 93）。
//   · 归因：每迭代只有 1 个 float4 load，寄存器 128 → 2 块/SM（16 warps），
//     in-flight 8KB，远小于「带宽×延迟 ≈ 46KB」。故加软件流水：先发下一行 load，再算当前行。
//   · W=18 的 bad 是非 4 倍数宽度（融合路径前提 W%4==0，正式路径不会走），本轮不测。
// 变体：V0 = 默认两遍路径（参照）；PF = 预取融合；NP = 无预取融合（同 b 轮 PX4）。
// 扫描：TH ∈ {64,96,128,192,240} × th ∈ {128,256}（th 影响 grid.x 与块数）。
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

// hrow 拆成 load / compute 两段，供软件流水使用
template <int K>
__device__ __forceinline__ float4 g15_load(const float* __restrict__ src, int rows, int xa, int ry)
{
    return *(const float4*)(src + (size_t)ry * rows + xa);
}

// 水平 15-tap（v 已 load 好；ry 用于边界 warp 的 global 镜像读）
template <int K>
__device__ __forceinline__ void g15_hrow2(const float* __restrict__ src, int rows, int W,
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

// PF=true：跨迭代预取（先发下一行 load）；PF=false：与 b 轮相同的直读版本
template <int K, int TH, bool PF>
__global__ void g32f_pf_kernel(const float* __restrict__ src, size_t sstep,
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
        g15_hrow2<K>(src, rows, W, xa, lane, ry, v, h[i]);
    }

    if (PF) {
        int rcur = mfold(y0 + HALF + 1, H);             // 首迭代要算的行（h 里已有到 y0+HALF）
        float4 vcur = g15_load<K>(src, rows, xa, rcur);
        for (int y = y0; y < y1; ++y) {
            const int rnext = mfold(y + HALF + 2, H);   // 下一迭代 y+1 要算的行
            const float4 vnext = g15_load<K>(src, rows, xa, rnext);   // 提前发
            float hn[4];
            g15_hrow2<K>(src, rows, W, xa, lane, rcur, vcur, hn);
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
            vcur = vnext; rcur = rnext;
        }
    } else {
        // 无预取版：每次迭代内先算 hrow（load 在 hrow 里）再垂直求和
        for (int y = y0; y < y1; ++y) {
            const int ry = mfold(y + HALF + 1, H);
            float hn[4];
            const float4 v = g15_load<K>(src, rows, xa, ry);
            g15_hrow2<K>(src, rows, W, xa, lane, ry, v, hn);
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
}

// 通用启动器：TH 模板 × th 运行时（th 只影响 grid/block 配置）
template <int TH, bool PF>
static void launchTH(const float* a, size_t ast, float* b, size_t bst, int W, int H, int th)
{
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    g32f_pf_kernel<15, TH, PF><<<g, th>>>(a, ast, b, bst, W, H);
}

static int runVariant(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                      int TH, int th, bool pf)
{
    if (pf) {
        switch (TH) {
        case 32:  launchTH<32, true>(dA, astep, dB, bstep, W, H, th); break;
        case 64:  launchTH<64, true>(dA, astep, dB, bstep, W, H, th); break;
        case 96:  launchTH<96, true>(dA, astep, dB, bstep, W, H, th); break;
        case 128: launchTH<128, true>(dA, astep, dB, bstep, W, H, th); break;
        case 192: launchTH<192, true>(dA, astep, dB, bstep, W, H, th); break;
        case 240: launchTH<240, true>(dA, astep, dB, bstep, W, H, th); break;
        default: return -1;
        }
    } else {
        switch (TH) {
        case 32:  launchTH<32, false>(dA, astep, dB, bstep, W, H, th); break;
        case 64:  launchTH<64, false>(dA, astep, dB, bstep, W, H, th); break;
        case 96:  launchTH<96, false>(dA, astep, dB, bstep, W, H, th); break;
        case 128: launchTH<128, false>(dA, astep, dB, bstep, W, H, th); break;
        case 192: launchTH<192, false>(dA, astep, dB, bstep, W, H, th); break;
        case 240: launchTH<240, false>(dA, astep, dB, bstep, W, H, th); break;
        default: return -1;
        }
    }
    return 0;
}

static double timeMs(const float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                     int reps, int TH, int th, int pf)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        if (TH < 0) gpu_gaussian_f32(dA, astep, dB, bstep, W, H, 15, 0.f);
        else        runVariant(dA, astep, dB, bstep, W, H, TH, th, pf != 0);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

struct Var { int TH; int th; int pf; const char* name; };

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
    // 正确性：只测 W%4==0 的形状（融合路径前提），含 4 个 bench 尺寸 + 极扁/小图
    struct Case { int W, H; };
    const Case chk[] = {{8, 8}, {8, 9}, {12, 5}, {16, 1}, {16, 3}, {36, 4}, {64, 3}, {100, 100},
                        {320, 240}, {720, 479}, {512, 512}, {1024, 768}, {640, 480}, {1280, 720},
                        {1920, 1080}, {4096, 64}, {3840, 64}};
    const Var cvars[] = {{32, 32, 1, "pf"}, {128, 128, 1, "pf"}, {128, 256, 1, "pf"},
                         {64, 128, 0, "np"}, {128, 128, 0, "np"}};
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
        for (const Var& v : cvars) {
            int th = v.th; if (th > (W + 3) / 4) th = std::max(32, ((W + 3) / 4 + 31) / 32 * 32);
            if (th < 32) th = 32;
            if (runVariant(dA, astep, dB, bstep, W, H, v.TH, th, v.pf != 0)) continue;
            CKV(cudaDeviceSynchronize());
            std::vector<float> out(N);
            CKV(cudaMemcpy2D(out.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            int bad = 0; float md = 0;
            for (size_t i = 0; i < N; ++i)
                if (out[i] != ref[i]) { ++bad; md = std::max(md, (float)fabs((double)out[i] - ref[i])); }
            if (bad) { printf("  %s TH%d th%d W=%d H=%d bad=%d maxdiff=%g\n", v.name, v.TH, v.th, W, H, bad, md); ++badTotal; }
        }
        printf("  chk %dx%d ok\n", W, H); fflush(stdout);
        cudaFree(dA); cudaFree(dB); cudaFree(dR);
    }
    if (badTotal) printf("CHECK_FAIL bad=%d\n", badTotal);

    // 性能：V0（默认两遍）+ PF/NP × TH × th
    const Var pvars[] = {
        { 64, 256, 1, "pf"}, { 96, 256, 1, "pf"}, {128, 256, 1, "pf"}, {192, 256, 1, "pf"}, {240, 256, 1, "pf"},
        {128, 128, 1, "pf"}, {192, 128, 1, "pf"},
        {128, 256, 0, "np"}, {192, 256, 0, "np"},
    };
    const int nV = sizeof(pvars) / sizeof(pvars[0]);
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
            const double m0 = timeMs(dA, astep, dB, bstep, W, H, reps, -1, 0, 0);
            if (m0 < bV0) bV0 = m0;
            for (int t = 0; t < nV; ++t) {
                const int idx = (t + r) % nV;
                int th = pvars[idx].th; if (th > (W + 3) / 4) th = std::max(32, ((W + 3) / 4 + 31) / 32 * 32);
                if (th < 32) th = 32;
                const double m = timeMs(dA, astep, dB, bstep, W, H, reps, pvars[idx].TH, th, pvars[idx].pf);
                if (m < best[idx]) best[idx] = m;
            }
        }
        printf("%4dx%-4d V0 %.4f", W, H, bV0);
        for (int i = 0; i < nV; ++i) printf("  %s%d/%d %.4f", pvars[i].name, pvars[i].TH, pvars[i].th, best[i]);
        printf("\n"); fflush(stdout);
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    printf(badTotal ? "PROBE_GF32K15C_CHECK_FAIL bad=%d\n" : "PROBE_GF32K15C_OK\n", badTotal);
    return 0;
}