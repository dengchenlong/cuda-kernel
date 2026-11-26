// probe_g15u8：u8 高斯 k=15 单遍融合（r67）
// 病灶：u8 k=15 一直走 v4 两遍 float tmp（读 1B + 写 4B + 读 4B + 写 1B = 10B/px），
// 4K 1.83ms 折算 45 GB/s（纯 uint4 拷贝天花板 93）。f32 侧 r63 已用单遍融合拿到 1.63x；
// u8 侧 out-of-place 单遍后只有 2B/px（读 1 + 写 1）。
// 结构照搬 gauss32_fuse15_kernel：4 px/线程（u8 侧一次读 uint 4B）、寄存器滚动 K 行
// 水平部分和、跨迭代软件预取（下一行 load 提前发）、15 级 fmaf 链。
// 算术次序与 v4 两遍路径完全同序（行 fmaf 从 0.f 起、列 fmaf 从 0.f 起）→ 逐位一致。
// 原地口径（benchmark 用 src==dst）：4px×256th 只覆盖 1024 列，4K 一行有 4 个 block，
// 跨 block 的水平 halo 会被邻块写掉（r48 的 k=7 同款坑），单块覆盖整行又要 960 线程
// （寄存器不允许）。所以原地走「融合写 workspace + 整图 uint4 拷回」= 4B/px（仍远低于 10）。
// 模式：check（形状 × TH 档 × {oop, ws}）/ sweep（四尺寸 × TH/th 档）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include "ops.h"
typedef unsigned int uint;

__constant__ float p_cw[31];

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); badTotal++; } } while(0)

__host__ __device__ __forceinline__ int mfold(int r, int H)
{
    int rr = r;
    if (rr < 0)  rr = -rr;
    if (rr >= H) rr = 2 * H - 2 - rr;
    return max(0, min(H - 1, rr));
}

template <int K, int TH, int LBM, int LBPB>
__global__ void __launch_bounds__(LBM, LBPB)
g15u8_kernel(const uint8_t* __restrict__ src, size_t sstep,
             uint8_t* __restrict__ dst, size_t dstep, int W, int H)
{
    constexpr int HALF = K / 2;
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 4;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 4) & ~3);
    const int rows = (int)sstep, orows = (int)dstep;

    float h[K][4];

    auto hrow = [&](int ry, uint w4, float o[4]) {
        const uint8_t* rp = src + (size_t)ry * rows;
        float v0 = (float)( w4       & 255u);
        float v1 = (float)((w4 >>  8) & 255u);
        float v2 = (float)((w4 >> 16) & 255u);
        float v3 = (float)((w4 >> 24) & 255u);
        float L[HALF], Rt[HALF];
#pragma unroll
        for (int j = 1; j <= HALF; ++j) {
            const int d = (j - 1) / 4 + 1;
            const int c = (j - 1) % 4;
            float lv, rv;
            if (c == 0)      { lv = __shfl_up_sync(0xffffffffu, v3, d); rv = __shfl_down_sync(0xffffffffu, v0, d); }
            else if (c == 1) { lv = __shfl_up_sync(0xffffffffu, v2, d); rv = __shfl_down_sync(0xffffffffu, v1, d); }
            else if (c == 2) { lv = __shfl_up_sync(0xffffffffu, v1, d); rv = __shfl_down_sync(0xffffffffu, v2, d); }
            else             { lv = __shfl_up_sync(0xffffffffu, v0, d); rv = __shfl_down_sync(0xffffffffu, v3, d); }
            L[j - 1]  = lv;
            Rt[j - 1] = rv;
        }
        if (lane < 2 || xa < HALF) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) L[j - 1] = (float)rp[mfold(xa - j, W)];
        }
        if (lane > 29 || xa + 4 + HALF > W) {
#pragma unroll
            for (int j = 1; j <= HALF; ++j) Rt[j - 1] = (float)rp[mfold(xa + 3 + j, W)];
        }
        float pb[4 + 2 * HALF];
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[j] = L[HALF - 1 - j];
        pb[HALF + 0] = v0; pb[HALF + 1] = v1; pb[HALF + 2] = v2; pb[HALF + 3] = v3;
#pragma unroll
        for (int j = 0; j < HALF; ++j) pb[HALF + 4 + j] = Rt[j];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < K; ++q) s = fmaf(p_cw[q], pb[i + q], s);
            o[i] = s;
        }
    };
    auto ld = [&](int ry) { return *(const uint*)(src + (size_t)ry * rows + xa); };

#pragma unroll
    for (int i = 0; i < K; ++i) {
        const int ry = mfold(y0 - HALF + i, H);
        hrow(ry, ld(ry), h[i]);
    }

    int rcur = mfold(y0 + HALF + 1, H);
    uint wcur = ld(rcur);
    for (int y = y0; y < y1; ++y) {
        const int rnext = mfold(y + HALF + 2, H);
        const uint wnext = ld(rnext);
        float hn[4];
        hrow(rcur, wcur, hn);
        if (act) {
            float vv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < K; ++q) s = fmaf(p_cw[q], h[q][j], s);
                vv[j] = s;
            }
            uint32_t o = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                int v = __float2int_rn(vv[j]);
                v = max(0, min(255, v));
                o |= (uint32_t)v << (8 * j);
            }
            *(uint*)(dst + (size_t)y * orows + x0) = o;
        }
#pragma unroll
        for (int i = 0; i < K - 1; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) h[i][j] = h[i + 1][j];
#pragma unroll
        for (int j = 0; j < 4; ++j) h[K - 1][j] = hn[j];
        wcur = wnext; rcur = rnext;
    }
}

// 整图 uint4 拷贝（原地口径第二段）
__global__ void copy16k(const uint8_t* __restrict__ s, size_t sstep,
                        uint8_t* __restrict__ d, size_t dstep, int W)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int y = blockIdx.y;
    const uint8_t* sp = s + (size_t)y * sstep;
    uint8_t* dp = d + (size_t)y * dstep;
    if (x + 16 <= W) *(uint4*)(dp + x) = *(const uint4*)(sp + x);
    else if (x < W) for (int i = x; i < W; ++i) dp[i] = sp[i];
}

static std::vector<float> mk_w(int ksize)
{
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
    return w;
}

static uint8_t* g_tmp = nullptr;
static size_t   g_tmpSz = 0;
static uint8_t* tmp_get(size_t need)
{
    if (g_tmpSz < need) {
        if (g_tmp) cudaFree(g_tmp);
        if (cudaMalloc(&g_tmp, need) != cudaSuccess) return nullptr;
        g_tmpSz = need;
    }
    return g_tmp;
}

template <int TH>
static void launchTh(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H, int th)
{
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    g15u8_kernel<15, TH, 256, 2><<<g, th>>>(s, ss, d, ds, W, H);
}

static int runFused(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H, int TH,
                    int th, bool wsCopy)
{
    const int need = (W + 3) / 4;
    int t = th;
    if (t > need) t = std::max(32, ((need + 31) / 32) * 32);
    if (t < 32) t = 32;
    uint8_t* out = d; size_t ostep = ds;
    if (wsCopy) {
        const size_t tstep = ((size_t)W + 15) & ~(size_t)15;
        out = tmp_get(tstep * H + 1024);
        if (!out) return -1;
        ostep = tstep;
    }
    switch (TH) {
    case 32:  launchTh<32>(s, ss, out, ostep, W, H, t); break;
    case 48:  launchTh<48>(s, ss, out, ostep, W, H, t); break;
    case 64:  launchTh<64>(s, ss, out, ostep, W, H, t); break;
    case 96:  launchTh<96>(s, ss, out, ostep, W, H, t); break;
    case 128: launchTh<128>(s, ss, out, ostep, W, H, t); break;
    case 192: launchTh<192>(s, ss, out, ostep, W, H, t); break;
    default: return -1;
    }
    if (wsCopy) {
        int fth = ((W + 15) / 16 + 31) / 32 * 32;
        if (fth > 256) fth = 256;
        if (fth < 32)  fth = 32;
        dim3 gc((unsigned)((W + 16 * fth - 1) / (16 * fth)), (unsigned)H);
        copy16k<<<gc, fth>>>(out, ostep, d, ds, W);
    }
    return 0;
}

struct Var { int TH; int th; bool ws; const char* name; };

// 计时：基准 gpu_gaussian（oop 或 ip）；变体按 v
static double timeMs(const uint8_t* dA, size_t astep, uint8_t* dB, size_t bstep, int W, int H,
                     int reps, const Var* v, bool ipRef)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        if (ipRef) {   // 原地参照：每轮先恢复原图（恢复不计入）
            cudaMemcpy2D(dB, bstep, dA, astep, W, H, cudaMemcpyDeviceToDevice);
        }
        cudaEventRecord(a);
        if (!v) {
            if (ipRef) gpu_gaussian(dB, bstep, dB, bstep, W, H, 15, 0.f);
            else       gpu_gaussian(dA, astep, dB, bstep, W, H, 15, 0.f);
        } else {
            runFused(dA, astep, dB, bstep, W, H, v->TH, v->th, v->ws);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

int main(int argc, char** argv)
{
    const char* mode = (argc > 1) ? argv[1] : "check";
    CK(cudaFree(0));
    {
        std::vector<float> w = mk_w(15);
        CK(cudaMemcpyToSymbol(p_cw, w.data(), sizeof(float) * 15));
    }
    int badTotal = 0;

    if (strcmp(mode, "check") == 0) {
        struct Case { int W, H; };
        const Case shapes[] = {
            {16, 3}, {36, 4}, {64, 1}, {100, 100}, {320, 240}, {720, 479},
            {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}, {4096, 64}, {16, 17}, {8, 1},
            {127, 129}, {33, 17}
        };
        for (const Case& c : shapes) {
            const int W = c.W, H = c.H;
            if (W % 4 != 0 || W < 8) {   // 融合路径约束：W%4==0 且 W>=8，其余回退 v4（正式入口保证）
                printf("  skip %5dx%-5d (W%%4!=0 or W<8) -> fallback v4\n", W, H);
                continue;
            }
            const size_t N = (size_t)W * H;
            std::vector<uint8_t> img(N);
            unsigned st = 12345u + (unsigned)W;
            for (size_t i = 0; i < N; ++i) { st = st * 1664525u + 1013904223u; img[i] = (uint8_t)((st >> 16) & 255); }
            uint8_t *dA = nullptr, *dR = nullptr, *dB = nullptr;
            size_t astep = 0, rstep = 0, bstep = 0;
            CKV(cudaMallocPitch(&dA, &astep, W, H));
            CKV(cudaMallocPitch(&dR, &rstep, W, H));
            CKV(cudaMallocPitch(&dB, &bstep, W, H));
            CKV(cudaMemcpy2D(dA, astep, img.data(), W, W, H, cudaMemcpyHostToDevice));
            // 参照 A：现有 v4 两遍（out-of-place）
            gpu_gaussian(dA, astep, dR, rstep, W, H, 15, 0.f);
            CKV(cudaDeviceSynchronize());
            std::vector<uint8_t> ref(N);
            CKV(cudaMemcpy2D(ref.data(), W, dR, rstep, W, H, cudaMemcpyDeviceToHost));
            // 参照 B：现有 v4 原地
            std::vector<uint8_t> refip(N);
            {
                uint8_t* dI = nullptr; size_t istep = 0;
                CKV(cudaMallocPitch(&dI, &istep, W, H));
                CKV(cudaMemcpy2D(dI, istep, img.data(), W, W, H, cudaMemcpyHostToDevice));
                gpu_gaussian(dI, istep, dI, istep, W, H, 15, 0.f);
                CKV(cudaDeviceSynchronize());
                CKV(cudaMemcpy2D(refip.data(), W, dI, istep, W, H, cudaMemcpyDeviceToHost));
                cudaFree(dI);
            }
            const int THs[] = {32, 64, 96, 128, 192};
            for (int ti = 0; ti < 5; ++ti) {
                for (int mode2 = 0; mode2 < 2; ++mode2) {
                    const bool ws = (mode2 == 1);
                    if (runFused(dA, astep, dB, bstep, W, H, THs[ti], 256, ws)) {
                        printf("  chk %dx%d TH%d launch unsupported\n", W, H, THs[ti]);
                        continue;
                    }
                    CKV(cudaDeviceSynchronize());
                    std::vector<uint8_t> out(N);
                    CKV(cudaMemcpy2D(out.data(), W, dB, bstep, W, H, cudaMemcpyDeviceToHost));
                    const std::vector<uint8_t>& R = ws ? refip : ref;
                    int bad = 0, md = 0, first = -1;
                    for (size_t i = 0; i < N; ++i)
                        if (out[i] != R[i]) { if (first < 0) first = (int)i; ++bad; md = std::max(md, abs((int)out[i] - (int)R[i])); }
                    printf("  chk %5dx%-5d TH%-3d %s bad=%d maxdiff=%d%s\n", W, H, THs[ti],
                           ws ? "ws" : "oop", bad, md,
                           bad ? (std::string(" first=") + std::to_string(first)).c_str() : "");
                    if (bad) ++badTotal;
                }
            }
            cudaFree(dA); cudaFree(dR); cudaFree(dB);
            fflush(stdout);
        }
        printf(badTotal ? "PROBE_G15U8_CHECK_FAIL badTotal=%d\n" : "PROBE_G15U8_CHECK_OK\n", badTotal);
        return badTotal ? 1 : 0;
    }

    // sweep：四尺寸；参照 V0（oop 两遍）、V0ip（原地两遍），变体交错轮转取 min
    const Var pv[] = {
        {32,  192, false, "T32t192"},
        {64,  128, false, "T64t128"},
        {96,  64,  false, "T96t64"},
        {96,  128, false, "T96t128"},
        {128, 64,  false, "T128t64"},
        {128, 128, false, "T128t128"},
        {192, 64,  false, "T192t64"},
        {192, 128, false, "T192t128"},
        {64,  64,  false, "T64t64"},
        {64,  128, true,  "W64t128"},
        {64,  64,  true,  "W64t64"},
        {96,  64,  true,  "W96t64"},
        {96,  128, true,  "W96t128"},
        {128, 64,  true,  "W128t64"},
        {128, 128, true,  "W128t128"},
        {192, 64,  true,  "W192t64"},
        {192, 128, true,  "W192t128"},
    };
    const int nV = sizeof(pv) / sizeof(pv[0]);
    struct Case { int W, H; };
    const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    for (const Case& c : perf) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        std::vector<uint8_t> img(N);
        unsigned st = 777u;
        for (size_t i = 0; i < N; ++i) { st = st * 1664525u + 1013904223u; img[i] = (uint8_t)((st >> 16) & 255); }
        uint8_t *dA = nullptr, *dB = nullptr;
        size_t astep = 0, bstep = 0;
        CK(cudaMallocPitch(&dA, &astep, W, H));
        CK(cudaMallocPitch(&dB, &bstep, W, H));
        CK(cudaMemcpy2D(dA, astep, img.data(), W, W, H, cudaMemcpyHostToDevice));
        const int reps = (N > 4000000) ? 20 : 50;
        double bV0 = 1e9, bV0i = 1e9;
        double best[32]; for (int i = 0; i < nV; ++i) best[i] = 1e9;
        for (int r = 0; r < 4; ++r) {
            const double m0  = timeMs(dA, astep, dB, bstep, W, H, reps, nullptr, false);
            if (m0 < bV0) bV0 = m0;
            const double m0i = timeMs(dA, astep, dB, bstep, W, H, reps, nullptr, true);
            if (m0i < bV0i) bV0i = m0i;
            for (int t = 0; t < nV; ++t) {
                const int idx = (t + r) % nV;
                const double m = timeMs(dA, astep, dB, bstep, W, H, reps, &pv[idx], false);
                if (m < best[idx]) best[idx] = m;
            }
        }
        printf("%4dx%-4d V0 %.4f  V0ip %.4f", W, H, bV0, bV0i);
        for (int i = 0; i < nV; ++i) printf("  %s %.4f", pv[i].name, best[i]);
        printf("\n"); fflush(stdout);
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    if (g_tmp) cudaFree(g_tmp);
    printf("PROBE_G15U8_SWEEP_DONE\n");
    return 0;
}
