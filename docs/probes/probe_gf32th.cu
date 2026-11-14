// probe_gf32th：gaussian-f32 k=5 融合内核 TH 扫参（r57 遗留③：f32 小图 TH 微调）
// 结构同 probe_g7th：7 档 TH × 4 尺寸 × 6 轮轮转交错 min + 逐 TH 正确性断言（vs gpu_gaussian_f32 默认）。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "ops.h"
using uint = unsigned int;

__constant__ float c_w5[5];   // [1,4,6,4,1]/16，与 gpu_gaussian_f32 sigma<=0 相同（host 端 upload）

__host__ __device__ __forceinline__ int gf32_mirror(int r, int H)
{
    int rr = r;
    if (rr < 0)  rr = -rr;
    if (rr >= H) rr = 2 * H - 2 - rr;
    return max(0, min(H - 1, rr));
}
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)

static void fillPatF(std::vector<float>& h, int W, int H, unsigned seed)
{
    unsigned st = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            st = st * 1664525u + 1013904223u;
            h[(size_t)y * W + x] = (float)((int)((st >> 16) & 255)) * 0.5f;
        }
}

template <int TH>
__global__ void g32f_kernel(const float* __restrict__ src, size_t sstep,
                            float* __restrict__ dst, size_t dstep, int W, int H)
{
    constexpr int K = 5, HALF = 2;
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
        float l1 = __shfl_up_sync(0xffffffffu, v.w, 1);
        float r1 = __shfl_down_sync(0xffffffffu, v.x, 1);
        float l2 = __shfl_up_sync(0xffffffffu, v.z, 1);
        float r2 = __shfl_down_sync(0xffffffffu, v.y, 1);
        if (lane == 0 || xa == 0) {
            if (xa == 0) { l1 = v.y; l2 = v.z; }
            else         { l1 = srow[xa - 1]; l2 = srow[xa - 2]; }
        }
        if (lane == 31 || xa + 4 >= W) {
            if (xa + 4 >= W) { r1 = v.z; r2 = v.y; }
            else             { r1 = srow[xa + 4]; r2 = srow[xa + 5]; }
        }
        float pb[4 + 2 * HALF];
        pb[0] = l2; pb[1] = l1;
        pb[HALF + 0] = v.x; pb[HALF + 1] = v.y; pb[HALF + 2] = v.z; pb[HALF + 3] = v.w;
        pb[HALF + 4] = r1; pb[HALF + 5] = r2;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < K; ++q) s = fmaf(c_w5[q], pb[i + q], s);
            o[i] = s;
        }
    };

#pragma unroll
    for (int i = 0; i < K; ++i) hrow(gf32_mirror(y0 - HALF + i, H), h[i]);
    for (int y = y0; y < y1; ++y) {
        float hn[4];
        hrow(gf32_mirror(y + HALF + 1, H), hn);
        if (act) {
            float vv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = 0.f;
#pragma unroll
                for (int q = 0; q < K; ++q) s = fmaf(c_w5[q], h[q][j], s);
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

static double timeOne(float* dA, size_t astep, float* dB, size_t bstep, int W, int H,
                      int TH, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    const int th = std::min(256, std::max(32, (((W + 3) / 4) + 31) / 32 * 32));
    dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        switch (TH) {
        case 8:   g32f_kernel<8>  <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        case 16:  g32f_kernel<16> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        case 32:  g32f_kernel<32> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        case 48:  g32f_kernel<48> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        case 64:  g32f_kernel<64> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        case 96:  g32f_kernel<96> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
        default:  g32f_kernel<128><<<g, th>>>(dA, astep, dB, bstep, W, H); break;
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
    {   // 与 src/gaussian.cu make_gauss_kernel k=5 sigma<=0 相同的 [1,4,6,4,1]/16
        const float hw[5] = {1.f/16, 4.f/16, 6.f/16, 4.f/16, 1.f/16};
        CK(cudaMemcpyToSymbol(c_w5, hw, sizeof(hw)));
    }
    int badTotal = 0;
    struct Case { int W, H; };
    const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int THs[] = {8, 16, 32, 48, 64, 96, 128};

    for (const Case& c : perf) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        std::vector<float> img(N);
        fillPatF(img, W, H, 777u);
        float *dA, *dB; size_t astep, bstep;
        CK(cudaMallocPitch(&dA, &astep, (size_t)W * 4, H));
        CK(cudaMallocPitch(&dB, &bstep, (size_t)W * 4, H));
        CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W * 4, (size_t)W * 4, H, cudaMemcpyHostToDevice));
        // 参照：默认 gpu_gaussian_f32（out-of-place）
        gpu_gaussian_f32(dA, astep, dB, bstep, W, H, 5, 0.f);
        CK(cudaDeviceSynchronize());
        std::vector<float> ref(N);
        CK(cudaMemcpy2D(ref.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
        // 逐 TH 正确性
        for (int TH : THs) {
            const int th = std::min(256, std::max(32, (((W + 3) / 4) + 31) / 32 * 32));
            dim3 g((unsigned)((W + 4 * th - 1) / (4 * th)), (unsigned)((H + TH - 1) / TH));
            switch (TH) {
            case 8:   g32f_kernel<8>  <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            case 16:  g32f_kernel<16> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            case 32:  g32f_kernel<32> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            case 48:  g32f_kernel<48> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            case 64:  g32f_kernel<64> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            case 96:  g32f_kernel<96> <<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            default:  g32f_kernel<128><<<g, th>>>(dA, astep, dB, bstep, W, H); break;
            }
            CK(cudaDeviceSynchronize());
            std::vector<float> out(N);
            CK(cudaMemcpy2D(out.data(), (size_t)W * 4, dB, bstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            int bad = 0; float md = 0;
            for (size_t i = 0; i < N; ++i)
                if (out[i] != ref[i]) { ++bad; md = std::max(md, (float)fabs((double)out[i] - ref[i])); }
            if (bad) { printf("  TH%d W=%d H=%d bad=%d maxdiff=%g\n", TH, W, H, bad, md); ++badTotal; }
        }
        // 6 轮轮转交错 min
        const int repsN = (N > 4000000) ? 20 : 60;
        double best[7]; for (int i = 0; i < 7; ++i) best[i] = 1e9;
        for (int r = 0; r < 6; ++r)
            for (int t = 0; t < 7; ++t) {
                const int idx = (t + r) % 7;
                const double ms = timeOne(dA, astep, dB, bstep, W, H, THs[idx], repsN);
                if (ms < best[idx]) best[idx] = ms;
            }
        printf("%4dx%-4d", W, H);
        for (int i = 0; i < 7; ++i) printf("  T%d %.4f", THs[i], best[i]);
        printf("\n"); fflush(stdout);
        CK(cudaFree(dA)); CK(cudaFree(dB));
    }
    printf(badTotal ? "PROBE_GF32TH_CHECK_FAIL bad=%d\n" : "PROBE_GF32TH_OK\n", badTotal);
    return 0;
}
