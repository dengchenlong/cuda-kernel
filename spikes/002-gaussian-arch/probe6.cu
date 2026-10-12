// probe6: gaussian u8 v4 —— OpenCV 式 4-patch 行遍 + 4-patch halo 列遍
// 行遍 v4: block 32x8, 每线程 4 输出, smem tile (4+2)*32/行 —— OpenCV row_filter 同构
// 列遍 v4: 每线程 4 输出(4 行 patch), smem halo 行复用 —— OpenCV column_filter 同构
// 对照 v3 (rows2+cols2, 已知 4K 最优)。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int MAX_K = 31;
__constant__ float c_w[MAX_K];

static void make_gauss_kernel(int ksize, double sigma, float* w)
{
    const double sigmaX = sigma > 0 ? sigma : 0.15 * ksize + 0.35;
    const double scale2X = -0.125 / (sigmaX * sigmaX);
    const int n2 = (ksize - 1) / 2;
    std::vector<double> halfV(n2 + 1);
    halfV[n2] = 1.0;
    double sumHalf = 0.0;
    for (int i = 0, x = 1 - ksize; i < n2; ++i, x += 2) {
        const double t = std::exp((double)x * x * scale2X);
        halfV[i] = t; sumHalf += t;
    }
    const double sum = 2.0 * sumHalf + 1.0;
    for (int i = 0; i <= n2; ++i) { w[i] = (float)(halfV[i] / sum); w[ksize-1-i] = w[i]; }
}

// ================= v4 rows: OpenCV 同构 =================
// 32x8 block, 每线程 4 输出, smem tile = (4+2)*32 floats/行 × 8 行
__global__ void rows4_kernel(const uint8_t* __restrict__ src, size_t sstep,
                             float* __restrict__ dst, size_t dstep, int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4, HALO = 1;
    const int half = k / 2;
    __shared__ float smem[BDY][(PPB + 2 * HALO) * BDX];

    const int y = blockIdx.y * BDY + threadIdx.y;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int xStart = blockIdx.x * (PPB * BDX) + threadIdx.x;
    float* sm = smem[threadIdx.y];

    // tile 覆盖 [xStart - HALO*BDX, xStart + (PPB+HALO)*BDX)
    auto load = [&](int idx, int x) {
        if (x < 0) x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        sm[idx] = (float)srow[x];
    };
    if (blockIdx.x > 0) {
#pragma unroll
        for (int j = 0; j < HALO; ++j) load(threadIdx.x + j * BDX, xStart - (HALO - j) * BDX);
    } else {
#pragma unroll
        for (int j = 0; j < HALO; ++j) load(threadIdx.x + j * BDX, xStart - (HALO - j) * BDX);
    }
    if (blockIdx.x + 2 < gridDim.x) {
#pragma unroll
        for (int j = 0; j < PPB; ++j) load(threadIdx.x + (HALO + j) * BDX, xStart + j * BDX);
#pragma unroll
        for (int j = 0; j < HALO; ++j) load(threadIdx.x + (PPB + HALO + j) * BDX, xStart + (PPB + j) * BDX);
    } else {
#pragma unroll
        for (int j = 0; j < PPB; ++j) load(threadIdx.x + (HALO + j) * BDX, xStart + j * BDX);
#pragma unroll
        for (int j = 0; j < HALO; ++j) load(threadIdx.x + (PPB + HALO + j) * BDX, xStart + (PPB + j) * BDX);
    }
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int x = xStart + j * BDX;
        if (x < W) {
            float sum = 0.f;
            const int base = threadIdx.x + (HALO + j) * BDX - anchor;
#pragma unroll 8
            for (int q = 0; q < k; ++q) sum = fmaf(c_w[q], sm[base + q], sum);
            dst[(size_t)y * (dstep >> 2) + x] = sum;
        }
    }
}

// ================= v4 cols: OpenCV 同构（列方向 4 patch + halo 行）=================
// block 32x8, 每线程 4 个输出行, smem tile (PPB+2*HALO)*8 行 × 32 列
__global__ void cols4_kernel(const float* __restrict__ src, size_t sstep,
                             uint8_t* __restrict__ dst, size_t dstep,
                             int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4;
    const int HALO = (k <= 16) ? 1 : 2;
    const int half = k / 2;
    __shared__ float smem[(PPB + 2 * 2) * BDY][BDX];   // 按 HALO=2 上界分配

    const int x = blockIdx.x * BDX + threadIdx.x;
    if (x >= W) return;
    const int yStart = blockIdx.y * (BDY * PPB) + threadIdx.y;
    const size_t rows = sstep >> 2;
    const float* scol = src + x;

    auto fetch = [&](int r) -> float {
        int rr = r;
        if (rr < 0) rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * rows];
    };
#pragma unroll
    for (int j = 0; j < PPB + 2 * HALO; ++j)
        smem[j * BDY + threadIdx.y][threadIdx.x] = fetch(yStart - HALO * BDY + j * BDY);
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int y = yStart + j * BDY;
        if (y < H) {
            float sum = 0.f;
#pragma unroll 8
            for (int q = 0; q < k; ++q)
                sum = fmaf(c_w[q], smem[(HALO + j) * BDY + threadIdx.y - anchor + q][threadIdx.x], sum);
            int v = __float2int_rn(sum);
            dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
        }
    }
}

// ================= v3（probe2 里最优组合）：rows2 + cols2 =================
__global__ void rows2_kernel(const uint8_t* __restrict__ src, size_t sstep,
                             float* __restrict__ dst, size_t dstep, int W, int k)
{
    extern __shared__ uint8_t tile2[];
    const int RT = blockDim.x;
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * (2 * RT);
    const int tid = threadIdx.x;
    const int N = 2 * RT + k - 1;
    const int half = k / 2;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int g0 = x0 - half;
    const bool aligned16 = ((reinterpret_cast<uintptr_t>(srow) & 15) == 0) && ((sstep & 15) == 0);
    if (aligned16 && (g0 >= 0) && (g0 + N <= W) && ((g0 & 15) == 0)) {
        const uint4* r4 = (const uint4*)(srow + g0);
        for (int i = tid; i < (N >> 4); i += RT) {
            const uint4 v = r4[i];
            const uint8_t* b = (const uint8_t*)&v;
#pragma unroll
            for (int q = 0; q < 16; ++q) tile2[i*16 + q] = b[q];
        }
        for (int i = (N & ~15) + tid; i < N; i += RT) {
            int x = g0 + i;
            if (x < 0)  x = -x;
            if (x >= W) x = 2 * W - 2 - x;
            tile2[i] = srow[x];
        }
    } else {
        for (int i = tid; i < N; i += RT) {
            int x = g0 + i;
            if (x < 0)  x = -x;
            if (x >= W) x = 2 * W - 2 - x;
            x = max(0, min(W - 1, x));
            tile2[i] = srow[x];
        }
    }
    __syncthreads();
    const int xa = x0 + tid;
    const int xb = xa + RT;
    if (xa >= W) return;
    float acca = 0.f, accb = 0.f;
    const uint8_t* pa = tile2 + tid;
    const uint8_t* pb = tile2 + tid + RT;
    for (int j = 0; j < k; ++j) {
        const float wj = c_w[j];
        acca = fmaf(wj, (float)pa[j], acca);
        if (xb < W) accb = fmaf(wj, (float)pb[j], accb);
    }
    dst[(size_t)y * (dstep >> 2) + xa] = acca;
    if (xb < W) dst[(size_t)y * (dstep >> 2) + xb] = accb;
}

__global__ void cols2_kernel(const float* __restrict__ src, size_t sstep,
                             uint8_t* __restrict__ dst, size_t dstep,
                             int W, int H, int k, int rpt)
{
    extern __shared__ float ring[];
    const int tid = threadIdx.x;
    const int x = blockIdx.x * 512 + 2 * tid;
    if (x >= W) return;
    const bool two = (x + 1 < W);
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k / 2;
    const size_t rows = sstep >> 2;
    const float* scol0 = src + x;
    const float* scol1 = src + x + 1;
    float* col0 = ring + 2 * tid;
    float* col1 = ring + 2 * tid + 1;
    auto fetch0 = [=](int r) -> float {
        int rr = r; if (rr < 0) rr = -rr; if (rr >= H) rr = 2*H - 2 - rr;
        rr = max(0, min(H - 1, rr)); return scol0[(size_t)rr * rows];
    };
    auto fetch1 = [=](int r) -> float {
        int rr = r; if (rr < 0) rr = -rr; if (rr >= H) rr = 2*H - 2 - rr;
        rr = max(0, min(H - 1, rr)); return scol1[(size_t)rr * rows];
    };
    for (int j = 0; j < k; ++j) {
        col0[j * 512] = fetch0(j - half + y0);
        if (two) col1[j * 512] = fetch1(j - half + y0);
    }
    int oldest = 0;
    for (int r = y0; r < y1; ++r) {
        float a0 = 0.f, a1 = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            const float wj = c_w[j];
            a0 = fmaf(wj, col0[slot * 512], a0);
            if (two) a1 = fmaf(wj, col1[slot * 512], a1);
            if (++slot == k) slot = 0;
        }
        int v = __float2int_rn(a0);
        dst[(size_t)r * dstep + x] = (uint8_t)max(0, min(255, v));
        if (two) { v = __float2int_rn(a1); dst[(size_t)r * dstep + x + 1] = (uint8_t)max(0, min(255, v)); }
        if (r + 1 < y1) {
            col0[oldest * 512] = fetch0(r + 1 + half);
            if (two) col1[oldest * 512] = fetch1(r + 1 + half);
            if (++oldest == k) oldest = 0;
        }
    }
}

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
static DeviceBuf g_ws;
static void* ws_grow(size_t need) {
    if (g_ws.sz < need) { if (g_ws.p) cudaFree(g_ws.p); CUDA_CHECK(cudaMalloc(&g_ws.p, need)); g_ws.sz = need; }
    return g_ws.p;
}
static inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

static void run_v3(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H, int k, cudaStream_t st = 0)
{
    const int rpt = H > 1080 ? (H + 63) / 64 : (H + 15) / 16;
    const size_t fstep = (size_t)W * 4;
    float* tmp = (float*)ws_grow(fstep * H + 1024);
    {
        const int RT = 128;
        dim3 g(idiv_up(W, 2 * RT), H), t(RT);
        rows2_kernel<<<g, t, 2 * RT + k - 1, st>>>(s, ss, tmp, fstep, W, k);
    }
    {
        dim3 g(idiv_up(W, 512), idiv_up(H, rpt)), t(256);
        cols2_kernel<<<g, t, k * 512 * 4, st>>>(tmp, fstep, d, ds, W, H, k, rpt);
    }
}
static void run_v4(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H, int k, cudaStream_t st = 0)
{
    const size_t fstep = (size_t)W * 4;
    float* tmp = (float*)ws_grow(fstep * H + 1024);
    {
        dim3 g(idiv_up(W, 32 * 4), idiv_up(H, 8)), t(32, 8);
        rows4_kernel<<<g, t, 0, st>>>(s, ss, tmp, fstep, W, H, k);
    }
    {
        const int HALO = (k <= 16) ? 1 : 2;
        dim3 g(idiv_up(W, 32), idiv_up(H, 8 * 4)), t(32, 8);
        cols4_kernel<<<g, t, 0, st>>>(tmp, fstep, d, ds, W, H, k);
    }
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int sizes[][2] = {{640,480},{1280,720},{1920,1080},{3840,2160}};
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);

        cv::cuda::GpuMat dG(m), dOut;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        uint8_t* d8; size_t ds8;
        CUDA_CHECK(cudaMallocPitch(&d8, &ds8, W, H));

        printf("== %dx%d ==\n", W, H);
        fflush(stdout);
        for (int k : {5, 7, 15}) {   // k=31: OpenCV 4.14 gaussian k=31 launch 报 invalid argument，跳过
            float hw[MAX_K];
            make_gauss_kernel(k, 0.f, hw);
            CUDA_CHECK(cudaMemcpyToSymbol(c_w, hw, sizeof(float) * k));

            // 预热
            for (int i = 0; i < 8; ++i) { run_v3(s8, ss8, d8, ds8, W, H, k); run_v4(s8, ss8, d8, ds8, W, H, k); }
            auto f = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 8; ++i) f->apply(dG, dOut);
            CUDA_CHECK(cudaDeviceSynchronize());

            auto bench = [&](auto fn) {
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                std::vector<float> ms;
                for (int i = 0; i < 20; ++i) {
                    cudaEventRecord(e0); fn(); cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
                }
                std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
                float r = ms[10];
                cudaEventDestroy(e0); cudaEventDestroy(e1);
                return r;
            };
            float cv = bench([&] { f->apply(dG, dOut); });
            float v3 = bench([&] { run_v3(s8, ss8, d8, ds8, W, H, k); });
            float v4 = bench([&] { run_v4(s8, ss8, d8, ds8, W, H, k); });
            printf("k=%-3d cv=%8.4f v3=%8.4f v4(opencv同构)=%8.4f\n", k, cv, v3, v4);
        }
        cudaFree(s8); cudaFree(d8);
    }
    return 0;
}
