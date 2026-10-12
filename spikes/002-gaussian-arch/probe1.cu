// probe1: gaussian 归因实验（spike 002）
// 现版 u8 gaussian：rows(u8->f32 tile MAC) + cols(f32 ring->u8)。k=15 输 OpenCV。
// 变体：
//   v0 = 现版
//   v1 = rows u8->u16 定点（k=15 权重是 [2,7,14,18,...]/64 的整型化？不行——k=15 sigma=0
//        的固定核不存在（OpenCV 固定核只到 k=9），k=15 用公式核权重是任意 float，无法定点化。
//   v1' = rows 改 2 输出/线程（f32 rows2 同款）+ uint4 装载
//   v2 = cols 改 2 列/线程
//   v3 = v1' + v2
// 另测：OpenCV separable 是 rows+cols 两 kernel，跟本库结构相同，差别只在小核调优。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

constexpr int MAX_K = 31;
constexpr int R_THREADS = 256;
constexpr int C_THREADS = 256;
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

// ---- v0 rows（现版）----
__global__ void g_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                              float* __restrict__ dst, size_t dstep, int W, int k)
{
    extern __shared__ uint8_t tile[];
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * R_THREADS;
    const int tid = threadIdx.x;
    const int half = k / 2;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int N = R_THREADS + k - 1;
    for (int i = tid; i < N; i += R_THREADS) {
        int x = x0 - half + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        tile[i] = srow[x];
    }
    __syncthreads();
    const int x = x0 + tid;
    if (x >= W) return;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += c_w[j] * tile[tid + j];
    dst[(size_t)y * (dstep >> 2) + x] = acc;
}

// ---- v1' rows：2 输出/线程 + uint4 装载（f32 rows2 同款，输入 u8）----
__global__ void g_rows2_kernel(const uint8_t* __restrict__ src, size_t sstep,
                               float* __restrict__ dst, size_t dstep, int W, int k)
{
    extern __shared__ uint8_t tile2[];      // 2*RT + k - 1
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

// ---- v0/v2 cols：v2 每线程 2 列 ----
__global__ void g_cols_kernel(const float* __restrict__ src, size_t sstep,
                              uint8_t* __restrict__ dst, size_t dstep,
                              int W, int H, int k, int rpt, bool dual)
{
    extern __shared__ float ring[];    // k * C_THREADS (v0) 或 k * 512 (v2)
    const int tid = threadIdx.x;
    const int x = dual ? (blockIdx.x * 512 + 2 * tid) : (blockIdx.x * C_THREADS + tid);
    if (x >= W) return;
    const bool two = dual && (x + 1 < W);
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k / 2;
    const size_t rows = sstep >> 2;
    const float* scol0 = src + x;
    const float* scol1 = src + x + 1;
    float* col0 = dual ? (ring + 2 * tid) : (ring + tid);
    float* col1 = ring + 2 * tid + 1;
    const int pitch = dual ? 512 : C_THREADS;

    auto fetch0 = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol0[(size_t)rr * rows];
    };
    auto fetch1 = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol1[(size_t)rr * rows];
    };

    for (int j = 0; j < k; ++j) {
        col0[j * pitch] = fetch0(j - half + y0);
        if (two) col1[j * pitch] = fetch1(j - half + y0);
    }
    int oldest = 0;
    for (int r = y0; r < y1; ++r) {
        float a0 = 0.f, a1 = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            const float wj = c_w[j];
            a0 = fmaf(wj, col0[slot * pitch], a0);
            if (two) a1 = fmaf(wj, col1[slot * pitch], a1);
            if (++slot == k) slot = 0;
        }
        int v = __float2int_rn(a0);
        dst[(size_t)r * dstep + x] = (uint8_t)max(0, min(255, v));
        if (two) {
            v = __float2int_rn(a1);
            dst[(size_t)r * dstep + x + 1] = (uint8_t)max(0, min(255, v));
        }
        if (r + 1 < y1) {
            col0[oldest * pitch] = fetch0(r + 1 + half);
            if (two) col1[oldest * pitch] = fetch1(r + 1 + half);
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

static void run_variant(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H,
                        int k, int variant, cudaStream_t st = 0)
{
    float hw[MAX_K];
    make_gauss_kernel(k, 0.f, hw);
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, hw, sizeof(float) * k));

    const int rpt = H > 1080 ? (H + 63) / 64 : (H + 15) / 16;
    const size_t fstep = (size_t)W * 4;
    float* tmp = (float*)ws_grow(fstep * H + 1024);

    if (variant == 0 || variant == 2) {
        const int smem = R_THREADS + k - 1;
        dim3 g(idiv_up(W, R_THREADS), H), t(R_THREADS);
        g_rows_kernel<<<g, t, smem, st>>>(s, ss, tmp, fstep, W, k);
    } else {
        const int RT = 128;
        const int smem = (2 * RT + k - 1);
        dim3 g(idiv_up(W, 2 * RT), H), t(RT);
        g_rows2_kernel<<<g, t, smem, st>>>(s, ss, tmp, fstep, W, k);
    }
    if (variant == 0 || variant == 1) {
        const int smem = k * C_THREADS * 4;
        dim3 g(idiv_up(W, C_THREADS), idiv_up(H, rpt)), t(C_THREADS);
        g_cols_kernel<<<g, t, smem, st>>>(tmp, fstep, d, ds, W, H, k, rpt, false);
    } else {
        const int smem = k * 512 * 4;
        dim3 g(idiv_up(W, 512), idiv_up(H, rpt)), t(C_THREADS);
        g_cols_kernel<<<g, t, smem, st>>>(tmp, fstep, d, ds, W, H, k, rpt, true);
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
        std::uniform_int_distribution<int> d(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)d(rng);

        cvo_like:;
        cv::cuda::GpuMat dG(m), dOut;
        for (int k : {5, 7, 15}) {
            auto f = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 5; ++i) f->apply(dG, dOut);
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> ms;
            for (int i = 0; i < 20; ++i) {
                cudaEventRecord(e0); f->apply(dG, dOut); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
            printf("cv  k=%-3d %8.4f ms\n", k, ms[10]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }

        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        uint8_t* d8; size_t ds8;
        CUDA_CHECK(cudaMallocPitch(&d8, &ds8, W, H));

        for (int k : {5, 7, 15}) {
            for (int v = 0; v <= 3; ++v) {
                for (int i = 0; i < 5; ++i) run_variant(s8, ss8, d8, ds8, W, H, k, v);
                CUDA_CHECK(cudaDeviceSynchronize());
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                std::vector<float> ms;
                for (int i = 0; i < 20; ++i) {
                    cudaEventRecord(e0); run_variant(s8, ss8, d8, ds8, W, H, k, v); cudaEventRecord(e1);
                    cudaEventSynchronize(e1);
                    float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
                }
                std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
                printf("v%d  k=%-3d %8.4f ms\n", v, k, ms[10]);
                cudaEventDestroy(e0); cudaEventDestroy(e1);
            }
        }
        printf("\n");
        cudaFree(s8); cudaFree(d8);
        break; // 先只跑 1080p 归因，快速迭代
        (void)0;
    }
    return 0;
}
