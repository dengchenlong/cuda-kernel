// probe2: gaussian 公平计时 —— 先统一预热（ws_grow/首次 launch/constant 上传），
// 然后每个 (k, variant) 交错测量，消除首跑污染。
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

__global__ void g_rows2_kernel(const uint8_t* __restrict__ src, size_t sstep,
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

__global__ void g_cols_kernel(const float* __restrict__ src, size_t sstep,
                              uint8_t* __restrict__ dst, size_t dstep,
                              int W, int H, int k, int rpt, bool dual)
{
    extern __shared__ float ring[];
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
        int rr = r; if (rr < 0) rr = -rr; if (rr >= H) rr = 2*H - 2 - rr;
        rr = max(0, min(H - 1, rr)); return scol0[(size_t)rr * rows];
    };
    auto fetch1 = [=](int r) -> float {
        int rr = r; if (rr < 0) rr = -rr; if (rr >= H) rr = 2*H - 2 - rr;
        rr = max(0, min(H - 1, rr)); return scol1[(size_t)rr * rows];
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
        if (two) { v = __float2int_rn(a1); dst[(size_t)r * dstep + x + 1] = (uint8_t)max(0, min(255, v)); }
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
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw, sizeof(float) * k, 0, cudaMemcpyHostToDevice, st));

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

static double bench_variant(const uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H, int k, int v)
{
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
        cudaEventRecord(e0); run_variant(s, ss, d, ds, W, H, k, v); cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
    }
    std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return ms[10];
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
        for (int k : {5, 7, 15}) {
            // 统一预热：4 个 variant 各跑 8 次 + OpenCV 8 次（把 ws/constant/JIT 都焐热）
            for (int v = 0; v <= 3; ++v)
                for (int i = 0; i < 8; ++i) run_variant(s8, ss8, d8, ds8, W, H, k, v);
            {
                auto f = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
                for (int i = 0; i < 8; ++i) f->apply(dG, dOut);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            // 交错计时：轮次制，每轮测 cv + v0..v3，共 3 轮取各中位
            double cvm = 1e9, vm[4] = {1e9, 1e9, 1e9, 1e9};
            auto f = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
            std::vector<float> cvms, vms[4];
            for (int round = 0; round < 3; ++round) {
                {
                    std::vector<float> ms;
                    for (int i = 0; i < 20; ++i) {
                        cudaEventRecord(e0); f->apply(dG, dOut); cudaEventRecord(e1);
                        cudaEventSynchronize(e1);
                        float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
                    }
                    std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
                    cvms.push_back(ms[10]);
                }
                for (int v = 0; v <= 3; ++v) {
                    std::vector<float> ms;
                    for (int i = 0; i < 20; ++i) {
                        cudaEventRecord(e0); run_variant(s8, ss8, d8, ds8, W, H, k, v); cudaEventRecord(e1);
                        cudaEventSynchronize(e1);
                        float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
                    }
                    std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
                    vms[v].push_back(ms[10]);
                }
            }
            std::sort(cvms.begin(), cvms.end());
            cvm = cvms[cvms.size() / 2];
            for (int v = 0; v <= 3; ++v) {
                std::sort(vms[v].begin(), vms[v].end());
                vm[v] = vms[v][vms[v].size() / 2];
            }
            printf("k=%-3d cv=%8.4f | v0(rows1+cols1)=%8.4f | v1(rows2+cols1)=%8.4f | v2(rows1+cols2)=%8.4f | v3(rows2+cols2)=%8.4f\n",
                   k, cvm, vm[0], vm[1], vm[2], vm[3]);
            cudaEventDestroy(e0); cudaEventDestroy(e1);
        }
        cudaFree(s8); cudaFree(d8);
    }
    return 0;
}
