// 高斯滤波：分离两遍。中间缓冲 float（与 OpenCV cv::cuda 分离滤波的 CV_32F buf 精度一致，可逐像素比对）
// 行遍：smem tile + O(k) MAC；列遍：每线程私有 smem 滚动 ring（k 槽），每输出行只读 1 行新数据，O(1)/像素
// 权重由 host 端按 OpenCV getGaussianKernel 相同公式生成，写入 constant memory
#include "ops.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

constexpr int MAX_K = 31;
constexpr int R_THREADS = 256;   // 行遍 block 线程
constexpr int C_THREADS = 256;   // 列遍 block 线程（=256 列/块）

__constant__ float c_w[MAX_K];

// ---------------- 行遍：u8 -> float ----------------
__global__ void gauss_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  float* __restrict__ dst, size_t dstep,   // dstep 单位字节
                                  int W, int k)
{
    extern __shared__ uint8_t tile[];           // R_THREADS + k - 1
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * R_THREADS;      // 本块输出起点
    const int half = k / 2;
    const int tid = threadIdx.x;

    const uint8_t* srow = src + (size_t)y * sstep;
    const int N = R_THREADS + k - 1;

    // 加载 tile（带 reflect101 边界）
    for (int i = tid; i < N; i += R_THREADS) {
        int x = x0 - half + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));              // 超窄图保护
        tile[i] = srow[x];
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;

    float acc = 0.f;
    for (int j = 0; j < k; ++j)
        acc += c_w[j] * tile[tid + j];

    dst[(size_t)y * (dstep >> 2) + x] = acc;
}

// ---------------- 列遍：float -> u8 ----------------
// 每线程处理一列，私有 k 槽 smem ring；每输出行只从 global 读 1 个新值 → O(1)/像素
__global__ void gauss_cols_kernel(const float* __restrict__ src, size_t sstep,   // sstep 字节
                                  uint8_t* __restrict__ dst, size_t dstep,
                                  int W, int H, int k)
{
    extern __shared__ float ring[];             // k * C_THREADS，线程 tid 私有列 = ring[tid + slot*C_THREADS]
    const int tid = threadIdx.x;
    const int x = blockIdx.x * C_THREADS + tid;
    if (x >= W) return;

    const int half = k / 2;
    const size_t srows = sstep >> 2;
    const float* scol = src + x;
    float* col = ring + tid;

    auto fetch = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));            // 超矮图保护
        return scol[(size_t)rr * srows];
    };

    // 预加载窗口行 r = -half .. -half + k - 1，slot 0 = 最旧
    for (int j = 0; j < k; ++j)
        col[j * C_THREADS] = fetch(j - half);
    int oldest = 0;

    for (int r = 0; r < H; ++r) {
        float acc = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            acc += c_w[j] * col[slot * C_THREADS];
            if (++slot == k) slot = 0;
        }
        int v = __float2int_rn(acc);
        v = max(0, min(255, v));
        dst[(size_t)r * dstep + x] = (uint8_t)v;

        if (r + 1 < H) {
            col[oldest * C_THREADS] = fetch(r + 1 + half);   // 滚入新行
            if (++oldest == k) oldest = 0;
        }
    }
}

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
void* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) { if (b.p) cudaFree(b.p); CUDA_CHECK(cudaMalloc(&b.p, need)); b.sz = need; }
    return b.p;
}
inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

} // anonymous namespace

// host: 与 OpenCV getGaussianKernel(ksize, sigma, CV_32F) 完全一致的权重
static void make_gauss_kernel(int ksize, double sigma, float* w)
{
    std::vector<double> kd(ksize);
    double sigmaX = sigma > 0 ? sigma : ((ksize - 1) * 0.5 - 1.) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;
    for (int i = 0; i < ksize; ++i) {
        const double x = i - (ksize - 1) * 0.5;
        kd[i] = std::exp(x * scale2X);
        sum += kd[i];
    }
    sum = 1. / sum;
    for (int i = 0; i < ksize; ++i) w[i] = (float)(kd[i] * sum);
}

void gpu_gaussian(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int ksize, float sigma, cudaStream_t stream)
{
    float hw[MAX_K];
    make_gauss_kernel(ksize, sigma, hw);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw, sizeof(float) * ksize, 0, cudaMemcpyHostToDevice, stream));

    const size_t fstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(fstep * height + 1024);

    {
        const int smem = R_THREADS + ksize - 1;
        dim3 g(idiv_up(width, R_THREADS), height), t(R_THREADS);
        gauss_rows_kernel<<<g, t, smem, stream>>>(d_src, srcStep, tmp, fstep, width, ksize);
    }
    {
        const int smem = ksize * C_THREADS * (int)sizeof(float);
        dim3 g(idiv_up(width, C_THREADS)), t(C_THREADS);
        gauss_cols_kernel<<<g, t, smem, stream>>>(tmp, fstep, d_dst, dstStep, width, height, ksize);
    }
}
