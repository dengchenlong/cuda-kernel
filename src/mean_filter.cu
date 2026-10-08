// 均值滤波：分离滑动窗口，O(1)/像素（与核尺寸无关），总访存 ~6B/像素
// 行遍：u8 -> u16 行和（reflect101 边界在取数时镜像）；列遍：u16 -> u8 归一化
// 边界语义与 cv::cuda::boxFilter 默认 BORDER_DEFAULT(=REFLECT_101) 一致
#include "ops.h"
#include <algorithm>

namespace {

constexpr int MF_THREADS = 256;
constexpr int MAX_MK = 255;
constexpr int MF_PAD = 512;   // smem 扫描宽度 = MF_THREADS + MAX_MK - 1 向上取 2 的幂

template <bool IS_ROW>
__device__ __forceinline__ int mirror101(int i, int len)
{
    if (i < 0)  i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// 行遍：每 block 处理一行的一个 256 像素段。用 smem 前缀和求窗口和 —— O(1)/像素且与核尺寸无关。
// （注意：不能按线程做"步进 256 的滑动更新"，窗口每步移动 256 像素却只加/减 1 个元素是错的。）
__global__ void mean_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                 uint16_t* __restrict__ tmp, size_t tstep,   // tstep 字节
                                 int W, int k)
{
    __shared__ uint32_t sp[MF_PAD];
    const int y   = blockIdx.y;
    const int x0  = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid = threadIdx.x;
    const int N   = MF_THREADS + k - 1;              // tile 宽度（含两侧窗口延伸）
    const uint8_t* srow = src + (size_t)y * sstep;

    // 加载 tile（带 reflect101 边界），超出 N 的 padding 填 0
    for (int i = tid; i < N; i += MF_THREADS)
        sp[i] = srow[mirror101<true>(x0 - half + i, W)];
    for (int i = N + tid; i < MF_PAD; i += MF_THREADS) sp[i] = 0;
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    uint32_t sum;
    if (k <= 15) {
        // 小核：直加窗口 O(k)（Hillis-Steele 全宽扫描对小核是纯开销）
        sum = 0;
#pragma unroll 4
        for (int j = 0; j < k; ++j) sum += sp[tid + j];
    } else {
        // 大核：Hillis-Steele inclusive scan：MF_PAD 个元素，每线程负责相邻 2 个
#pragma unroll
        for (int d = 1; d < MF_PAD; d <<= 1) {
            const int i0 = 2 * tid, i1 = 2 * tid + 1;
            uint32_t a = 0, b = 0;
            if (i0 >= d) a = sp[i0 - d];
            if (i1 >= d) b = sp[i1 - d];
            __syncthreads();
            if (i0 >= d) sp[i0] += a;
            if (i1 >= d) sp[i1] += b;
            __syncthreads();
        }
        // 窗口 [x-half, x+half] = tile 索引 [tid, tid+k-1]
        sum = sp[tid + k - 1] - (tid > 0 ? sp[tid - 1] : 0);
    }
    tmp[(size_t)y * (tstep >> 1) + x] = (uint16_t)sum;
}

// 列遍：每线程一列，沿 y 滑动，u16 和 -> 归一化 u8
__global__ void mean_cols_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
                                  uint8_t* __restrict__ dst, size_t dstep,
                                  int W, int H, int kh, float inv_area)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int half = kh >> 1;
    const uint16_t* tcol = (const uint16_t*)((const uint8_t*)tmp + x * 2);
    const size_t trows = tstep >> 1;

    auto fetch = [=](int r) -> uint32_t {
        return tcol[(size_t)mirror101<false>(r, H) * trows];
    };

    uint32_t sum = 0;
    for (int j = 0; j < kh; ++j) sum += fetch(-half + j);

    for (int y = 0; y < H; ++y) {
        const int v = __float2int_rn((float)sum * inv_area);
        dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
        if (y + 1 < H)
            sum += fetch(y + 1 + half) - fetch(y - half);
    }
}

// 列遍（行分块版）：每线程处理一列的一段行区间，全场线程数 = W * ceil(H/rpt)。
// 小核（k<=15）专用：每输出行做 O(k) 直加（tmp 是 u16 行和，直加比滑动 ring 更省寄存器/更少依赖链）
__global__ void mean_cols_blocked_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
                                          uint8_t* __restrict__ dst, size_t dstep,
                                          int W, int H, int kh, float inv_area, int rpt)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int half = kh >> 1;
    const uint16_t* tcol = (const uint16_t*)((const uint8_t*)tmp + x * 2);
    const size_t trows = tstep >> 1;

    auto fetch = [=](int r) -> uint32_t {
        return tcol[(size_t)mirror101<false>(r, H) * trows];
    };

    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    for (int y = y0; y < y1; ++y) {
        uint32_t sum = 0;
#pragma unroll 4
        for (int j = -half; j <= half; ++j) sum += fetch(y + j);
        const int v = __float2int_rn((float)sum * inv_area);
        dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
    }
}

// k=3 专用：单遍 3x3 直加（smem tile，每 block 32x8 输出）
// 两遍分离法对 k=3 的 u16 tmp 往返带宽（4B/px）占比过高；单遍直加有效流量 ~2B/px。
// 整数窗口和与分离法完全一致，输出逐位相同。
__global__ void mean3_direct_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                    uint8_t* __restrict__ dst, size_t dstep,
                                    int W, int H, float inv_area)
{
    constexpr int TW = 32, TH = 8;
    constexpr int TN = (TH + 2) * (TW + 2);   // 340 字节 tile
    __shared__ uint8_t tile[TH + 2][TW + 2];
    const int x0 = blockIdx.x * TW;
    const int y0 = blockIdx.y * TH;
    const int tid = threadIdx.x;              // 256 线程

    for (int i = tid; i < TN; i += 256) {
        const int ty = i / (TW + 2), tx = i - ty * (TW + 2);
        tile[ty][tx] = src[(size_t)mirror101<false>(y0 - 1 + ty, H) * sstep +
                           mirror101<true>(x0 - 1 + tx, W)];
    }
    __syncthreads();

    const int ty = tid >> 5, tx = tid & 31;
    const int x = x0 + tx, y = y0 + ty;
    if (x < W && y < H) {
        const int sy = ty + 1, sx = tx + 1;
        uint32_t sum = tile[sy - 1][sx - 1] + tile[sy - 1][sx] + tile[sy - 1][sx + 1] +
                       tile[sy][sx - 1]     + tile[sy][sx]     + tile[sy][sx + 1] +
                       tile[sy + 1][sx - 1] + tile[sy + 1][sx] + tile[sy + 1][sx + 1];
        const int v = __float2int_rn((float)sum * inv_area);
        dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
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

void gpu_mean(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
              int width, int height, int kw, int kh, cudaStream_t stream)
{
    const float inv_area = 1.0f / (float)(kw * kh);
    if (kw == 3 && kh == 3) {
        // k=3 单遍直加：有效流量 ~2B/px，远胜两遍分离法的 6B/px
        dim3 g(idiv_up(width, 32), idiv_up(height, 8)), t(256);
        mean3_direct_kernel<<<g, t, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, inv_area);
        return;
    }
    const size_t tstep = (size_t)width * 2;
    uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);

    dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
    mean_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);

    if (kw <= 15 && kh <= 15) {
        // 小核列遍走行分块（全场 W*ceil(H/rpt) 线程）；4K 实测 rpt=64 最优，小图 rpt=16 更好
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2(idiv_up(width, MF_THREADS), idiv_up(height, rpt)), t2(MF_THREADS);
        mean_cols_blocked_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                        width, height, kh, inv_area, rpt);
    } else {
        // 大核：每列一线程滑动窗口 O(1)/像素（行分块会让每线程重复 O(k) 直加，不划算）
        dim3 g2(idiv_up(width, MF_THREADS)), t2(MF_THREADS);
        mean_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh, inv_area);
    }
}

// ================= CV_32F 输入均值滤波 =================
// 分离两遍（与 u8 版同构，元素改 float）：行遍 tile 前缀和，列遍滑动窗口。
// 窗口和 ≤255×255×255 ≈ 1.66e7 < 2^24，float 精确；输出 = 和×(1/area)，
// 与 OpenCV boxFilter(CV_32FC1) 的 float 累加同序，逐位一致（smoke 容差 1e-2 兜底）。
namespace {

__global__ void mean32_rows_kernel(const float* __restrict__ src, size_t sstep,
                                   float* __restrict__ tmp, size_t tstep,
                                   int W, int k)
{
    __shared__ float sp[MF_PAD];
    const int y   = blockIdx.y;
    const int x0  = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid = threadIdx.x;
    const int N   = MF_THREADS + k - 1;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < N; i += MF_THREADS)
        sp[i] = srow[mirror101<true>(x0 - half + i, W)];
    for (int i = N + tid; i < MF_PAD; i += MF_THREADS) sp[i] = 0.f;
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    float sum;
    if (k <= 15) {
        sum = 0.f;
#pragma unroll 4
        for (int j = 0; j < k; ++j) sum += sp[tid + j];
    } else {
        // 大核：Hillis-Steele inclusive scan（每线程负责相邻 2 元），P[i]=Σ_{j<=i} sp[j]
        for (int d = 1; d < MF_PAD; d <<= 1) {
            const int i0 = 2 * tid, i1 = 2 * tid + 1;
            float a = 0.f, b = 0.f;
            if (i0 >= d) a = sp[i0 - d];
            if (i1 >= d) b = sp[i1 - d];
            __syncthreads();
            if (i0 >= d) sp[i0] += a;
            if (i1 >= d) sp[i1] += b;
            __syncthreads();
        }
        sum = sp[tid + k - 1] - (tid ? sp[tid - 1] : 0.f);
    }
    tmp[(size_t)y * (tstep >> 2) + x] = sum;
}

__global__ void mean32_cols_blocked_kernel(const float* __restrict__ tmp, size_t tstep,
                                           float* __restrict__ dst, size_t dstep,
                                           int W, int H, int k, float inv_area, int rpt)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k >> 1;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol = tmp + x;

    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mirror101<false>(j - half + y0, H) * rows];
    for (int r = y0; r < y1; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        if (r + 1 < y1) {
            acc += scol[(size_t)mirror101<false>(r + 1 + half, H) * rows]
                 - scol[(size_t)mirror101<false>(r - half, H) * rows];
        }
    }
}

__global__ void mean32_cols_kernel(const float* __restrict__ tmp, size_t tstep,
                                   float* __restrict__ dst, size_t dstep,
                                   int W, int H, int k, float inv_area)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int half = k >> 1;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol = tmp + x;

    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mirror101<false>(j - half, H) * rows];
    for (int r = 0; r < H; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        acc += scol[(size_t)mirror101<false>(r + 1 + half, H) * rows]
             - scol[(size_t)mirror101<false>(r - half, H) * rows];
    }
}

} // anonymous namespace

void gpu_mean_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                  int width, int height, int kw, int kh, cudaStream_t stream)
{
    const float inv_area = 1.0f / (float)(kw * kh);
    const size_t tstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(tstep * height + 1024);

    dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
    mean32_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);

    if (kh <= 15) {
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2(idiv_up(width, MF_THREADS), idiv_up(height, rpt)), t2(MF_THREADS);
        mean32_cols_blocked_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                          width, height, kh, inv_area, rpt);
    } else {
        dim3 g2(idiv_up(width, MF_THREADS)), t2(MF_THREADS);
        mean32_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh, inv_area);
    }
}
