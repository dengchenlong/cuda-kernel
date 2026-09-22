// 腐蚀/膨胀（二值与灰度统一，min/max 语义）
// 核心优化：van Herk/Gil-Werman 分离算法 —— 任意核尺寸 O(1)/像素（OpenCV CUDA 是每轴 O(k) 的暴力窗）
// 水平/垂直各一遍。边界语义与 OpenCV morphologyDefaultBorderValue 一致：
//   腐蚀(最小值滤波)边界视为 255（恒等元），膨胀(最大值滤波)边界视为 0 —— 等价于窗口 clamp 到图像内。
#include "ops.h"
#include <algorithm>

namespace {

constexpr int VHG_B = 128;   // 段长（=block 线程数）

struct MaxOp { __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return max(a, b); } static __device__ uint8_t ident() { return 0; } };
struct MinOp { __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return min(a, b); } static __device__ uint8_t ident() { return 255; } };

// 一行/一列 = 一条线。block 处理一条线上的一个输出段(tile)。
// 需要左邻段的后缀扫描和右邻段的前缀扫描，故 smem 里加载 3 个段。
template <typename Op>
__global__ void vhg_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint8_t* __restrict__ dst, size_t dstep,
                                int W, int k)
{
    __shared__ uint8_t sin_[3 * VHG_B];
    __shared__ uint8_t sP[3 * VHG_B];   // 段内前缀扫描
    __shared__ uint8_t sS[3 * VHG_B];   // 段内后缀扫描

    const int y   = blockIdx.y;
    const int tx  = blockIdx.x;                 // 输出段索引
    const int tid = threadIdx.x;
    const Op op;

    const uint8_t* srow = src + (size_t)y * sstep;

    // 加载 [tx-1, tx+1] 三个段
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int gx = (tx - 1 + t) * VHG_B + tid;
        uint8_t v = Op::ident();
        if (gx >= 0 && gx < W) v = srow[gx];
        sin_[t * VHG_B + tid] = v;
    }
    __syncthreads();

    // 每段做前缀 + 后缀扫描（Hillis-Steele, inclusive）
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        uint8_t* seg = sin_ + t * VHG_B;
        uint8_t* P   = sP + t * VHG_B;
        uint8_t* S   = sS + t * VHG_B;
        P[tid] = seg[tid];
        S[tid] = seg[tid];
        __syncthreads();
        for (int d = 1; d < VHG_B; d <<= 1) {
            uint8_t pv = Op::ident(), sv = Op::ident();
            if (tid >= d) pv = P[tid - d];
            if (tid + d < VHG_B) sv = S[tid + d];
            __syncthreads();
            P[tid] = op(P[tid], pv);
            S[tid] = op(S[tid], sv);
            __syncthreads();
        }
    }

    // 输出：本段的 tid 号像素
    const int x = tx * VHG_B + tid;
    if (x >= W) return;

    int a = x - k / 2;              // 窗口左端
    int b = a + k - 1;              // 窗口右端
    a = max(a, 0);
    b = min(b, W - 1);

    const int ta = a / VHG_B, ja = a - ta * VHG_B;   // a 所在段（相对 tx 为 -1/0）
    const int tb = b / VHG_B, jb = b - tb * VHG_B;   // b 所在段（相对 tx 为 0/+1）
    const int ra = ta - tx + 1;
    const int rb = tb - tx + 1;

    const uint8_t va = sS[ra * VHG_B + ja];
    const uint8_t vb = sP[rb * VHG_B + jb];
    dst[(size_t)y * dstep + x] = op(va, vb);
}

template <typename Op>
__global__ void vhg_cols_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint8_t* __restrict__ dst, size_t dstep,
                                int H, int k)
{
    __shared__ uint8_t sin_[3 * VHG_B];
    __shared__ uint8_t sP[3 * VHG_B];
    __shared__ uint8_t sS[3 * VHG_B];

    const int x   = blockIdx.y;                 // 列号
    const int ty  = blockIdx.x;                 // 输出段索引（沿列方向）
    const int tid = threadIdx.x;
    const Op op;

    // 加载 3 个段（沿行方向跨列加载：同一线程固定列，行递增）
#pragma unroll
    for (int t = 0; t < 3; ++t) {
        const int gy = (ty - 1 + t) * VHG_B + tid;
        uint8_t v = Op::ident();
        if (gy >= 0 && gy < H) v = src[(size_t)gy * sstep + x];
        sin_[t * VHG_B + tid] = v;
    }
    __syncthreads();

#pragma unroll
    for (int t = 0; t < 3; ++t) {
        uint8_t* seg = sin_ + t * VHG_B;
        uint8_t* P   = sP + t * VHG_B;
        uint8_t* S   = sS + t * VHG_B;
        P[tid] = seg[tid];
        S[tid] = seg[tid];
        __syncthreads();
        for (int d = 1; d < VHG_B; d <<= 1) {
            uint8_t pv = Op::ident(), sv = Op::ident();
            if (tid >= d) pv = P[tid - d];
            if (tid + d < VHG_B) sv = S[tid + d];
            __syncthreads();
            P[tid] = op(P[tid], pv);
            S[tid] = op(S[tid], sv);
            __syncthreads();
        }
    }

    const int y = ty * VHG_B + tid;
    if (y >= H) return;

    int a = y - k / 2;
    int b = a + k - 1;
    a = max(a, 0);
    b = min(b, H - 1);

    const int ta = a / VHG_B, ja = a - ta * VHG_B;
    const int tb = b / VHG_B, jb = b - tb * VHG_B;
    const int ra = ta - ty + 1;
    const int rb = tb - ty + 1;

    const uint8_t va = sS[ra * VHG_B + ja];
    const uint8_t vb = sP[rb * VHG_B + jb];
    dst[(size_t)y * dstep + x] = op(va, vb);
}

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
uint8_t* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) {
        if (b.p) cudaFree(b.p);
        CUDA_CHECK(cudaMalloc(&b.p, need));
        b.sz = need;
    }
    return (uint8_t*)b.p;
}

inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

template <typename Op>
void morph(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
           int width, int height, int kw, int kh, cudaStream_t stream)
{
    if (width <= 0 || height <= 0) return;
    uint8_t* tmp = ws_grow((size_t)width * height + 1024);

    // 水平：每块 128 线程处理一行的一个段
    dim3 g1(idiv_up(width, VHG_B), height), t1(VHG_B);
    vhg_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, (size_t)width, width, kw);
    // 垂直：blockIdx.y = 列号
    dim3 g2(idiv_up(height, VHG_B), width), t2(VHG_B);
    vhg_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, (size_t)width, d_dst, dstStep, height, kh);
}

} // anonymous namespace

void gpu_erode(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
               int width, int height, int kw, int kh, cudaStream_t stream)
{
    morph<MinOp>(d_src, srcStep, d_dst, dstStep, width, height, kw, kh, stream);
}

void gpu_dilate(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                int width, int height, int kw, int kh, cudaStream_t stream)
{
    morph<MaxOp>(d_src, srcStep, d_dst, dstStep, width, height, kw, kh, stream);
}
