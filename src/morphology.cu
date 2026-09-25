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
    __shared__ uint8_t sM[3];           // 各段整段归约

    const int y   = blockIdx.y;
    const int tx  = blockIdx.x;                 // 输出段索引
    const int tid = threadIdx.x;
    const Op op;

    const uint8_t* srow = src + (size_t)y * sstep;

    // 加载 [tx-1, tx+1] 三个段。越界位置填恒等元（腐蚀=255/膨胀=0），
    // 与 OpenCV morphologyDefaultBorderValue 的常量边界语义一致，且避免任何越界读
    // 与负索引反射算术（后者曾因 64 位地址高位算错触发 illegal memory access）。
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

    // 各段整段归约（跨段窗口的 middle 项）
    if (tid == 0) {
        uint8_t m0 = sP[0], m1 = sP[VHG_B], m2 = sP[2 * VHG_B];
        sM[0] = m0; sM[1] = m1; sM[2] = m2;
    }
    __syncthreads();

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

    // VHG 正确语义：
    //  - 同段窗口 [ja..jb]：suffix[ja]/prefix[jb] 都会多覆盖窗口外的元素 → 必须段内直接归约
    //  - 跨段窗口：suffix[ra][ja](=seg[ja..段尾]) op 中间整段 op prefix[rb][jb](=段头..jb)
    uint8_t v;
    if (ra == rb) {
        const uint8_t* seg = sin_ + ra * VHG_B;
        v = seg[ja];
        for (int i = ja + 1; i <= jb; ++i) v = op(v, seg[i]);
    } else {
        v = op(sS[ra * VHG_B + ja], sP[rb * VHG_B + jb]);
        for (int t = ra + 1; t < rb; ++t) v = op(v, sM[t]);
    }
    dst[(size_t)y * dstep + x] = v;
}

template <typename Op>
__global__ void vhg_cols_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint8_t* __restrict__ dst, size_t dstep,
                                int H, int k)
{
    __shared__ uint8_t sin_[3 * VHG_B];
    __shared__ uint8_t sP[3 * VHG_B];
    __shared__ uint8_t sS[3 * VHG_B];
    __shared__ uint8_t sM[3];           // 各段整段归约

    const int x   = blockIdx.y;                 // 列号
    const int ty  = blockIdx.x;                 // 输出段索引（沿列方向）
    const int tid = threadIdx.x;
    const Op op;

    // 加载 3 个段（沿列方向）。越界位置填恒等元（理由同行遍）。
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

    if (tid == 0) {
        sM[0] = sP[0]; sM[1] = sP[VHG_B]; sM[2] = sP[2 * VHG_B];
    }
    __syncthreads();

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

    uint8_t v;
    if (ra == rb) {
        const uint8_t* seg = sin_ + ra * VHG_B;
        v = seg[ja];
        for (int i = ja + 1; i <= jb; ++i) v = op(v, seg[i]);
    } else {
        v = op(sS[ra * VHG_B + ja], sP[rb * VHG_B + jb]);
        for (int t = ra + 1; t < rb; ++t) v = op(v, sM[t]);
    }
    dst[(size_t)y * dstep + x] = v;
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
    // 注意：临时缓冲必须用与 d_dst 相同的 step 布局（用 tight width 会造成
    // vhg_cols_kernel 以 dstStep 跨行读写时越界/读串行——之前 smoke 中 gray 小核差异即由此而来）
    size_t tmpStep = 0;
    uint8_t* tmp;
    CUDA_CHECK(cudaMallocPitch(&tmp, &tmpStep, width, height));

    // 水平：每块 128 线程处理一行的一个段
    dim3 g1(idiv_up(width, VHG_B), height), t1(VHG_B);
    vhg_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tmpStep, width, kw);
    // 垂直：blockIdx.y = 列号
    dim3 g2(idiv_up(height, VHG_B), width), t2(VHG_B);
    vhg_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, tmpStep, d_dst, dstStep, height, kh);
    cudaFree(tmp);
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
