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
    const size_t tstep = (size_t)width * 2;
    uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);

    dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
    mean_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);

    const float inv_area = 1.0f / (float)(kw * kh);
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
