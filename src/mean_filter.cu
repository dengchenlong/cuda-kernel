// 均值滤波：分离滑动窗口，O(1)/像素（与核尺寸无关），总访存 ~6B/像素
// 行遍：u8 -> u16 行和（reflect101 边界在取数时镜像）；列遍：u16 -> u8 归一化
// 边界语义与 cv::cuda::boxFilter 默认 BORDER_DEFAULT(=REFLECT_101) 一致
#include "ops.h"
#include <algorithm>

namespace {

constexpr int MF_THREADS = 256;
constexpr int MAX_MK = 255;

template <bool IS_ROW>
__device__ __forceinline__ int mirror101(int i, int len)
{
    if (i < 0)  i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// 行遍：每 block 处理一行的一个 256 像素段，线程内沿 tile 步进做滑动和
__global__ void mean_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                 uint16_t* __restrict__ tmp, size_t tstep,   // tstep 字节
                                 int W, int k)
{
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid = threadIdx.x;
    const uint8_t* srow = src + (size_t)y * sstep;

    auto fetch = [=](int x) -> uint32_t {
        return srow[mirror101<true>(x, W)];
    };

    const int x = x0 + tid;
    if (x >= W) return;

    // 初始窗口和
    uint32_t sum = 0;
    for (int j = 0; j < k; ++j) sum += fetch(x - half + j);
    tmp[(size_t)y * (tstep >> 1) + x] = (uint16_t)sum;

    // 同一线程沿 x 方向按 block 宽度步进：滑动更新
    for (int xx = x + MF_THREADS; xx < W; xx += MF_THREADS) {
        sum += fetch(xx + half) - fetch(xx - half - 1);
        tmp[(size_t)y * (tstep >> 1) + xx] = (uint16_t)sum;
    }
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

    dim3 g2(idiv_up(width, MF_THREADS)), t2(MF_THREADS);
    mean_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh,
                                             1.0f / (float)(kw * kh));
}
