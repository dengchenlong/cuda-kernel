// 灰度直方图（256 bins）
// 通用版：uint4 向量化读 + 每块共享内存私有直方图 + smem 原子累加，最后合并到全局
// 二值版(0/255)：popcount 直数，把直方图问题变成纯带宽问题（二值图下比通用版快数倍）
#include "ops.h"

namespace {

constexpr int HIST_THREADS = 256;
constexpr int HIST_BLOCKS_DEFAULT = 256;

__global__ void hist_kernel(const uint8_t* __restrict__ src, size_t sstep,
                            int W, int H, int* __restrict__ d_hist)
{
    __shared__ int shist[256];
    for (int i = threadIdx.x; i < 256; i += HIST_THREADS) shist[i] = 0;
    __syncthreads();

    const bool aligned = ((sstep & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const size_t N = (size_t)W * H;

    if (aligned) {
        const uint4* p4 = (const uint4*)src;
        const size_t n4 = N >> 4;
        const size_t stride = (size_t)gridDim.x * HIST_THREADS;
        for (size_t i = blockIdx.x * (size_t)HIST_THREADS + threadIdx.x; i < n4; i += stride) {
            const uint4 v = p4[i];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
            }
        }
        // 尾部（N % 16）标量处理
        for (size_t i = (n4 << 4) + blockIdx.x * (size_t)HIST_THREADS + threadIdx.x; i < N;
             i += (size_t)gridDim.x * HIST_THREADS)
            atomicAdd(&shist[src[i]], 1);
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += HIST_THREADS)
            if (shist[i]) atomicAdd(&d_hist[i], shist[i]);
    } else {
        // 标量回退：行宽非 16 倍数时逐像素（正确性兜底）
        for (int y = blockIdx.x; y < H; y += gridDim.x) {
            const uint8_t* row = src + (size_t)y * sstep;
            for (int x = threadIdx.x; x < W; x += HIST_THREADS)
                atomicAdd(&shist[row[x]], 1);
        }
        __syncthreads();
        for (int i = threadIdx.x; i < 256; i += HIST_THREADS)
            if (shist[i]) atomicAdd(&d_hist[i], shist[i]);
    }
}

// 二值版：统计 0 与 255 的个数（假设像素只有 0/255；其他值不关心）
__global__ void hist_binary_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   int W, int H, int* d_hist)   // d_hist[0]=255个数, d_hist[255]=0个数 之外的 bin 不写
{
    const bool aligned = ((sstep & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const size_t N = (size_t)W * H;
    unsigned zeros = 0, ones = 0;

    if (aligned) {
        const uint4* p4 = (const uint4*)src;
        const size_t n4 = N >> 4;
        const size_t stride = (size_t)gridDim.x * HIST_THREADS;
        for (size_t i = blockIdx.x * (size_t)HIST_THREADS + threadIdx.x; i < n4; i += stride) {
            const uint4 v = p4[i];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned byte255 = ((w[u] & 0xff) == 0xff) | (((w[u] >> 8) & 0xff) == 0xff) << 1 |
                                         (((w[u] >> 16) & 0xff) == 0xff) << 2 | (((w[u] >> 24) & 0xff) == 0xff) << 3;
                const unsigned byte0   = ((w[u] & 0xff) == 0x00) | (((w[u] >> 8) & 0xff) == 0x00) << 1 |
                                         (((w[u] >> 16) & 0xff) == 0x00) << 2 | (((w[u] >> 24) & 0xff) == 0x00) << 3;
                ones += __popc(byte255);
                zeros += __popc(byte0);
            }
        }
    } else {
        for (int y = blockIdx.x; y < H; y += gridDim.x) {
            const uint8_t* row = src + (size_t)y * sstep;
            for (int x = threadIdx.x; x < W; x += HIST_THREADS) {
                const uint8_t p = row[x];
                zeros += (p == 0);
                ones += (p == 255);
            }
        }
    }

    // warp 归约 + 块内共享归约
    __shared__ unsigned s0[32], s255[32];
    for (int d = 16; d > 0; d >>= 1) {
        zeros += __shfl_down_sync(0xffffffffu, zeros, d);
        ones += __shfl_down_sync(0xffffffffu, ones, d);
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { s0[wid] = zeros; s255[wid] = ones; }
    __syncthreads();
    if (wid == 0) {
        zeros = (lane < HIST_THREADS / 32) ? s0[lane] : 0;
        ones = (lane < HIST_THREADS / 32) ? s255[lane] : 0;
        for (int d = 16; d > 0; d >>= 1) {
            zeros += __shfl_down_sync(0xffffffffu, zeros, d);
            ones += __shfl_down_sync(0xffffffffu, ones, d);
        }
        if (lane == 0) {
            if (zeros) atomicAdd(&d_hist[0], zeros);
            if (ones) atomicAdd(&d_hist[255], ones);
        }
    }
}

} // anonymous namespace

void gpu_histogram(const uint8_t* d_src, size_t srcStep,
                   int width, int height, int* d_hist, cudaStream_t stream)
{
    dim3 g(HIST_BLOCKS_DEFAULT), t(HIST_THREADS);
    hist_kernel<<<g, t, 0, stream>>>(d_src, srcStep, width, height, d_hist);
}

// 二值图专用：调用前把 d_hist[0] 和 d_hist[255] 清零（其余 bin 恒为 0）
void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream);

void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream)
{
    dim3 g(HIST_BLOCKS_DEFAULT), t(HIST_THREADS);
    hist_binary_kernel<<<g, t, 0, stream>>>(d_src, srcStep, width, height, d_hist);
}
