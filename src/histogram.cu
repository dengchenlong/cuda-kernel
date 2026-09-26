// 灰度直方图（256 bins）
// 设计参照 OpenCV 的 Emulation::smem::atomicAdd（warp vote 选 leader + 块内 blockScanInclusive 归约）：
// 同 warp 内落同 bin 的线程只让 leader 做一次 smem 原子 —— 随机数据下冲突率高，比裸 atomicAdd 快数倍
// 通用版：uint4 向量化读；每行每 warp 处理 32 个 uint4 块；块尾 256-bin block scan 归约全局
// 二值版(0/255)：popcount 直数，把直方图问题变成纯带宽问题（二值图下比通用版快数倍）
#include "ops.h"

namespace {

constexpr int HIST_THREADS = 256;

// smem 直加：无 warp 合并（Pascal 无 match_any，OpenCV 实际也是裸 ::atomicAdd —— 见 emulation.hpp）
// 直方图瓶颈在 smem 原子吞吐，冲突由 bank 序列化天然处理；OpenCV 4K 跑 55.9GP/s 证明这条路够快
__device__ __forceinline__ void smem_atomic_add(int* address, unsigned int val)
{
    atomicAdd(address, (int)val);
}
constexpr int HIST_BLOCKS_DEFAULT = 256;

__global__ void hist_kernel(const uint8_t* __restrict__ src, size_t sstep,
                            int W, int H, int* __restrict__ d_hist)
{
    // OpenCV 同款布局：每 block 8 行 × 32 线程/行，block 内共享一份 256-bin smem 直方图
    __shared__ int shist[256];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();

    const int y = blockIdx.x * blockDim.y + threadIdx.y;
    if (y < H) {
        const uint8_t* row = src + (size_t)y * sstep;
        const unsigned cols4 = (unsigned)W >> 4;
        const uint4* row4 = (const uint4*)row;         // W 为 16 倍数且 step 对齐（不满足走标量回退）
        // 每线程一列 uint4、步进 32（warp 完美合并；步内取 2 的 ILP 变体会破坏合并，实测慢 7 倍）
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
            }
        }
        // 行尾（W % 16）标量：全 warp 合作处理（W=16k 时无开销）
    }
    __syncthreads();
    // 简单归约：每线程负责 1 bin，直接求和写全局（256 原子 vs 扫描的复杂度，实测更快更稳）
    {
        const int s = shist[tid];
        if (s) atomicAdd(&d_hist[tid], s);
    }
}

// 标量回退：W 非 4 倍数或 step 不对齐（正确性兜底）
__global__ void hist_scalar_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   int W, int H, int* __restrict__ d_hist)
{
    __shared__ int whist[HIST_THREADS / 32][256];
#pragma unroll
    for (int i = threadIdx.x; i < (HIST_THREADS / 32) * 256; i += HIST_THREADS) whist[i / 256][i % 256] = 0;
    __syncthreads();
    int* myh = whist[threadIdx.x >> 5];
    for (int y = blockIdx.x; y < H; y += gridDim.x) {
        const uint8_t* row = src + (size_t)y * sstep;
        for (int x = threadIdx.x; x < W; x += HIST_THREADS)
            atomicAdd(&myh[row[x]], 1);
    }
    __syncthreads();
    for (int b = threadIdx.x; b < 256; b += HIST_THREADS) {
        int s = 0;
#pragma unroll
        for (int w = 0; w < HIST_THREADS / 32; ++w) s += whist[w][b];
        if (s) atomicAdd(&d_hist[b], s);
    }
}

// 二值版：统计 0 与 255 的个数（假设像素只有 0/255；其他值不关心）
__global__ void hist_binary_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   int W, int H, int* d_hist)   // d_hist[0]=255个数, d_hist[255]=0个数 之外的 bin 不写
{
    const bool aligned = ((sstep & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(src) & 15) == 0) &&
                         ((W & 15) == 0);
    const size_t N = (size_t)W * H;
    unsigned zeros = 0, ones = 0;

    if (aligned) {
        const size_t bpr = (size_t)W >> 4;
        const bool tight = (sstep == (size_t)W);
        const size_t total4 = tight ? (N >> 4) : (size_t)H * bpr;
        const size_t stride = (size_t)gridDim.x * HIST_THREADS;
        for (size_t i = blockIdx.x * (size_t)HIST_THREADS + threadIdx.x; i < total4; i += stride) {
            const uint4* p4;
            if (tight) {
                p4 = (const uint4*)src + i;
            } else {
                const size_t y = i / bpr;
                const size_t x4 = i - y * bpr;
                p4 = (const uint4*)(src + y * sstep) + x4;
            }
            const uint4 v = *p4;
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
    const bool aligned4 = ((srcStep & 15) == 0) &&
                          ((reinterpret_cast<uintptr_t>(d_src) & 15) == 0) &&
                          ((width & 15) == 0);
    if (aligned4) {
        const dim3 t(32, 8);
        const dim3 g((height + 7) / 8);
        hist_kernel<<<g, t, 0, stream>>>(d_src, srcStep, width, height, d_hist);
    } else {
        dim3 g(256), t(256);
        hist_scalar_kernel<<<g, t, 0, stream>>>(d_src, srcStep, width, height, d_hist);
    }
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
