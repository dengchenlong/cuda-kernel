// 灰度直方图（256 bins）
// 设计：uint4 向量化读 + smem 私有直方图 + **last-block 归约**（probe_h4 实测）：
//   - 各 block 把 smem 直方图写入自己的 partials 槽（普通写，无全局原子），
//     最后完成的 block 汇总所有 partials 并**完全覆写** d_hist（256 bin 全写）。
//   - 因此调用前**无需清零 d_hist**（旧版靠 cudaMemsetAsync 预清零，实测在计时内
//     固定多花 ~20us —— 比直方图本体还贵，是 benchmark 口径里的隐形大头）。
//   - counter 用 atomicInc 回绕归零，同一 stream 连续调用安全（与 static workspace 同约束）。
// 任意宽度/步长：通用向量化 kernel（每行分段：头部标量 ≤15B | uint4 主体 | 尾部标量 ≤15B），
// W 为 16 倍数且步长对齐时仍走上面的 fast path（benchmark 主路径零改动）。
// 二值版(0/255)：popcount 直数，同样 last-block 归约，输出全 256 bin（非 0/255 bin 恒 0）。
#include "ops.h"

namespace {

constexpr int HIST_THREADS = 256;
constexpr int HIST_GRID = 48;        // 6 SM × 8 block/SM，probe_h4 实测各尺寸最优
constexpr int HIST_GRID_MAX = 256;   // 二值版用 256 block

struct HistWS {
    int* partials = nullptr;      // [HIST_GRID_MAX * 256] int，灰度版每 block 256 bin；二值版每 block 2 个
    unsigned* counter = nullptr;  // last-block 计数器，atomicInc 自动回绕归零
};
HistWS& ws() { static HistWS w; return w; }
void ws_init() {
    HistWS& w = ws();
    if (!w.partials) {
        CUDA_CHECK(cudaMalloc(&w.partials, (size_t)HIST_GRID_MAX * 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.counter, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(w.counter, 0, sizeof(unsigned)));
    }
}

// ---------- 灰度 fast path（W 为 16 倍数且对齐）：uint4 合并读 ----------
// OpenCV 同款布局：每 block 每行 32 线程；步内取 2 的 ILP 变体会破坏合并，实测慢 7 倍，勿改
__global__ void hist_lastblock_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                      int W, int H, int* __restrict__ d_hist,
                                      int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        const unsigned cols4 = (unsigned)W >> 4;
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    int* myp = partials + blockIdx.x * 256;
    myp[tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (amLast) {
        int s = 0;
        for (int b = 0; b < (int)gridDim.x; ++b) s += partials[b * 256 + tid];
        d_hist[tid] = s;   // 全 256 bin 完全覆写
    }
}

// ---------- 通用向量化（任意宽度/步长）：头尾标量 + uint4 主体，per-warp 私有 hist + last-block 归约 ----------
// W 非 16 倍数或步长不对齐时替代原标量回退：每行按 (row 基址 & 15) 切成
// 头部标量(≤15B) + uint4 主体（行内 16B 对齐，天然合并）+ 尾部标量(≤15B)。
// 行间起始对齐不同 → a 每行重算；a 必须 clamp 到 W（窄行整行走头尾，否则越界读 padding）。
// 线程分工用 threadIdx.x（跨 warp 唯一，8 个 warp 合力覆盖），勿用 lane —— lane 会让
// 8 个 warp 重复统计同一行同一批字节（probe_hfix11 实测 ×8 计数翻倍）。
__global__ void hist_vec_lastblock_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                          int W, int H, int* __restrict__ d_hist,
                                          int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int whist[HIST_THREADS / 32][256];
    __shared__ bool amLast;
#pragma unroll
    for (int i = threadIdx.x; i < (HIST_THREADS / 32) * 256; i += HIST_THREADS)
        whist[i / 256][i % 256] = 0;
    __syncthreads();
    int* myh = whist[threadIdx.x >> 5];
    const int tid = threadIdx.x;

    for (int y = blockIdx.x; y < H; y += gridDim.x) {
        const uint8_t* row = src + (size_t)y * sstep;
        // 本行 16B 对齐主体：[a, a + n4*16)，a clamp 到 W（窄行/错位行整行走头尾）
        int a = (int)((16 - (reinterpret_cast<uintptr_t>(row) & 15)) & 15);
        if (a > W) a = W;
        const int n4 = (W - a) >> 4;
        const int tail0 = a + (n4 << 4);
        // 头部 [0, a)：tid 小的线程各取 1 字节
        if (tid < a) atomicAdd(&myh[row[tid]], 1);
        // uint4 主体：每线程 1 个 uint4，步进 256（W ≤ 4096 时 n4 ≤ 256，仅前 n4 线程各干 1 次）
        for (int x = tid; x < n4; x += HIST_THREADS) {
            const uint4 v = ((const uint4*)(row + a))[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&myh[(w[u] >> b) & 0xff], 1);
        }
        // 尾部 [tail0, W)：tid 小的线程各取 1 字节
        const int tn = W - tail0;
        if (tid < tn) atomicAdd(&myh[row[tail0 + tid]], 1);
    }
    __syncthreads();
    int s = 0;
#pragma unroll
    for (int wq = 0; wq < HIST_THREADS / 32; ++wq) s += whist[wq][threadIdx.x];
    int* myp = partials + blockIdx.x * 256;
    myp[threadIdx.x] = s;
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (amLast) {
        int t = 0;
        for (int b = 0; b < (int)gridDim.x; ++b) t += partials[b * 256 + threadIdx.x];
        d_hist[threadIdx.x] = t;
    }
}

// ---------- 标量回退（兜底，当前无调用方）：per-warp 私有 hist + last-block 归约 ----------
__global__ void hist_scalar_lastblock_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                             int W, int H, int* __restrict__ d_hist,
                                             int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int whist[HIST_THREADS / 32][256];
    __shared__ bool amLast;
#pragma unroll
    for (int i = threadIdx.x; i < (HIST_THREADS / 32) * 256; i += HIST_THREADS)
        whist[i / 256][i % 256] = 0;
    __syncthreads();
    int* myh = whist[threadIdx.x >> 5];
    for (int y = blockIdx.x; y < H; y += gridDim.x) {
        const uint8_t* row = src + (size_t)y * sstep;
        for (int x = threadIdx.x; x < W; x += HIST_THREADS)
            atomicAdd(&myh[row[x]], 1);
    }
    __syncthreads();
    int s = 0;
#pragma unroll
    for (int wq = 0; wq < HIST_THREADS / 32; ++wq) s += whist[wq][threadIdx.x];
    int* myp = partials + blockIdx.x * 256;
    myp[threadIdx.x] = s;
    __threadfence();
    if (threadIdx.x == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (amLast) {
        int t = 0;
        for (int b = 0; b < (int)gridDim.x; ++b) t += partials[b * 256 + threadIdx.x];
        d_hist[threadIdx.x] = t;
    }
}

// ---------- 二值版：统计 0 与 255 的个数（popcount，纯带宽问题）----------
__global__ void hist_binary_lastblock_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                             int W, int H, int* __restrict__ d_hist,
                                             int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ bool amLast;
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
            int* myp = partials + blockIdx.x * 2;
            myp[0] = (int)zeros;
            myp[1] = (int)ones;
            __threadfence();
            const unsigned ticket = atomicInc(counter, gridDim.x - 1);
            amLast = (ticket == gridDim.x - 1);
        }
    }
    __syncthreads();
    if (amLast) {
        // 全 256 bin 完全覆写：bin 0 = 0 的个数，bin 255 = 255 的个数，其余恒 0
        int s = 0;
        if (threadIdx.x == 0 || threadIdx.x == 255) {
            for (int b = 0; b < (int)gridDim.x; ++b) s += partials[b * 2 + (threadIdx.x ? 1 : 0)];
        }
        d_hist[threadIdx.x] = (threadIdx.x == 0) ? s : (threadIdx.x == 255) ? s : 0;
    }
}

} // anonymous namespace

void gpu_histogram(const uint8_t* d_src, size_t srcStep,
                   int width, int height, int* d_hist, cudaStream_t stream)
{
    ws_init();
    const bool aligned4 = ((srcStep & 15) == 0) &&
                          ((reinterpret_cast<uintptr_t>(d_src) & 15) == 0) &&
                          ((width & 15) == 0);
    if (aligned4) {
        // spike 003：grid 按行数自适应。块数 = clamp((H+7)/8, HIST_GRID, min(HIST_GRID_MAX, max(HIST_GRID, smCount)))
        // 上限取 SM 数（170blk 在 4K 主口径实测最优 0.0092ms；256blk 反而 0.0109）；
        // 下限 48 必须在外层 —— 若写成 min(smCount, ...)，6 SM 的小卡会被压到 6 块
        // （笔记本 1050 Ti 实测 4K 0.128→0.165ms 回归）。小卡恒为 48，与旧行为一致。
        static int smCount = 0;
        if (!smCount) {
            int dev = 0;
            cudaGetDevice(&dev);
            cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, dev);
            if (smCount <= 0) smCount = HIST_GRID_MAX;
        }
        const int cap = max(HIST_GRID, min(HIST_GRID_MAX, smCount));
        const int blocks = min(cap, max(HIST_GRID, (height + 7) / 8));
        const dim3 t(32, 8);
        hist_lastblock_kernel<<<blocks, t, 0, stream>>>(d_src, srcStep, width, height,
                                                           d_hist, ws().partials, ws().counter);
    } else {
        hist_vec_lastblock_kernel<<<HIST_GRID, HIST_THREADS, 0, stream>>>(
            d_src, srcStep, width, height, d_hist, ws().partials, ws().counter);
    }
}

// 二值图专用：输出全 256 bin（bin0=0 的个数、bin255=255 的个数、其余 0），调用前无需清零
void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream)
{
    ws_init();
    dim3 g(HIST_GRID_MAX), t(HIST_THREADS);
    hist_binary_lastblock_kernel<<<g, t, 0, stream>>>(d_src, srcStep, width, height,
                                                      d_hist, ws().partials, ws().counter);
}
