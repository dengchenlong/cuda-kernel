// 灰度直方图（256 bins）
// 设计（r36 定案）：**三条路径统一 last-block 归约，调用前无需清零**
//   - 各 block 维护 smem 私有直方图（256 int）→ 块结束写 partials[block][256] →
//     atomicInc 计数回绕，末块 4 路累加器展开求和并**完全覆写**全 256 bin。
//   - 因此不再有调用内 cudaMemsetAsync：那 1KB memset 在本机（Windows/WDDM，每次 launch ~10us）
//     实测 10~42us，小图上占原耗时 40%+，比内核本体还贵。
//   - 历史：spike 003c 曾用 cv 同构 per-block global atomicAdd（省掉末块串行），代价是必须预清零；
//     r36 probe_hgray 重测四档尺寸，last-block（grid 固定 48）全面更快：
//     640 0.0573→0.0174、720 0.0645→0.0261、1080p 0.0788→0.0420、4K 0.1582→0.1044（ms）。
//   - 取数：16B 对齐走 uint4（比 4B 快 3~8%），只满足 4B 对齐时走 4B 版（cv 同构，覆盖面广）。
// 任意宽度/步长：通用向量化 kernel（每行分段：头部标量 ≤15B | uint4 主体 | 尾部标量 ≤15B），
// 4B 对齐不满足时走该回退路径（同样 last-block 完全覆写）。
// 二值版(0/255)：专用 popc 聚合 fast path（见 hist_bin_popc_kernel），非 16B 对齐时回退
// 通用向量化路径；两条路径输出语义一致（bin0=0 的个数、bin255=255 的个数、其余恒 0）。
#include "ops.h"

namespace {

constexpr int HIST_THREADS = 256;
constexpr int HIST_GRID = 48;        // 回退路径的 block 数（6 SM 时代调优值，保留 Pascal 下限行为）
constexpr int HIST_BIN_GRID = 48;    // 二值 popc 路径的 block 数（48×256 = 6SM×2048 满载）

struct HistWS {
    int* partials = nullptr;      // 回退路径：[48 * 256] int 每 block 256 bin
    int* bin3 = nullptr;          // 二值 popc 路径：[48 * 3] int 每 block {n0, n255, bad}
    unsigned* counter = nullptr;  // last-block 计数器，atomicInc 自动回绕归零
};
HistWS& ws() { static HistWS w; return w; }
void ws_init() {
    HistWS& w = ws();
    if (!w.partials) {
        CUDA_CHECK(cudaMalloc(&w.partials, (size_t)HIST_GRID * 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.bin3, (size_t)HIST_BIN_GRID * 3 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.counter, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(w.counter, 0, sizeof(unsigned)));
    }
}

// ---------- fast path：last-block 归约（4B / uint4 两档），调用前无需清零 ----------
// r36 改版（probe_hgray 实测，原 global-atomic + 调用内 memsetAsync 方案在 4K 为 0.1582ms）：
//   弃用 per-block global atomicAdd 的唯一理由是不想每次调用都 memset —— 但 memsetAsync 的
//   固定开销实测 10~42us（小图上占原耗时 40%+，Windows/WDDM 下每次 launch 就是 ~10us），
//   而 last-block 归约（块写 partials → 末块覆写 256 bin）在四档尺寸都更快：
//   640 0.0573→0.0174、720 0.0645→0.0261、1080p 0.0788→0.0420、4K 0.1582→0.1044。
//   形态：grid 固定 48（48×256 = 6SM×2048 满载），块内 smem 私有 256 bin，
//   末块 4 路累加器展开求和（避免单链 latency），完全覆写全 256 bin。
//   取数：16B 对齐走 uint4（快 3~8%），否则退回 4B（cv 同构，覆盖面广）。
__global__ void hist_lb_vec_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   int W, int H, int* __restrict__ d_hist,
                                   int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int n4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row = (const uint4*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < n4; x += blockDim.x) {
            const uint4 v = row[x];
            const unsigned w4[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w4[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    partials[blockIdx.x * 256 + tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!amLast) return;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, q = 0;
    for (; q + 4 <= (int)gridDim.x; q += 4) {
        a0 += partials[(q + 0) * 256 + tid];
        a1 += partials[(q + 1) * 256 + tid];
        a2 += partials[(q + 2) * 256 + tid];
        a3 += partials[(q + 3) * 256 + tid];
    }
    for (; q < (int)gridDim.x; ++q) a0 += partials[q * 256 + tid];
    d_hist[tid] = a0 + a1 + a2 + a3;
}

// 4B 版：16B 对齐不满足但 4B 对齐（W%4==0、步长 4 倍数）时使用，与 cv histogram256 同构取数
__global__ void hist_lb_4b_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  int W, int H, int* __restrict__ d_hist,
                                  int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int c4 = W >> 2;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const unsigned* row = (const unsigned*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < c4; x += blockDim.x) {
            const unsigned data = row[x];
#pragma unroll
            for (int b = 0; b < 32; b += 8)
                atomicAdd(&shist[(data >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    partials[blockIdx.x * 256 + tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!amLast) return;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, q = 0;
    for (; q + 4 <= (int)gridDim.x; q += 4) {
        a0 += partials[(q + 0) * 256 + tid];
        a1 += partials[(q + 1) * 256 + tid];
        a2 += partials[(q + 2) * 256 + tid];
        a3 += partials[(q + 3) * 256 + tid];
    }
    for (; q < (int)gridDim.x; ++q) a0 += partials[q * 256 + tid];
    d_hist[tid] = a0 + a1 + a2 + a3;
}

// ---------- 通用向量化回退（任意宽度/步长）：头尾标量 + uint4 主体，per-warp 私有 hist + last-block 归约 ----------
// W 非 4 倍数或步长不对齐时使用：每行按 (row 基址 & 15) 切成
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
        d_hist[threadIdx.x] = t;   // 全 256 bin 完全覆写
    }
}

// ---------- 二值(0/255)专用：popc 聚合 + last-block 完全覆写 ----------
// 动机（r34 实测，4K）：二值数据每个 warp 的 32 lane 全落 bin0/bin255，smem 原子 32 路串行，
// 同一实现比灰度数据慢 4.4x（0.6185ms vs 0.1413ms，字节流量完全相同）。
// r35 probe_binh3 实测（4K，只算内核本体）：旧原子路径 0.5919ms；纯读天花板 0.0922ms；
//   __vcmpeq4 版 0.6656ms（ptxas 把 vcmp4 展开成 ~12 条 PRMT/LOP/POPC，比原子还慢）；
//   popc(w)>>3 + 位掩码校验 0.0993ms —— 已到读带宽天花板（89 GB/s，纯读 90 GB/s）。
// 定案：每字一条 POPC 数 255 个数（popc/8），0 的个数 = 4 - popc/8；
//   合法性判定 m = (w>>7)&0x01010101、要求 (m*0x7F) == (w & 0x7F7F7F7F)（4 条 ALU，无 PRMT）；
//   块内一旦发现非二值字节，整块丢弃寄存器结果并回退通用 smem 原子路径（结果仍正确，仅慢）。
// 归约口径：块只写 partials/bin3（无全局原子），末块完全覆写全部 256 bin →
//   调用前无需清零，省掉 ~12~16us 的 cudaMemsetAsync（小图上占原耗时 40%+）。
__device__ __forceinline__ void hist_bin_acc(unsigned w, unsigned& n0, unsigned& n255, unsigned& bad)
{
    const unsigned m = (w >> 7) & 0x01010101u;
    bad |= (w & 0x7F7F7F7Fu) ^ (m * 0x7Fu);
    const unsigned c = __popc(w) >> 3;   // 二值数据：每 0xFF 字节 8 位
    n255 += c;
    n0   += 4u - c;
}

__global__ void hist_bin_popc_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                     int W, int H, int* __restrict__ d_hist,
                                     int* __restrict__ partials, int* __restrict__ bin3,
                                     unsigned* __restrict__ counter)
{
    __shared__ int shist[256];       // 仅回退路径（非二值数据）使用
    __shared__ int s_cnt[8][2];      // 每 warp 的 n0 / n255
    __shared__ int s_n0, s_n255, s_bad;
    __shared__ bool s_last;
    const int tid  = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (tid == 0) s_bad = 0;
    __syncthreads();

    const int n4 = W >> 4;           // host 保证 W%16==0，无尾部
    unsigned n0 = 0, n255 = 0, bad = 0;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row = (const uint4*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < n4; x += blockDim.x) {
            const uint4 v = row[x];
            hist_bin_acc(v.x, n0, n255, bad);
            hist_bin_acc(v.y, n0, n255, bad);
            hist_bin_acc(v.z, n0, n255, bad);
            hist_bin_acc(v.w, n0, n255, bad);
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        n0   += __shfl_down_sync(0xffffffffu, n0, off);
        n255 += __shfl_down_sync(0xffffffffu, n255, off);
        bad  |= __shfl_down_sync(0xffffffffu, bad, off);
    }
    if (lane == 0 && warp < 8) { s_cnt[warp][0] = (int)n0; s_cnt[warp][1] = (int)n255; }
    if (bad) atomicExch(&s_bad, 1);
    __syncthreads();
    if (tid == 0) {
        int t0 = 0, tf = 0;
#pragma unroll
        for (int wq = 0; wq < 8; ++wq) { t0 += s_cnt[wq][0]; tf += s_cnt[wq][1]; }
        s_n0 = t0; s_n255 = tf;
    }
    __syncthreads();

    if (s_bad) {
        // 非二值数据：本块负责的行改用通用原子路径重扫（寄存器统计从未提交，不会重复计数）
        shist[tid] = 0;
        __syncthreads();
        for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
            const uint4* row = (const uint4*)(src + (size_t)y * sstep);
            for (int x = threadIdx.x; x < n4; x += blockDim.x) {
                const uint4 v = row[x];
                const unsigned w4[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
                for (int u = 0; u < 4; ++u)
#pragma unroll
                    for (int b = 0; b < 32; b += 8)
                        atomicAdd(&shist[(w4[u] >> b) & 0xff], 1);
            }
        }
        __syncthreads();
        partials[blockIdx.x * 256 + tid] = shist[tid];   // 256 线程 == 256 bin
        if (tid == 0) {                                  // bin3 的 n0/n255 必须清零，
            bin3[blockIdx.x * 3 + 0] = 0;                // 否则末块会把上一轮的陈旧值再加一遍
            bin3[blockIdx.x * 3 + 1] = 0;
            bin3[blockIdx.x * 3 + 2] = 1;
        }
    } else if (tid == 0) {
        bin3[blockIdx.x * 3 + 0] = s_n0;
        bin3[blockIdx.x * 3 + 1] = s_n255;
        bin3[blockIdx.x * 3 + 2] = 0;
    }
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        s_last = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!s_last) return;

    // 末块汇总并完全覆写 256 bin（无前置清零依赖）
    if (tid == 0) {
        int t0 = 0, tf = 0, hb = 0;
        for (int q = 0; q < (int)gridDim.x; ++q) {
            t0 += bin3[q * 3 + 0];
            tf += bin3[q * 3 + 1];
            hb |= bin3[q * 3 + 2];
        }
        s_n0 = t0; s_n255 = tf; s_bad = hb;
    }
    __syncthreads();
    int val;
    if (s_bad) {
        int t = 0;
        for (int q = 0; q < (int)gridDim.x; ++q)
            if (bin3[q * 3 + 2]) t += partials[q * 256 + tid];
        if (tid == 0)   t += s_n0;
        if (tid == 255) t += s_n255;
        val = t;
    } else {
        val = (tid == 0) ? s_n0 : ((tid == 255) ? s_n255 : 0);
    }
    d_hist[tid] = val;
}

} // anonymous namespace

static void hist_impl(const uint8_t* d_src, size_t srcStep, int width, int height,
                      int* d_hist, cudaStream_t stream)
{
    if (width <= 0 || height <= 0) return;
    ws_init();
    // r36：三条路径都由 last-block 完全覆写全 256 bin，故不再做调用内 memsetAsync
    // （那 1KB memset 实测 10~42us，小图上比内核本体还贵）。
    const bool a16 = ((width & 15) == 0) && ((srcStep & 15) == 0) &&
                     ((reinterpret_cast<uintptr_t>(d_src) & 15) == 0);
    const bool a4  = ((width & 3) == 0) && ((srcStep & 3) == 0) &&
                     ((reinterpret_cast<uintptr_t>(d_src) & 3) == 0);
    const dim3 t(32, 8);
    if (a16) {
        hist_lb_vec_kernel<<<HIST_GRID, t, 0, stream>>>(
            d_src, srcStep, width, height, d_hist, ws().partials, ws().counter);
    } else if (a4) {
        hist_lb_4b_kernel<<<HIST_GRID, t, 0, stream>>>(
            d_src, srcStep, width, height, d_hist, ws().partials, ws().counter);
    } else {
        hist_vec_lastblock_kernel<<<HIST_GRID, HIST_THREADS, 0, stream>>>(
            d_src, srcStep, width, height, d_hist, ws().partials, ws().counter);
    }
}

// 二值专用分派：16B 对齐 + 宽度 16 倍数时走 popc 聚合 fast path（r35 新增，见 hist_bin_popc_kernel），
// 其余情况沿用通用实现（灰度同路径，语义一致）。
// fast path 由末块完全覆写 256 bin，故不做 memsetAsync（省 12~16us 固定开销）。
static void hist_binary_impl(const uint8_t* d_src, size_t srcStep, int width, int height,
                             int* d_hist, cudaStream_t stream)
{
    const bool fast16 = width > 0 && height > 0 &&
                        ((width & 15) == 0) && ((srcStep & 15) == 0) &&
                        ((reinterpret_cast<uintptr_t>(d_src) & 15) == 0);
    if (!fast16) {
        hist_impl(d_src, srcStep, width, height, d_hist, stream);
        return;
    }
    ws_init();
    const dim3 t(32, 8);
    hist_bin_popc_kernel<<<HIST_BIN_GRID, t, 0, stream>>>(
        d_src, srcStep, width, height, d_hist, ws().partials, ws().bin3, ws().counter);
}

void gpu_histogram(const uint8_t* d_src, size_t srcStep,
                   int width, int height, int* d_hist, cudaStream_t stream)
{
    hist_impl(d_src, srcStep, width, height, d_hist, stream);
}

// 二值图专用：语义上 = bin0=0 的个数、bin255=255 的个数、其余 bin 恒 0。
// r35 起走 popc 聚合路径（16B 对齐且 W%16==0）：二值数据 warp 内全落两个 bin，smem 原子
// 32 路串行是唯一瓶颈（r34 4K 实测 0.6185ms，灰度同代码 0.1413ms），改按字 popc 后
// 原子次数趋零；非对齐/非二值输入自动回退通用路径（正确性不受影响）。
void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream)
{
    hist_binary_impl(d_src, srcStep, width, height, d_hist, stream);
}
