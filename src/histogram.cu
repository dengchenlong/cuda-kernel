// 灰度直方图（256 bins）
// 设计（spike 003c 定案）：**cv 同构 per-block global atomicAdd 归约**
//   - 各 block 维护 smem 私有直方图（256 int），块结束后每线程把非零 bin 一次性
//     atomicAdd 到全局 d_hist（OpenCV histogram256Kernel 同款归约）。
//   - 调用方契约不变：本函数内部先 cudaMemsetAsync 清零 d_hist（1KB，与 cv 的
//     hist.setTo(0) 同代价），再启动统计 kernel —— 多次/连续调用安全。
//   - 弃用旧 last-block 方案的原因（probe5/6 实测，4K gray）：
//     last-block 0.0100-0.0123ms vs global-atomic 0.0062-0.0064ms（cv 0.0089）。
//     __threadfence() × 170 块 + 末块串行汇总是 4K 下的尾部瓶颈，且 2160=270×8
//     时 cv 网格恰好无迭代尾巴。1080p 及以下同样占优（0.0049 vs 0.0088）。
//   - 读宽 4B（uint）与 16B（uint4）在 L2 命中场景下无显著差（probe6: 0.0064 vs 0.0063），
//     取 4B：fast path 对齐条件从 16B 放宽到 4B，覆盖面更广。
// 任意宽度/步长：通用向量化 kernel（每行分段：头部标量 ≤15B | uint4 主体 | 尾部标量 ≤15B），
// 4B 对齐不满足时走该回退路径（last-block 完全覆写，前置清零对其无害）。
// 二值版(0/255)：与灰度同一路径 —— bin0/bin255 计数正确，其余 bin 原子加 0 恒为 0；
// 非对齐回退同灰度版（输出语义一致）。
#include "ops.h"

namespace {

constexpr int HIST_THREADS = 256;
constexpr int HIST_GRID = 48;        // 回退路径的 block 数（6 SM 时代调优值，保留 Pascal 下限行为）

struct HistWS {
    int* partials = nullptr;      // 回退路径：[48 * 256] int 每 block 256 bin
    unsigned* counter = nullptr;  // last-block 计数器，atomicInc 自动回绕归零
};
HistWS& ws() { static HistWS w; return w; }
void ws_init() {
    HistWS& w = ws();
    if (!w.partials) {
        CUDA_CHECK(cudaMalloc(&w.partials, (size_t)HIST_GRID * 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&w.counter, sizeof(unsigned)));
        CUDA_CHECK(cudaMemset(w.counter, 0, sizeof(unsigned)));
    }
}

// ---------- fast path：4B 对齐，cv 同构（smem 私有 + per-block global atomicAdd） ----------
// grid = divUp(H, 8)：每 block 恰管 8 行（y<H 兜底），与 cv histogram256 完全同构。
// 调用前 d_hist 必须已清零（host 端 memsetAsync）。
__global__ void hist_ga_kernel(const uint8_t* __restrict__ src, size_t sstep,
                               int W, int H, int* __restrict__ d_hist)
{
    __shared__ int shist[256];
    const int y = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    if (y < H) {
        const unsigned* row = (const unsigned*)(src + (size_t)y * sstep);
        const int cols_4 = W >> 2;   // 对齐条件保证 W%4==0，无尾部标量
        for (int x = threadIdx.x; x < cols_4; x += blockDim.x) {
            const unsigned data = row[x];
            atomicAdd(&shist[(data >> 0) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 8) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
    }
    __syncthreads();
    const int histVal = shist[tid];
    if (histVal > 0) ::atomicAdd(d_hist + tid, histVal);
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

} // anonymous namespace

static void hist_impl(const uint8_t* d_src, size_t srcStep, int width, int height,
                      int* d_hist, cudaStream_t stream)
{
    ws_init();
    // 预清零在计时口径内（与 cv calcHist 的 hist.setTo(0) 同代价同语义），
    // fast path 的 per-block global atomicAdd 依赖它；回退路径完全覆写，清零对其无害。
    if (width > 0 && height > 0)
        CUDA_CHECK(cudaMemsetAsync(d_hist, 0, 256 * sizeof(int), stream));

    const bool aligned4 = width > 0 && height > 0 &&
                          ((srcStep & 3) == 0) &&
                          ((reinterpret_cast<uintptr_t>(d_src) & 3) == 0) &&
                          ((width & 3) == 0);
    if (aligned4) {
        const dim3 t(32, 8);
        hist_ga_kernel<<<(height + 7) / 8, t, 0, stream>>>(d_src, srcStep, width, height, d_hist);
    } else if (width > 0 && height > 0) {
        hist_vec_lastblock_kernel<<<HIST_GRID, HIST_THREADS, 0, stream>>>(
            d_src, srcStep, width, height, d_hist, ws().partials, ws().counter);
    }
}

void gpu_histogram(const uint8_t* d_src, size_t srcStep,
                   int width, int height, int* d_hist, cudaStream_t stream)
{
    hist_impl(d_src, srcStep, width, height, d_hist, stream);
}

// 二值图专用：语义上 = bin0=0 的个数、bin255=255 的个数、其余 bin 恒 0。
// 统计上与灰度同构（0/255 以外的 bin 原子加 0），直接走同一实现 —— probe6 实测该路径
// 4K 0.0060ms，优于旧 popcount last-block 方案的 0.0098ms（cv 0.0088）。
void gpu_histogram_binary(const uint8_t* d_src, size_t srcStep,
                          int width, int height, int* d_hist, cudaStream_t stream)
{
    hist_impl(d_src, srcStep, width, height, d_hist, stream);
}
