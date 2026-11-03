// 原地（src==dst）分块卷积内核的「halo 行暂存」通用实现
// （2026-11-03 r46 在 gaussian 上引入，2026-11-04 抽成头文件并推广到 mean/morph k=3）
//
// 问题：融合内核按 TH 行分块，块要读自己上下 R 行 halo，而那些行属于相邻 block、会被相邻
// block 写掉（跨 block 无顺序保证）。整图 workspace + 整图拷回能work但白扔一遍全图流量
// （4K 0.18ms）。解法：前置小内核只把「会被别人覆盖、但本块还要读」的行拷进 stash：
//   · 每条 chunk 边界 c = j*TH 存一个 2R 行窗口 [c-R, c+R)（above R 行 + below R 行）
//   · 末块底部镜像折回的行用额外 R+2 行底段（含内核预取 y+R+1 折回的最深一行）
// 行距 = 16B 对齐的 srcStep；4K k=5(R=2) 是 138 行 = 530KB（整图 8.3MB）。
//
// 使用方必须保证（r46 probe_stash1 抓到的坑）：
//   · 块内每行一次 __syncthreads()：warp 之间没有顺序保证，快 warp 在迭代 y+R+1 已经写掉
//     该行，慢 warp 在迭代 y 预取的正是它 —— stash 只解决跨 block，块内必须靠 barrier。
//   · 预载之后也要补一次（预载读的是本块自己的行，快的 warp 可能已经在写）。
// out-of-place（dst 独立）不要走 STASH=true，否则白付 barrier。
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace halo_stash {

__host__ __device__ inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

// stash 需要多少行（调用方据此分配：(srcStep+15)&~15 行距 × rows + 1KB 余量）
template <int R>
inline int rows_needed(int height, int TH) { return idiv_up(height, TH) * 2 * R + (R + 2); }

// 行号 → stash 槽位；-1 = 本块自己的行，直接读 src
template <int R>
__device__ __forceinline__ int slot(int ry, int y0, int j0, int y1, int j1, int H, int nby)
{
    if (y1 == H && ry >= H - (R + 2))            // 末块：底部镜像行（写入之后仍要读）
        return nby * (2 * R) + (ry - (H - (R + 2)));
    if (ry < y0)                                 // above halo（属于上一块）
        return j0 * (2 * R) + (ry - (y0 - R));
    if (ry >= y1)                                // below halo（属于下一块）
        return j1 * (2 * R) + R + (ry - y1);
    return -1;
}

// stash 填充：一行一个 block（16B 向量拷贝；W%16!=0 时尾块按字节）
template <int R>
__global__ void fill_kernel(const uint8_t* __restrict__ src, size_t sstep,
                            uint8_t* __restrict__ stash, size_t tstep,
                            int W, int H, int TH)
{
    const int rowId = blockIdx.y;
    const int nby   = idiv_up(H, TH);
    const int nwin  = nby * 2 * R;
    int sy;
    if (rowId < nwin) {
        const int jw = rowId / (2 * R);
        const int t  = rowId - jw * (2 * R);
        sy = jw * TH - R + t;
    } else {
        sy = H - (R + 2) + (rowId - nwin);
    }
    sy = max(0, min(H - 1, sy));
    const uint8_t* sp = src + (size_t)sy * sstep;
    uint8_t* dp = stash + (size_t)rowId * tstep;
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    if (x + 16 <= W) {
        *(uint4*)(dp + x) = *(const uint4*)(sp + x);
    } else if (x < W) {
        for (int i = x; i < W; ++i) dp[i] = sp[i];
    }
}

// 填充 stash（同一 stream 内先于融合内核执行），返回基址并回填行距
template <int R>
const uint8_t* prepare(const uint8_t* d_src, size_t srcStep, int width, int height, int TH,
                       size_t& stashStep, void* buf, cudaStream_t stream)
{
    const int rows = rows_needed<R>(height, TH);
    stashStep = (srcStep + 15) & ~(size_t)15;    // 16B 对齐行距（8B 对齐的 src 也能用）
    uint8_t* st = (uint8_t*)buf;
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(width, 16 * th), rows);
    fill_kernel<R><<<g, th, 0, stream>>>(d_src, srcStep, st, stashStep, width, height, TH);
    return st;
}

} // namespace halo_stash
