// 积分图：两遍法
// Pass1: 行内前缀和 —— 每 warp 处理一行，uint4 向量化读入(每 lane 16B)，warp shuffle 扫描
// Pass2: 列方向扫描 —— 分层：先扫 chunk，再用 carry 矩阵加前缀（对 W%4!=0 有标量回退路径）
#include "ops.h"
#include <vector>
#include <algorithm>
#include <cstdlib>

namespace {

constexpr int SEG_BYTES = 512;      // 32 lanes * 16B
constexpr int CHUNK_H   = 512;      // 列扫描 chunk 高度
constexpr int CT_THREADS = 128;     // 列扫描 block 线程数 (每线程 4 列 → 512 列/block)

// ---------------- Pass 1: 行扫描 ----------------
// v2 launch shape（spike 001 实测）：每 block 8 行、每 warp 独占一行，grid=(H+7)/8。
// 旧版固定 12 block×256（96 warp）在 sm_120（170 SM）上只占一小部分 SM，
// 4K 0.53ms；新 shape 4K 0.19ms（2.8x），全尺寸超 OpenCV（720p 4.1x/4K 2.2x）。
// 内核逐指令与旧版 grid-stride 版一致，数值路径零改动（sm_61 同样受益或持平）。
//
// r50（probe_int1 分段实测）：这段曾是全库最大的单点损失 —— 4K 1.34ms 只有 31 GB/s，
// 而 pass2 的同类内核能跑 93 GB/s。根因不是算法而是**写粒度**：每 lane 负责 16 字节、
// 前缀和逐条 u32 标量存，一条 store 的 32 个 lane 落在 32 个相隔 64B 的地址上
// （= 32 个不同 sector，每个 sector 只写 4/32 字节）→ 写放大把带宽压到 1/3。
// 改成「16 个前缀先算进寄存器，再用 4 条 uint4 store 写回」后 4K 0.468ms（89 GB/s，2.9x），
// 数值路径零改动（同一批 run 值，只是打包方式不同）。要求 dst 侧 16B 对齐，否则回落标量存。
__global__ void integral_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                     uint32_t* __restrict__ dst, size_t dstep,
                                     int W, int H)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 512;      // 32 lanes * 16B
    const bool ldVec = ((sstep & 15) == 0) &&
                       ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const bool stVec = ((dstep & 15) == 0) &&
                       ((reinterpret_cast<uintptr_t>(dst) & 15) == 0);

    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;

    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;          // 本 lane 负责 16 字节
        uint32_t vals[16];
        uint32_t partial = 0;
        const bool full = (my0 + 16 <= W);

        if (full && ldVec) {
            // 快路径：对齐 uint4 读
            union { uint4 v; uint8_t b[16]; } u;
            u.v = *reinterpret_cast<const uint4*>(srow + my0);
#pragma unroll
            for (int k = 0; k < 16; ++k) { vals[k] = u.b[k]; partial += vals[k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int x = my0 + k;
                uint32_t v = 0;
                if (x < W) v = srow[x];
                vals[k] = v; partial += v;
            }
        }

        // warp 内 inclusive scan
        uint32_t inc = partial;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        const uint32_t excl = inc - partial + carry;
        const uint32_t seg_total = __shfl_sync(0xffffffffu, inc, 31);
        carry += seg_total;

        uint32_t run = excl;
        if (full && stVec) {
            uint32_t o[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) { run += vals[k]; o[k] = run; }
            uint4* op = (uint4*)(drow + my0);
            op[0] = make_uint4(o[0],  o[1],  o[2],  o[3]);
            op[1] = make_uint4(o[4],  o[5],  o[6],  o[7]);
            op[2] = make_uint4(o[8],  o[9],  o[10], o[11]);
            op[3] = make_uint4(o[12], o[13], o[14], o[15]);
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                run += vals[k];
                const int x = my0 + k;
                if (x < W) drow[x] = run;
            }
        }
    }
}

// ---------------- Pass 2: 列方向扫描 ----------------
// r50（probe_int1 实测）：旧的「按 CHUNK_H 分块 + carry 回加」策略在 sm_61 上是净亏损 ——
// chunk 内的列扫描要读 4N 写 4N，回加 carry 再读 4N 写 4N，共 16N 字节；而列扫描本身
// 不需要分块就能一次走完（每线程独占自己的列，读依赖只在累加器里）。实测 4K：
// 分块 0.707+0.536=1.243ms vs 单遍 0.708ms（1col）/0.691ms（4col）→ 单遍省掉一半流量。
// 单遍的代价是并行度只剩 W 个线程（分块版是 W × nChunks），所以 W 小时仍保留分块路径：
// 实测 4K(W=3840) 单遍 1col 0.708 完胜分块 1.243；1080p(W=1920) 0.183 vs 0.363；720p 0.087 vs 0.172；
// 640 虽然 nChunks=1（本来就不分块）但 1col 的 640 线程比旧 4col 的 160 线程快 33%（0.044 vs 0.0655）。
// 阈值取 W >= 512。
__global__ void integral_cols_1col_kernel(const uint32_t* __restrict__ in, size_t istep,
                                          uint32_t* __restrict__ out, size_t ostep,
                                          int W, int H)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    const uint32_t* ip = (const uint32_t*)((const uint8_t*)in) + c;
    uint32_t* op = (uint32_t*)((uint8_t*)out) + c;
    uint32_t car = 0;
    for (int r = 0; r < H; ++r) {
        car += *ip;
        *op = car;
        ip = (const uint32_t*)((const uint8_t*)ip + istep);
        op = (uint32_t*)((const uint8_t*)op + ostep);
    }
}

// ---------------- Pass 2 回退: chunk 内列扫描 + 写 carry 矩阵 ----------------
// 每线程处理 4 列 (uint4)，串行走 chunkH 行
__global__ void integral_cols_kernel(const uint32_t* __restrict__ in, size_t istep,
                                     uint32_t* __restrict__ out, size_t ostep,
                                     uint32_t* __restrict__ carry,   // [nChunks][W]
                                     int W, int H, int nChunks)
{
    const int chunk = blockIdx.y;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;

    const uint4* ip = (const uint4*)((const uint8_t*)in  + (size_t)r0 * istep) + c0 / 4;
    uint4*       op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;

    uint4 car = make_uint4(0, 0, 0, 0);
    for (int r = r0; r < r1; ++r) {
        uint4 v = *ip;
        v.x += car.x; v.y += car.y; v.z += car.z; v.w += car.w;
        *op = v;
        car = v;
        ip = (const uint4*)((const uint8_t*)ip + istep);
        op = (uint4*)((uint8_t*)op + ostep);
    }
    if (nChunks > 1) {
        uint32_t* cr = carry + (size_t)chunk * W + c0;
        cr[0] = car.x; cr[1] = car.y; cr[2] = car.z; cr[3] = car.w;
    }
}

// ---------------- Pass 2b: 把前面 chunk 的 carry 累加进本 chunk ----------------
__global__ void integral_carry_add_kernel(uint32_t* __restrict__ out, size_t ostep,
                                          const uint32_t* __restrict__ carry, // [nChunks][W]
                                          int W, int H)
{
    const int chunk = blockIdx.y + 1;                 // chunk 0 无需处理
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;

    uint4 exc = make_uint4(0, 0, 0, 0);
    for (int j = 0; j < chunk; ++j) {
        const uint32_t* cr = carry + (size_t)j * W + c0;
        exc.x += cr[0]; exc.y += cr[1]; exc.z += cr[2]; exc.w += cr[3];
    }

    uint4* op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;
    for (int r = r0; r < r1; ++r) {
        uint4 v = *op;
        v.x += exc.x; v.y += exc.y; v.z += exc.z; v.w += exc.w;
        *op = v;
        op = (uint4*)((uint8_t*)op + ostep);
    }
}

// ---------------- 标量回退 (W%4 != 0 或步长不对齐时) ----------------
__global__ void integral_cols_scalar_kernel(const uint32_t* __restrict__ in, size_t istep,
                                            uint32_t* __restrict__ out, size_t ostep,
                                            uint32_t* __restrict__ carry,
                                            int W, int H, int nChunks)
{
    const int chunk = blockIdx.y;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;

    const uint32_t* ip = (const uint32_t*)((const uint8_t*)in  + (size_t)r0 * istep) + c;
    uint32_t*       op = (uint32_t*)((uint8_t*)out + (size_t)r0 * ostep) + c;
    uint32_t car = 0;
    for (int r = r0; r < r1; ++r) {
        car += *ip;
        *op = car;
        ip = (const uint32_t*)((const uint8_t*)ip + istep);
        op = (uint32_t*)((uint8_t*)op + ostep);
    }
    if (nChunks > 1) carry[(size_t)chunk * W + c] = car;
}

__global__ void integral_carry_add_scalar_kernel(uint32_t* __restrict__ out, size_t ostep,
                                                 const uint32_t* __restrict__ carry,
                                                 int W, int H)
{
    const int chunk = blockIdx.y + 1;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;

    uint32_t exc = 0;
    for (int j = 0; j < chunk; ++j) exc += carry[(size_t)j * W + c];

    uint32_t* op = (uint32_t*)((uint8_t*)out + (size_t)r0 * ostep) + c;
    for (int r = r0; r < r1; ++r) {
        *op += exc;
        op = (uint32_t*)((uint8_t*)op + ostep);
    }
}

// ================= r51: 融合单遍路径（6N 字节） =================
// 现状（r50 后）：pass1 行前缀写 4N + pass2 读 4N 写 4N = 13N 字节，4K 1.18ms 已贴带宽天花板
// （89~93 GB/s），所以只能砍流量。理想流量 = 读 src 1N + 写 dst 4N = 5N。
// 单遍的障碍是「写 dst[y][x] 前必须知道上方所有行在 x 列的累加和」，而块间无顺序保证。
// 解法：把这个「上方累加和」提前独立算出来（它只依赖 src，不依赖任何块的输出）：
//   A) chunk_sum：把图像按 TH 行切成 nChunks 段，每段按列求「这 TH 行的行和」（只读 1N，
//      写一个 nChunks*W*4 的极小数组，4K 是 34*3840*4 = 522KB）。
//   B) chunk_scan：每线程一列，把 A 的结果原地改成跨 chunk 的 exclusive 前缀 = dst[r0-1][x]。
//   C) fused：每 block 一个 chunk，256 线程 × 16 列，寄存器里存 16 个列累加器；逐行做块内
//      行扫描（warp shuffle + 跨 warp smem）得到行内前缀，加进累加器后直接写 dst。
// 流量 = 1N(A 读) + 1N(C 读) + 4N(C 写) = 6N，理论 2.17x 于 13N。
// 约束：W%16==0 且 W<=4096（256 线程 × 16 列）且 src/dst 与步长 16B 对齐；其余走旧两遍路径。
// H<64 也走旧路径（块数太少，融合路径的启动/前缀开销反超）。

__global__ void integral_chunk_sum_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                          uint32_t* __restrict__ csum, int W, int H, int TH)
{
    const int ngroups = W >> 4;                 // 每 chunk 的 16 列组数
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ((H + TH - 1) / TH) * ngroups;
    if (gid >= total) return;
    const int chunk = gid / ngroups;
    const int c0 = (gid - chunk * ngroups) * 16;
    const int r0 = chunk * TH;
    const int r1 = min(H, r0 + TH);

    uint32_t acc[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) acc[k] = 0;
    const uint8_t* p = src + (size_t)r0 * sstep + c0;
    for (int r = r0; r < r1; ++r) {
        union { uint4 v; uint8_t b[16]; } u;
        u.v = *reinterpret_cast<const uint4*>(p);
#pragma unroll
        for (int k = 0; k < 16; ++k) acc[k] += u.b[k];
        p += sstep;
    }
    uint32_t* o = csum + (size_t)chunk * W + c0;
    uint4* op = (uint4*)o;
    op[0] = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
    op[1] = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
    op[2] = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
    op[3] = make_uint4(acc[12], acc[13], acc[14], acc[15]);
}

// B：跨 chunk 的 exclusive 前缀（每线程一列，原地改写 csum）
__global__ void integral_chunk_scan_kernel(uint32_t* __restrict__ csum, int W, int nChunks)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;
    uint32_t s = 0;
    for (int k = 0; k < nChunks; ++k) {
        uint32_t* pp = csum + (size_t)k * W + c;
        const uint32_t v = *pp;
        *pp = s;
        s += v;
    }
}

// ---------------- r51 融合单遍路径的调度参数 ----------------
// 目标：nChunks ≈ 64（6 SM 上每 SM ~10 个 block，足够填坑又不至于让 chunk_sum 的前缀数组变大）。
// TH 取 32 的倍数；下限 32（行内屏障/dst 行距跳转的固定成本摊薄）；上限 256。
// CK_TH_INT / CK_NOFUSE 仅供 probe 扫参与 A/B 用，正常路径不受影响；
// 两者都在首次访问时缓存进 static（getenv 在 0.03ms 级小图上不可忽略）。
constexpr int FUSED_MIN_H = 96;      // H 太小时块数不够，走旧两遍更稳

inline int env_th_override() {
    static int v = []() {
        const char* e = getenv("CK_TH_INT");
        return (e && *e) ? atoi(e) : 0;
    }();
    return v;
}

inline bool env_nofuse() {
    static bool b = (getenv("CK_NOFUSE") != nullptr);
    return b;
}

inline int fused_th(int H) {
    const int ov = env_th_override();
    if (ov >= 8) return ov;
    // r52 实测修正（probe_int10，5 轮交错 A/B，四个尺寸各自扫 TH=32..256）：
    //   4K      TH=96 (nC=23) 0.4626ms 89.7 GB/s  ← 最优；TH=64 (nC=34，原默认) 0.5663 仅 73.2
    //   1080p   TH=32 (nC=34) 0.1290 最优，TH=48 打平（1% 内），TH≥64 递减
    //   720p    TH=32 (nC=23) 0.0645 最优；640  TH=32 (nC=15) 最优（TH=32 是下限）
    // 甜点 = 「每 SM ~4 个常驻 block」⇒ nChunks ≈ 24（fth=256 时 60 regs × 256 = 15360，
    // 65536 regs/SM ⇒ 4 块/SM）。旧规则 ceil(H/64) 在 4K 上给 34 个 chunk（每 SM 5.7 块），
    // 块多了但每块只有 64 行、屏障占比例反而升高。新规则直接按 nChunks≈24 定 TH。
    int th = (H + 23) / 24;                 // ceil(H/24)
    th = ((th + 16) / 32) * 32;             // 取最近的 32 倍数
    if (th < 32) th = 32;
    if (th > 256) th = 256;
    return th;
}

// 块内 exclusive 前缀扫描（v2：每行只 1 次 __syncthreads）。
// 旧版每行 2 次屏障（warp 扫描后写 smem → 屏障 → 8 lane 再扫 → 屏障）。
// v2 去掉第二次屏障：跨 warp 偏移改为「每个线程自己把 wsum[0..warp-1] 加起来」
// （nwarps ≤ 8，全是同地址广播读，比 shfl 链还便宜），并用**双缓冲 wsum[2][8]** 消除
// 「下一轮写 / 本轮读」的竞争 —— 屏障保证线程间最多 1 轮 skew，双缓冲刚好够。
struct FusedSmem {
    uint32_t wsum[2][8];
    uint32_t wsum_init[8];
};

// C：融合内核（每 block 一个 chunk，TH 行 × 全 W）
// 线程数按宽度自适应：th = ceil(W/16) 向上取整到 warp 倍数（W=3840 → 256，1920 → 128，
// 1280 → 96，640 → 64）。旧版固定 256 线程，1080p 有 136/256 线程空转。
__global__ void integral_fused_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                      uint32_t* __restrict__ dst, size_t dstep,
                                      const uint32_t* __restrict__ pre,   // [nChunks][W] 或 null
                                      int W, int H, int TH)
{
    __shared__ FusedSmem sm;
    const int chunk = blockIdx.x;
    const int r0 = chunk * TH;
    if (r0 >= H) return;
    const int r1 = min(H, r0 + TH);
    const int tid  = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c0 = tid * 16;
    // W%16==0 但 W 不是 (32*16) 的整数倍时最后一两个线程越界。**不能提前 return**：
    // 那样该 warp 不会写 wsum[]，跨 warp 前缀读到未初始化值（smoke 128x128 抓到 maxdiff=253）。
    // 改为「保留线程 + valid 掩码」：无效线程贡献 seg=0、跳过 load/store，屏障与扫描保持齐整。
    const bool valid = (c0 < W);

    // ---- 起始列累加器 = 「前 chunk 个 chunk 的列和」再做一次块内沿 x 的前缀 ----
    // pre[k][x] 是列和（不是 2D 前缀），而 acc 要的是 2D 前缀 = 列和沿 x 的前缀。
    uint32_t acc[16];
    {
        uint32_t pv[16];
        uint32_t seg = 0;
        if (pre && valid) {
            const uint4* pp = (const uint4*)(pre + (size_t)chunk * W + c0);
            const uint4 a0 = pp[0], a1 = pp[1], a2 = pp[2], a3 = pp[3];
            pv[0] = a0.x;  pv[1] = a0.y;  pv[2] = a0.z;  pv[3] = a0.w;
            pv[4] = a1.x;  pv[5] = a1.y;  pv[6] = a1.z;  pv[7] = a1.w;
            pv[8] = a2.x;  pv[9] = a2.y;  pv[10] = a2.z; pv[11] = a2.w;
            pv[12] = a3.x; pv[13] = a3.y; pv[14] = a3.z; pv[15] = a3.w;
            const uint32_t q0 = pv[0] + pv[1] + pv[2] + pv[3];
            const uint32_t q1 = pv[4] + pv[5] + pv[6] + pv[7];
            const uint32_t q2 = pv[8] + pv[9] + pv[10] + pv[11];
            const uint32_t q3 = pv[12] + pv[13] + pv[14] + pv[15];
            seg = (q0 + q1) + (q2 + q3);
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) pv[k] = 0;
        }
        // 起始扫描用独立的小数组（不用双缓冲，命中一次就够）
        uint32_t inc = seg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        if (lane == 31) sm.wsum_init[warp] = inc;
        __syncthreads();
        uint32_t off = 0;
        for (int w = 0; w < warp; ++w) off += sm.wsum_init[w];
        uint32_t run = off + (inc - seg);
#pragma unroll
        for (int k = 0; k < 16; ++k) { run += pv[k]; acc[k] = run; }
        __syncthreads();     // wsum_init 下一轮（下一行）不会再用，但保持对称开销可忽略
    }

    const uint8_t* p = src + (size_t)r0 * sstep + c0;
    uint32_t* q = (uint32_t*)((uint8_t*)dst + (size_t)r0 * dstep) + c0;
    const unsigned ones = 0x01010101u;
    for (int r = r0; r < r1; ++r) {
        uint32_t o[16];
        uint32_t seg = 0;
        if (valid) {
            const uint4 v = *reinterpret_cast<const uint4*>(p);
            o[0] = v.x & 0xffu;         o[1] = (v.x >> 8) & 0xffu;
            o[2] = (v.x >> 16) & 0xffu; o[3] = v.x >> 24;
            o[4] = v.y & 0xffu;         o[5] = (v.y >> 8) & 0xffu;
            o[6] = (v.y >> 16) & 0xffu; o[7] = v.y >> 24;
            o[8] = v.z & 0xffu;         o[9] = (v.z >> 8) & 0xffu;
            o[10] = (v.z >> 16) & 0xffu; o[11] = v.z >> 24;
            o[12] = v.w & 0xffu;        o[13] = (v.w >> 8) & 0xffu;
            o[14] = (v.w >> 16) & 0xffu; o[15] = v.w >> 24;
            // 段和用 4 条 __dp4a（u8x4 点积，sm_61 起支持）替代 15 条加法
            seg = __dp4a(v.x, ones, __dp4a(v.y, ones, __dp4a(v.z, ones, __dp4a(v.w, ones, 0u))));
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) o[k] = 0;
        }

        uint32_t inc = seg;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        const int buf = r & 1;
        if (lane == 31) sm.wsum[buf][warp] = inc;
        __syncthreads();                        // 每行仅此一次屏障
        uint32_t off = 0;
        for (int w = 0; w < warp; ++w) off += sm.wsum[buf][w];
        const uint32_t run0 = off + (inc - seg);
        if (valid) {
            uint32_t run = run0;
#pragma unroll
            for (int k = 0; k < 16; ++k) { run += o[k]; acc[k] += run; }
            uint4* qp = (uint4*)q;
            qp[0] = make_uint4(acc[0],  acc[1],  acc[2],  acc[3]);
            qp[1] = make_uint4(acc[4],  acc[5],  acc[6],  acc[7]);
            qp[2] = make_uint4(acc[8],  acc[9],  acc[10], acc[11]);
            qp[3] = make_uint4(acc[12], acc[13], acc[14], acc[15]);
            p += sstep;
            q = (uint32_t*)((uint8_t*)q + dstep);
        }
    }
}

// 简单的进程级 workspace（benchmark 进程生命周期用，够用）
struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
void* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) {
        if (b.p) cudaFree(b.p);
        CUDA_CHECK(cudaMalloc(&b.p, need));
        b.sz = need;
    }
    return b.p;
}

inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

} // anonymous namespace

void gpu_integral(const uint8_t* d_src, size_t srcStep, uint32_t* d_dst, size_t dstStep,
                  int width, int height, cudaStream_t stream)
{
    if (width <= 0 || height <= 0) return;

    // ---- r51 融合单遍路径（6N 字节）：W%16==0、W<=4096、16B 对齐、H 够高时分派 ----
    {
        const bool a16 = ((width & 15) == 0) && ((srcStep & 15) == 0) && ((dstStep & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(d_src) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(d_dst) & 15) == 0);
        const bool nofuse = env_nofuse();   // 仅供 probe 做 A/B 对照
        if (a16 && width <= 4096 && height >= FUSED_MIN_H && !nofuse) {
            const int TH = fused_th(height);
            const int nChunks = idiv_up(height, TH);
            uint32_t* pre = (nChunks > 1)
                ? (uint32_t*)ws_grow((size_t)nChunks * width * sizeof(uint32_t)) : nullptr;
            if (nChunks > 1) {
                const int groups = (width >> 4);
                const int total = nChunks * groups;
                const int blk = 256, grd = idiv_up(total, blk);
                integral_chunk_sum_kernel<<<grd, blk, 0, stream>>>(d_src, srcStep, pre, width, height, TH);
                const int scanG = idiv_up(width, 256);
                integral_chunk_scan_kernel<<<scanG, 256, 0, stream>>>(pre, width, nChunks);
            }
            // 线程数按宽度自适应（每线程 16 列，向上取整到 warp）：W=640→64、1280→96、
            // 1920→128→128、3840→256。旧版固定 256 线程，小图上大半个块空转。
            const int tw = (width + 15) >> 4;
            const int fth = ((tw + 31) >> 5) << 5;
            integral_fused_kernel<<<nChunks, fth, 0, stream>>>(
                d_src, srcStep, d_dst, dstStep, pre, width, height, TH);
            return;
        }
    }

    // Pass 1：每 block 8 行（每 warp 一行），grid=(H+7)/8（spike 001：sm_120 上 2.8x）
    {
        const int warpsPerBlock = 8;           // 256 线程
        const int blocks = (height + 7) / 8;
        integral_rows_kernel<<<blocks, warpsPerBlock * 32, 0, stream>>>(
            d_src, srcStep, d_dst, dstStep, width, height);
    }

    const int nChunks = idiv_up(height, CHUNK_H);
    const bool vecOK = ((dstStep & 15) == 0) && ((srcStep & 3) == 0) &&
                       ((reinterpret_cast<uintptr_t>(d_dst) & 15) == 0) &&
                       ((width & 3) == 0);

    uint32_t* carry = nullptr;
    if (nChunks > 1)
        carry = (uint32_t*)ws_grow((size_t)nChunks * width * sizeof(uint32_t));

    if (vecOK && width >= 512) {
        // r50：单遍列扫描（每线程一列走完 H 行）。W>=512 时 512+ 个线程足以占住 SM、
        // 又省掉「回加 carry」的 8N 字节流量，实测 4K 1.243→0.708ms。
        const int colBlocks = idiv_up(width, 256);
        integral_cols_1col_kernel<<<colBlocks, 256, 0, stream>>>(
            d_dst, dstStep, d_dst, dstStep, width, height);
    } else if (width >= 512 && (dstStep & 3) == 0 &&
                   ((reinterpret_cast<uintptr_t>(d_dst) & 3) == 0)) {
            // 宽图但非 16B 对齐 / W%4!=0：单遍路径不要求 16B，4B 对齐即可（原先会掉进标量回退）
        const int colBlocks = idiv_up(width, 256);
        integral_cols_1col_kernel<<<colBlocks, 256, 0, stream>>>(
            d_dst, dstStep, d_dst, dstStep, width, height);
    } else if (vecOK) {
        const int colBlocks = idiv_up(width, CT_THREADS * 4);
        if (nChunks == 1) {
            // 单 chunk：无需 carry 矩阵，但仍必须做列方向扫描（此前此处提前 return 是 bug）
            dim3 g(colBlocks), t(CT_THREADS);
            integral_cols_kernel<<<g, t, 0, stream>>>(
                d_dst, dstStep, d_dst, dstStep, nullptr, width, height, 1);
        } else {
            dim3 g(colBlocks, nChunks), t(CT_THREADS);
            integral_cols_kernel<<<g, t, 0, stream>>>(
                d_dst, dstStep, d_dst, dstStep, carry, width, height, nChunks);
            dim3 g2(colBlocks, nChunks - 1);
            integral_carry_add_kernel<<<g2, t, 0, stream>>>(
                d_dst, dstStep, carry, width, height);
        }
    } else {
        const int colBlocks = idiv_up(width, 256);
        if (nChunks == 1) {
            dim3 g(colBlocks), t(256);
            integral_cols_scalar_kernel<<<g, t, 0, stream>>>(
                d_dst, dstStep, d_dst, dstStep, nullptr, width, height, 1);
        } else {
            dim3 g(colBlocks, nChunks), t(256);
            integral_cols_scalar_kernel<<<g, t, 0, stream>>>(
                d_dst, dstStep, d_dst, dstStep, carry, width, height, nChunks);
            dim3 g2(colBlocks, nChunks - 1);
            integral_carry_add_scalar_kernel<<<g2, t, 0, stream>>>(
                d_dst, dstStep, carry, width, height);
        }
    }
}

// ================= CV_32F 输入积分图 =================
// 行遍：warp 分段扫描（每 lane 16 个 float 段内顺序累加 + shuffle 扫描，与 u8 版同构），
// 列遍：每线程一列分 chunk 顺序累加 + carry 回加。
// 顺序均与 OpenCV 逐元素一致；大图累加和超出 2^24 后 float 舍入与分块粒度相关，
// 语义为「与 OpenCV 同构的 float 积分图」，允许相对舍入差（smoke 用相对容差校验）。
namespace {

__global__ void integral_rows_f32_kernel(const float* __restrict__ src, size_t sstep,
                                         float* __restrict__ dst, size_t dstep,
                                         int W, int H)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int nwarps = (gridDim.x * blockDim.x) >> 5;
    constexpr int SEG = 16;

    for (int y = warp; y < H; y += nwarps) {
        const float* srow = src + (size_t)y * (sstep >> 2);
        float* drow = (float*)((uint8_t*)dst + (size_t)y * dstep);
        float carry = 0.f;

        for (int x0 = 0; x0 < W; x0 += 32 * SEG) {
            const int my0 = x0 + lane * SEG;
            float vals[SEG];
            float partial = 0.f;
#pragma unroll
            for (int k = 0; k < SEG; ++k) {
                const int x = my0 + k;
                float v = 0.f;
                if (x < W) v = srow[x];
                vals[k] = v;
                partial += v;
            }
            float inc = partial;
#pragma unroll
            for (int d = 1; d < 32; d <<= 1) {
                float t = __shfl_up_sync(0xffffffffu, inc, d);
                if (lane >= d) inc += t;
            }
            const float excl = inc - partial + carry;
            carry += __shfl_sync(0xffffffffu, inc, 31);

            float run = excl;
#pragma unroll
            for (int k = 0; k < SEG; ++k) {
                run += vals[k];
                const int x = my0 + k;
                if (x < W) drow[x] = run;
            }
        }
    }
}

__global__ void integral_cols_f32_kernel(const float* __restrict__ in, size_t istep,
                                         float* __restrict__ out, size_t ostep,
                                         float* __restrict__ carry,
                                         int W, int H, int nChunks)
{
    const int chunk = blockIdx.y;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;

    const size_t rows = istep >> 2, orows = ostep >> 2;
    const float* ip = in + (size_t)r0 * rows + c;
    float*       op = out + (size_t)r0 * orows + c;
    float car = 0.f;
    for (int r = r0; r < r1; ++r) {
        car += *ip;
        *op = car;
        ip += rows; op += orows;
    }
    if (nChunks > 1) carry[(size_t)chunk * W + c] = car;
}

__global__ void integral_carry_add_f32_kernel(float* __restrict__ out, size_t ostep,
                                              const float* __restrict__ carry,
                                              int W, int H)
{
    const int chunk = blockIdx.y + 1;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c  = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= W) return;

    float exc = 0.f;
    for (int j = 0; j < chunk; ++j) exc += carry[(size_t)j * W + c];

    const size_t orows = ostep >> 2;
    float* op = out + (size_t)r0 * orows + c;
    for (int r = r0; r < r1; ++r) {
        *op += exc;
        op += orows;
    }
}

} // anonymous namespace

void gpu_integral_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                      int width, int height, cudaStream_t stream)
{
    if (width <= 0 || height <= 0) return;
    {
        const int warpsPerBlock = 8;
        const int blocks = 12;
        integral_rows_f32_kernel<<<blocks, warpsPerBlock * 32, 0, stream>>>(
            d_src, srcStep, d_dst, dstStep, width, height);
    }
    const int nChunks = idiv_up(height, CHUNK_H);
    float* carry = nullptr;
    if (nChunks > 1)
        carry = (float*)ws_grow((size_t)nChunks * width * sizeof(float));
    const int colBlocks = idiv_up(width, 256);
    if (nChunks == 1) {
        dim3 g(colBlocks), t(256);
        integral_cols_f32_kernel<<<g, t, 0, stream>>>(
            d_dst, dstStep, d_dst, dstStep, nullptr, width, height, 1);
    } else {
        dim3 g(colBlocks, nChunks), t(256);
        integral_cols_f32_kernel<<<g, t, 0, stream>>>(
            d_dst, dstStep, d_dst, dstStep, carry, width, height, nChunks);
        dim3 g2(colBlocks, nChunks - 1), t2(256);
        integral_carry_add_f32_kernel<<<g2, t2, 0, stream>>>(d_dst, dstStep, carry, width, height);
    }
}
