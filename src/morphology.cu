// 腐蚀/膨胀（二值与灰度统一，min/max 语义）
// 核心优化：van Herk/Gil-Werman 分离算法 —— 任意核尺寸 O(1)/像素（OpenCV CUDA 是每轴 O(k) 的暴力窗）
// 水平/垂直各一遍。边界语义与 OpenCV morphologyDefaultBorderValue 一致：
//   腐蚀(最小值滤波)边界视为 255（恒等元），膨胀(最大值滤波)边界视为 0 —— 等价于窗口 clamp 到图像内。
#include "ops.h"
#include "halo_stash.cuh"
#include <algorithm>
#include <cstdlib>

typedef unsigned int uint;   // Windows 下 CUDA 头不保证提供 uint

namespace {

constexpr int VHG_B = 128;   // 段长（=block 线程数）

struct MaxOp { __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return max(a, b); } static __device__ uint8_t ident() { return 0; } };
struct MinOp { __device__ uint8_t operator()(uint8_t a, uint8_t b) const { return min(a, b); } static __device__ uint8_t ident() { return 255; } };

// 一行/一列 = 一条线。block 处理一条线上的一个输出段(tile)。
// 需要左邻段的后缀扫描和右邻段的前缀扫描，故 smem 里加载 3 个段。
// 小核（k<=7）直接窗口快路径：VHG 的三段加载+双扫描对小核是纯开销
template <typename Op>
__global__ void direct_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int k)
{
    __shared__ uint8_t tile[256 + 7];           // 256 + MAX_SMALL_K - 1
    const int y   = blockIdx.y;
    const int x0  = blockIdx.x * 256;
    const int tid = threadIdx.x;
    const Op op;
    const uint8_t* srow = src + (size_t)y * sstep;

    // 越界填恒等元（腐蚀=255/膨胀=0），与 morphologyDefaultBorderValue 语义一致
    for (int i = tid; i < 256 + k - 1; i += 256) {
        const int gx = x0 - k / 2 + i;
        tile[i] = (gx >= 0 && gx < W) ? srow[gx] : Op::ident();
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    uint8_t v = tile[tid];
#pragma unroll 4
    for (int j = 1; j < k; ++j) v = op(v, tile[tid + j]);
    dst[(size_t)y * dstep + x] = v;
}

// 小核列遍：每线程一列的一段行区间（与高斯列遍同款分块，避免全场仅 W 线程）
// k<=3 用 3 元直加展开（无 ring，寄存器直读）；k>3 用 k 槽 ring
template <typename Op>
__global__ void direct_cols_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H, int k, int rpt)
{
    const int x   = blockIdx.x * 256 + threadIdx.x;
    const int y0  = blockIdx.y * rpt;
    const int y1  = min(H, y0 + rpt);
    const Op op;
    if (x >= W) return;

    auto fetch = [&](int r) -> uint8_t {
        return (r >= 0 && r < H) ? src[(size_t)r * sstep + x] : Op::ident();
    };
    if (k <= 3) {
        // k=3 快路径：三寄存器滚动，无取模、无数组下标
        uint8_t a = fetch(y0 - 1), b = fetch(y0);
        for (int r = y0; r < y1; ++r) {
            const uint8_t c = fetch(r + 1);
            uint8_t v = op(a, b);
            v = op(v, c);
            dst[(size_t)r * dstep + x] = v;
            a = b; b = c;
        }
        return;
    }
    // 滑动窗口 ring（k 槽）
    uint8_t ringv[7];
    for (int j = 0; j < k; ++j) ringv[j] = fetch(j - k / 2 + y0);
    int oldest = 0;

    for (int r = y0; r < y1; ++r) {
        uint8_t v = ringv[oldest];
        for (int j = 1; j < k; ++j) v = op(v, ringv[(oldest + j) % k]);
        dst[(size_t)r * dstep + x] = v;
        if (r + 1 < y1) {
            ringv[oldest] = fetch(r + 1 + k / 2);
            if (++oldest == k) oldest = 0;
        }
    }
}

// ---------------- k=3 单遍融合（2026-10-27） ----------------
// 4K 实测：两遍（rows 直加 + cols 行分块）1.46ms 里大头不是算法，而是两遍共 4 次全图流量
// 与行遍 32400 个小 block 的调度开销；本内核单遍完成，16 像素/线程、uint4 取数、
// 水平 3 邻域用 vminu4/vmaxu4 + funnelshift 在寄存器里做，垂直用滚动寄存器，
// 实测 0.196ms（4K，84 GB/s ≈ 纯拷贝天花板的 92%），vs 两遍 7.4x。
// 语义：边界按「窗口 clamp 到图内」，与 morphologyDefaultBorderValue（erode 255/dilate 0）等价。
struct VMinU8 { __device__ __forceinline__ static uint op4(uint a, uint b) { return __vminu4(a, b); }
                __device__ __forceinline__ static uint8_t op1(uint8_t a, uint8_t b) { return min(a, b); } };
struct VMaxU8 { __device__ __forceinline__ static uint op4(uint a, uint b) { return __vmaxu4(a, b); }
                __device__ __forceinline__ static uint8_t op1(uint8_t a, uint8_t b) { return max(a, b); } };

template <typename Op> struct VOpOf;
template <> struct VOpOf<MinOp> { using V = VMinU8; };
template <> struct VOpOf<MaxOp> { using V = VMaxU8; };

template <typename V, int TH, bool STASH>
__global__ void morph3_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H, const uint8_t* __restrict__ stash, size_t stashStep)
{
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int j1   = STASH ? (y1 / TH) : 0;
    const int lane = tid & 31;
    const bool act = (gx < W);
    // 非活跃线程仍要参与 shuffle，给一个 16B 对齐的安全取数地址
    // （W 非 16 倍数时向下取整对齐，否则 uint4 取数会 misaligned）；
    // 活跃的尾块线程照旧读自己的 gx（16 对齐，pitch 保证不越界），这样邻居拿到的是真数据
    const int gxl  = act ? gx : ((W - 16) & ~15);


    // 原地（dst==src）：内核保持最瘦，H>=1000 走前置 stash 只暂存 halo 行（r47），
    // 小图仍由 host 端「融合写 workspace + 整图拷回」，因为 barrier 固定代价在小图上反超收益。
    // （历史：在内核里做 halo + warp 边界预存会让寄存器 31→52、4K 慢 2x，不要重走。）
    uint hp[4], hc[4], hn[4];

    // 一行内的水平 3 邻域 min/max（4 个 32 位字打包 16 字节）
    auto hrow = [&](int ry, uint o[4]) {
        const int sl = STASH ? halo_stash::slot<1>(ry, y0, blockIdx.y, y1, j1, H, gridDim.y) : -1;
        const uint8_t* rp = (sl < 0) ? (src + (size_t)ry * sstep)
                                     : (stash + (size_t)sl * stashStep);
        const uint4 w = *(const uint4*)(rp + gxl);
        uint prv = __shfl_up_sync(0xffffffffu, w.w, 1);
        uint nxt = __shfl_down_sync(0xffffffffu, w.x, 1);
        // 跨 warp 与图像左右边界不能靠 shuffle
        if (lane == 0) prv = (gxl < 4) ? (w.x << 24) : *(const uint*)(rp + gxl - 4);
        if (lane == 31 || gxl + 16 >= W)
            nxt = (gxl + 16 >= W) ? (w.w >> 24) : *(const uint*)(rp + gxl + 16);
        const uint ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint lo = (i == 0) ? prv : ws[i - 1];
            const uint hi = (i == 3) ? nxt : ws[i + 1];
            o[i] = V::op4(ws[i], V::op4(__funnelshift_r(lo, ws[i], 24),
                                        __funnelshift_l(ws[i], hi, 24)));
        }
    };

    hrow(max(0, y0 - 1), hp);
    hrow(y0, hc);
    // 原地：预载读的是本块自己的行，快的 warp 可能已经写完行 y0 → 同步一次
    if (STASH) __syncthreads();
    for (int y = y0; y < y1; ++y) {
        hrow(min(H - 1, y + 1), hn);
        if (act) {
            uint o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = V::op4(hp[i], V::op4(hc[i], hn[i]));
            *(uint4*)(dst + (size_t)y * dstep + gx) = make_uint4(o[0], o[1], o[2], o[3]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) { hp[i] = hc[i]; hc[i] = hn[i]; }
        // 原地：块内 warp 会互相超车（warp 间无顺序保证）—— 快 warp 在迭代 y+1 已经写掉行
        // y+1，慢 warp 在迭代 y 预取的正是它。每行一次 __syncthreads 把 skew 压到 0。
        if (STASH) __syncthreads();
    }
}

// ---------------- 大核（k>7）线性走查内核（r53 新增） ----------------
// 旧 VHG 的病灶（probe_mcol asym 实测，4K、k=15）：行遍 12.42ms、列遍 12.37ms、(1,1) 基线 0.98ms，
// 且 k=31 与 k=15 几乎同速 ⇒ 耗时与窗口长度无关，全是固定结构开销：每个 block 只产出 128 像素，
// 却要 load 3 段 + 做 42 次 __syncthreads 的 Hillis-Steele 双扫描（4K = 6.5 万 block × 42 屏障）。
// 现在改成「一线程一条线，在 smem 里顺序走查」的 van Herk 分解：
//   U[c] = op(块首 .. c)、D[c] = op(c .. 块尾)，块长 = k；输出窗口 [i, i+k-1] = op(D[i], U[i+k-1])。
// 每线程只维护一个 running 值（正走算 U、反走算 D），O(1)/像素、零屏障链。
// 边界：载入时 clamp 到图内（min/max 的恒等元与 clamp 等价，与旧路径语义一致）。
constexpr int ML_TC  = 64;   // 行遍每 block 的输出列数
constexpr int ML_RB  = 128;  // 行遍每 block 的输出行数（= 线程数，一线程一行）
constexpr int ML_RB2 = 64;   // 列遍每 block 的输出行数
constexpr int ML_KMAX = 33;  // 走查内核支持的核上限（k<=33 ⇒ 半径 <=16，smem 装得下）

// 行遍：srow/tile + sU 各 ML_RB 行（STR 奇数行距错 bank）+ sout 输出暂存（保证写回合并）
template <typename Op>
__global__ void morph_rows_line_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                       uint8_t* __restrict__ dst, size_t dstep,
                                       int W, int H, int k, int STR, int TC)
{
    extern __shared__ uint8_t ml[];
    const int R    = k >> 1;
    const int SCW  = TC + k - 1;               // 载入列跨度
    const int r0   = blockIdx.y * ML_RB;
    const int c0   = blockIdx.x * TC;
    const int tid  = threadIdx.x;
    const int nt   = blockDim.x;
    const int nr   = min(ML_RB, H - r0);
    const int nc   = min(TC, W - c0);

    uint8_t* srow = ml;                        // ML_RB * STR
    uint8_t* sU   = srow + (size_t)ML_RB * STR;
    uint8_t* sout = sU + (size_t)ML_RB * STR;  // ML_RB * TC

    for (int p = 0; p < nr; ++p) {
        const uint8_t* sp = src + (size_t)(r0 + p) * sstep;
        uint8_t* tp = srow + (size_t)p * STR;
        for (int c = tid; c < SCW; c += nt) {
            const int gx = min(max(c0 - R + c, 0), W - 1);
            tp[c] = sp[gx];
        }
    }
    __syncthreads();

    if (tid < nr) {                            // 一线程一行：正走 U、反走 D
        const Op op;
        const uint8_t* tp = srow + (size_t)tid * STR;
        uint8_t* up = sU + (size_t)tid * STR;
        uint8_t* out = sout + (size_t)tid * TC;
        int m = 0;
        uint8_t u = Op::ident();
        for (int c = 0; c < SCW; ++c) {
            const uint8_t v = tp[c];
            u = (m == 0) ? v : op(u, v);
            if (++m == k) m = 0;
            up[c] = u;
        }
        uint8_t d = Op::ident();
        m = (SCW - 1) % k;
        for (int c = SCW - 1; c >= 0; --c) {
            const uint8_t v = tp[c];
            d = (m == k - 1) ? v : op(v, d);
            if (m == 0) m = k - 1; else --m;
            if (c < nc) out[c] = op(d, up[c + k - 1]);
        }
    }
    __syncthreads();

    for (int p = 0; p < nr; ++p) {             // 合并写回
        const uint8_t* o = sout + (size_t)p * TC;
        uint8_t* dp = dst + (size_t)(r0 + p) * dstep + c0;
        for (int c = tid; c < nc; c += nt) dp[c] = o[c];
    }
}

// 列遍：sv/sU 各 LR 行 × TC 列；线程 t 负责一列，正走填 U、反走算 D 直接写 dst
// （同一行的 32 个 lane 落在连续 32 字节 → 写天然合并，不需要输出暂存）
template <typename Op>
__global__ void morph_cols_line_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                       uint8_t* __restrict__ dst, size_t dstep,
                                       int W, int H, int k)
{
    extern __shared__ uint8_t ml[];
    const int R   = k >> 1;
    const int TC  = ML_TC * 2;                 // 每 block 列数（= 线程数，一线程一列）
    const int LR  = ML_RB2 + 2 * R;            // 载入行数
    const int x0  = blockIdx.x * TC;
    const int y0  = blockIdx.y * ML_RB2;
    const int t   = threadIdx.x;
    const int x   = x0 + t;

    uint8_t* sv = ml;                          // LR * TC
    uint8_t* sU = sv + (size_t)LR * TC;        // LR * TC

    const bool act = (x < W);
    const int  xc  = act ? x : (W - 1);
    for (int p = 0; p < LR; ++p) {
        const int gy = min(max(y0 - R + p, 0), H - 1);
        sv[(size_t)p * TC + t] = src[(size_t)gy * sstep + xc];
    }
    __syncthreads();
    if (!act) return;

    const int nr = min(ML_RB2, H - y0);
    const Op op;
    int m = 0;
    uint8_t u = Op::ident();
    for (int p = 0; p < LR; ++p) {
        const uint8_t v = sv[(size_t)p * TC + t];
        u = (m == 0) ? v : op(u, v);
        if (++m == k) m = 0;
        sU[(size_t)p * TC + t] = u;
    }
    uint8_t d = Op::ident();
    m = (LR - 1) % k;
    uint8_t* dp = dst + (size_t)y0 * dstep + x;
    for (int p = LR - 1; p >= 0; --p) {
        const uint8_t v = sv[(size_t)p * TC + t];
        d = (m == k - 1) ? v : op(v, d);
        if (m == 0) m = k - 1; else --m;
        if (p < nr) dp[(size_t)p * dstep] = op(d, sU[(size_t)(p + k - 1) * TC + t]);
    }
}

template <typename Op>
void morph_line_passes(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                       int width, int height, int kw, int kh, uint8_t* tmp, size_t tmpStep,
                       cudaStream_t stream)
{
    const int STR  = (ML_TC + kw - 1) | 1;      // 奇数行距：相邻行错 bank
    const size_t smH = (size_t)ML_RB * STR * 2 + (size_t)ML_RB * ML_TC;
    dim3 gh((width + ML_TC - 1) / ML_TC, (height + ML_RB - 1) / ML_RB);
    morph_rows_line_kernel<Op><<<gh, ML_RB, smH, stream>>>(
        d_src, srcStep, tmp, tmpStep, width, height, kw, STR, ML_TC);

    const int TC2  = ML_TC * 2;
    const int LR2  = ML_RB2 + 2 * (kh >> 1);
    const size_t smV = (size_t)LR2 * TC2 * 2;
    dim3 gv((width + TC2 - 1) / TC2, (height + ML_RB2 - 1) / ML_RB2);
    morph_cols_line_kernel<Op><<<gv, TC2, smV, stream>>>(
        tmp, tmpStep, d_dst, dstStep, width, height, kh);
}

// 整图 uint4 拷贝（原地路径用：融合先写 workspace，再拷回 dst）
__global__ void copy16_kernel(const uint8_t* __restrict__ s, size_t sstep,
                              uint8_t* __restrict__ d, size_t dstep, int W)
{
    const int x = (blockIdx.x * blockDim.x + threadIdx.x) * 16;
    const int y = blockIdx.y;
    const uint8_t* sp = s + (size_t)y * sstep;
    uint8_t* dp = d + (size_t)y * dstep;
    if (x + 16 <= W) {
        *(uint4*)(dp + x) = *(const uint4*)(sp + x);
    } else if (x < W) {
        for (int i = x; i < W; ++i) dp[i] = sp[i];
    }
}

template <typename Op, bool STASH>
void morph3_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int TH, const uint8_t* stash, size_t stashStep,
                  cudaStream_t stream)
{
    using V = typename VOpOf<Op>::V;
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    const int gxx = (width + 16 * th - 1) / (16 * th);
    const int th3 = (TH <= 8) ? 8 : (TH <= 16 ? 16 : (TH <= 32 ? 32 : 64));
    dim3 g(gxx, (height + th3 - 1) / th3);
    if (th3 == 8)       morph3_fuse_kernel<V, 8,  STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else if (th3 == 16) morph3_fuse_kernel<V, 16, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else if (th3 == 32) morph3_fuse_kernel<V, 32, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else                morph3_fuse_kernel<V, 64, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
}

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

// 原地路径选择：大图（H>=1000）走 stash，小图仍走整图拷回（barrier 固定代价在小图上反超收益）。
// CK_FORCE_STASH=1 强制开启，给 probe 覆盖 TH/nby/尾块 chunk 的角落组合用。
bool stash_enabled(int height)
{
    static const bool forced = [] {
        const char* s = getenv("CK_FORCE_STASH");
        return s && s[0] == '1';
    }();
    return forced || height >= 1000;
}

// TH 覆盖（仅 probe 用：CK_TH3=8/16/32/64 扫 chunk 划分对 stash 填充量与并行度的影响）
int th3_override(int def)
{
    const char* s = getenv("CK_TH3");
    if (!s) return def;
    const int v = atoi(s);
    return (v == 8 || v == 16 || v == 32 || v == 64) ? v : def;
}

// probe 用：CK_MORPH_LEGACY=1 停用 r53 的走查内核，回落到 VHG（A/B 对照）。
// 每次调用都读 env（不缓存），因为 probe 要在同一进程里交替切换两条路径；
// getenv 约 0.1us，而本函数处理的图最小也在 40us 量级，开销可忽略。
inline bool morph_legacy_only()
{
    const char* s = getenv("CK_MORPH_LEGACY");
    return s && s[0] == '1';
}

template <typename Op>
void morph(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
           int width, int height, int kw, int kh, cudaStream_t stream)
{
    if (width <= 0 || height <= 0) return;

    // k=3 单遍融合路径（2026-10-27）：out-of-place 单内核；in-place 先存 2 行/块边界再融合。
    // 要求 16B 对齐（基址与 pitch），否则回退两遍路径（结果一致，只是慢）。
    const bool inplace = ((const void*)d_src == (const void*)d_dst);
    // 原地融合要求整行在一个 block 内（水平边界只可能是图像边界），W > 4096 时回退两遍路径
    const bool fuse3 = (kw == 3 && kh == 3 && width >= 16 && width <= 4096 && (width % 16 == 0) &&
                        (srcStep % 16 == 0) && (dstStep % 16 == 0) &&
                        ((uintptr_t)d_src % 16 == 0) && ((uintptr_t)d_dst % 16 == 0));
    if (fuse3) {
        // 实测（r49 probe_th3）：morph 现有分档已在各尺寸取到最优（640 TH8、720 TH8、1080 TH16、
        // 4K TH64），不做改动；th3_override 只给 probe 扫 TH=32/64 用。
        const int TH = th3_override(height <= 800 ? 8 : (height <= 1200 ? 16 : 64));
        if (inplace) {
            if (stash_enabled(height)) {
                // 大图原地（r47）：前置 stash 只暂存 halo 行，省掉整图拷回
                size_t sstep = ((size_t)srcStep + 15) & ~(size_t)15;
                const int rows = halo_stash::rows_needed<1>(height, TH);
                uint8_t* buf = ws_grow((size_t)rows * sstep + 1024);
                const uint8_t* st = halo_stash::prepare<1>(d_src, srcStep, width, height, TH,
                                                           sstep, buf, stream);
                morph3_fused<Op, true>(d_src, srcStep, d_dst, dstStep, width, height, TH, st, sstep, stream);
            } else {
                // 小图原地：先融合写到 workspace（每行按 16 对齐的紧排），再整图 uint4 拷回 dst。
                const int tstep = (width + 15) & ~15;
                uint8_t* tmp = ws_grow((size_t)tstep * height + 1024);
                morph3_fused<Op, false>(d_src, srcStep, tmp, tstep, width, height, TH, nullptr, 0, stream);
                int th = ((width + 15) / 16 + 31) / 32 * 32;
                if (th > 256) th = 256;
                dim3 gc((width + 16 * th - 1) / (16 * th), height);
                copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
            }
        } else {
            morph3_fused<Op, false>(d_src, srcStep, d_dst, dstStep, width, height, TH, nullptr, 0, stream);
        }
        return;
    }

    // 注意：临时缓冲必须用与 d_dst 相同的 step 布局（用 tight width 会造成
    // vhg_cols_kernel 以 dstStep 跨行读写时越界/读串行——之前 smoke 中 gray 小核差异即由此而来）
    // 用静态 workspace：cudaMallocPitch/cudaFree 每次调用会同步设备，是计时大坑
    size_t tmpStep = (size_t)width;
    uint8_t* tmp = ws_grow(tmpStep * height + 1024);

    const bool smallK = kw <= 7 && kh <= 7;
    if (smallK) {
        // 4K 实测（probe_k3）：rows->cols 两遍直加窗口 1.19ms，比 VHG 双趟（1.95ms）快 39%，
        // 也比此前 rows 直加+ring 慢路径快 —— k<=7 全部走此路径
        dim3 g1(idiv_up(width, 256), height), t1(256);
        direct_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tmpStep, width, kw);
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2(idiv_up(width, 256), idiv_up(height, rpt)), t2(256);
        direct_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, tmpStep, d_dst, dstStep, width, height, kh, rpt);
    } else if (!morph_legacy_only() && kw <= ML_KMAX && kh <= ML_KMAX) {
        // r53：大核走「一线程一条线 + smem 顺序走查」，替掉 42 屏障的 VHG（见 probe_mcol）
        morph_line_passes<Op>(d_src, srcStep, d_dst, dstStep, width, height, kw, kh,
                              tmp, tmpStep, stream);
    } else {
        // 水平：每块 128 线程处理一行的一个段
        dim3 g1(idiv_up(width, VHG_B), height), t1(VHG_B);
        vhg_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tmpStep, width, kw);
        // 垂直：blockIdx.y = 列号
        dim3 g2(idiv_up(height, VHG_B), width), t2(VHG_B);
        vhg_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, tmpStep, d_dst, dstStep, height, kh);
    }
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

// ================= CV_32F 输入腐蚀/膨胀 =================
// f32 上 min/max 与 OpenCV 的 f32 形态学逐位一致；边界恒等元（腐蚀 +inf、膨胀 -inf）
// 等价于窗口 clamp 到图像内（morphologyDefaultBorderValue 语义）。
// 路线：分离两遍直加窗口（rows O(kW) + cols 行分块 O(kH)），k≤31 覆盖常用场景。
namespace {

struct MinOpF { __device__ float operator()(float a, float b) const { return fminf(a, b); } static __device__ float ident() { return __int_as_float(0x7f800000); } };
struct MaxOpF { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } static __device__ float ident() { return __int_as_float(0xff800000); } };

template <typename Op>
__global__ void morph32_rows_kernel(const float* __restrict__ src, size_t sstep,
                                    float* __restrict__ dst, size_t dstep,
                                    int W, int k)
{
    __shared__ float tile[512];                 // 256 + k - 1，k ≤ 257
    const int y   = blockIdx.y;
    const int x0  = blockIdx.x * 256;
    const int tid = threadIdx.x;
    const Op op;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < 256 + k - 1; i += 256) {
        const int gx = x0 - k / 2 + i;
        tile[i] = (gx >= 0 && gx < W) ? srow[gx] : Op::ident();
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    float v = tile[tid];
#pragma unroll 4
    for (int j = 1; j < k; ++j) v = op(v, tile[tid + j]);
    dst[(size_t)y * (dstep >> 2) + x] = v;
}

template <typename Op>
__global__ void morph32_cols_kernel(const float* __restrict__ src, size_t sstep,
                                    float* __restrict__ dst, size_t dstep,
                                    int W, int H, int k, int rpt)
{
    const int x   = blockIdx.x * 256 + threadIdx.x;
    const int y0  = blockIdx.y * rpt;
    const int y1  = min(H, y0 + rpt);
    const Op op;
    if (x >= W) return;
    const size_t rows = sstep >> 2, orows = dstep >> 2;

    auto fetch = [&](int r) -> float {
        return (r >= 0 && r < H) ? src[(size_t)r * rows + x] : Op::ident();
    };
    if (k <= 3) {
        // k≤3：三寄存器滚动
        float a = fetch(y0 - 1), b = fetch(y0);
        for (int r = y0; r < y1; ++r) {
            const float c = fetch(r + 1);
            float v = op(a, b);
            v = op(v, c);
            dst[(size_t)r * orows + x] = v;
            a = b; b = c;
        }
        return;
    }
    // k>3：k 槽 ring（动态下标走 local memory，正确性优先）
    float ringv[255];
    for (int j = 0; j < k; ++j) ringv[j] = fetch(j - k / 2 + y0);
    int oldest = 0;
    for (int r = y0; r < y1; ++r) {
        float v = ringv[oldest];
        for (int j = 1; j < k; ++j) v = op(v, ringv[(oldest + j) % k]);
        dst[(size_t)r * orows + x] = v;
        if (r + 1 < y1) {
            ringv[oldest] = fetch(r + 1 + k / 2);
            if (++oldest == k) oldest = 0;
        }
    }
}

// ---------------- 大核（k>7）滑窗寄存器内核（r54 新增） ----------------
// 病灶（probe_mf32 asym，4K）：列遍是主因 —— (1,31) 23.28ms vs (31,1) 3.84ms vs (1,1) 2.26ms：
// morph32_cols_kernel 的 float ringv[255] 动态下标落 local memory（每像素 O(k) 次 local 读 +
// 一列一写 32 sector 放大），行遍直加窗口则被 smem 逐元素读放大。
// u8 侧 r53 的「smem 整线走查」在 f32 上装不下（两张 float 线表 128 列 × 96 行 ≈ 98KB > 48KB），
// 改用滑窗寄存器：窗口 K 值常驻寄存器，滑动 = 移位 + 补新值，O(1)/像素、零 local、零屏障。
// rows：一线程一行（RB=128），block 协作载入 [x0-R, x0+TB-1+R] 列段（线性映射、合并读），
//       输出写 smem 暂存（SOUTP=TB+1 错 bank）后合并写回；
// cols：一线程一列、TB 行一段，global 读天然合并（warp 同行邻列），输出攒寄存器、段尾合并写回。
// 边界 clamp + 恒等元，与旧路径语义一致（probe_mf32 check 68 用例新旧逐位一致）。
template <typename Op, int K, int TB, int RB>
__global__ void morph32_rows_win_kernel(const float* __restrict__ src, size_t sstep,
                                        float* __restrict__ dst, size_t dstep,
                                        int W, int H)
{
    constexpr int RAD   = K >> 1;
    constexpr int SCW   = TB + K - 1;       // 载入列跨度
    constexpr int SCWP  = SCW | 1;          // 奇数行距错 bank
    constexpr int SOUTP = TB + 1;           // 输出暂存行距错 bank
    __shared__ float s_in[RB * SCWP];
    __shared__ float s_out[RB * SOUTP];

    const int x0  = blockIdx.x * TB;
    const int y0  = blockIdx.y * RB;
    const int tid = threadIdx.x;
    const Op op;

    // 协作载入（线性映射：相邻线程相邻列；越界列/行填恒等元 = clamp 语义）
    for (int idx = tid; idx < RB * SCW; idx += RB) {
        const int r  = idx / SCW;
        const int c  = idx - r * SCW;
        const int gy = y0 + r;
        const int gx = x0 - RAD + c;
        s_in[(size_t)r * SCWP + c] = (gy < H && gx >= 0 && gx < W)
                                     ? src[(size_t)gy * (sstep >> 2) + gx] : Op::ident();
    }
    __syncthreads();

    if (y0 + tid < H) {                     // 一线程一行：窗口寄存器滑动
        const float* rp = s_in + (size_t)tid * SCWP;
        float* out      = s_out + (size_t)tid * SOUTP;
        float w[K];
#pragma unroll
        for (int j = 0; j < K; ++j) w[j] = rp[j];
#pragma unroll
        for (int i = 0; i < TB; ++i) {
            float v = w[0];
#pragma unroll
            for (int j = 1; j < K; ++j) v = op(v, w[j]);
            out[i] = v;
            if (i + 1 < TB) {
#pragma unroll
                for (int j = 0; j < K - 1; ++j) w[j] = w[j + 1];
                w[K - 1] = rp[i + K];       // rp 已 clamp，直接续读
            }
        }
    }
    __syncthreads();

    const int nc = min(TB, W - x0);
    for (int idx = tid; idx < RB * nc; idx += RB) {   // 线性映射合并写回
        const int r  = idx / nc;
        const int c  = idx - r * nc;
        const int gy = y0 + r;
        if (gy < H) dst[(size_t)gy * (dstep >> 2) + x0 + c] = s_out[(size_t)r * SOUTP + c];
    }
}

// cols：一线程一列，TB 行一段；窗口寄存器滑动；输出攒寄存器、段尾合并写回
template <typename Op, int K, int TB>
__global__ void morph32_cols_win_kernel(const float* __restrict__ src, size_t sstep,
                                        float* __restrict__ dst, size_t dstep,
                                        int W, int H)
{
    constexpr int RAD = K >> 1;
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * TB;
    if (x >= W) return;
    const Op op;
    const size_t rows = sstep >> 2, orows = dstep >> 2;

    float w[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int gy = y0 - RAD + j;
        w[j] = (gy >= 0 && gy < H) ? src[(size_t)gy * rows + x] : Op::ident();
    }
    float out[TB];
#pragma unroll
    for (int i = 0; i < TB; ++i) {
        float v = w[0];
#pragma unroll
        for (int j = 1; j < K; ++j) v = op(v, w[j]);
        out[i] = v;
        if (i + 1 < TB) {
#pragma unroll
            for (int j = 0; j < K - 1; ++j) w[j] = w[j + 1];
            const int gy = y0 + i + RAD + 1;      // 窗口下移一格的新下端（clamp）
            w[K - 1] = (gy < H) ? src[(size_t)gy * rows + x] : Op::ident();
        }
    }
#pragma unroll
    for (int i = 0; i < TB; ++i) {
        const int y = y0 + i;
        if (y < H) dst[(size_t)y * orows + x] = out[i];
    }
}

// 滑窗内核覆盖奇数 k ∈ [9,33]（与 u8 走查内核的 k≤33 对齐）；返回 false = 该轴走旧路径
template <typename Op>
bool morph32_rows_win(int k, const float* src, size_t sstep, float* tmp, size_t tstep,
                      int W, int H, cudaStream_t stream)
{
    constexpr int TB = 8, RB = 128;
    switch (k) {
    case 9:  { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 9,  TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 11: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 11, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 13: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 13, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 15: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 15, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 17: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 17, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 19: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 19, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 21: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 21, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 23: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 23, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 25: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 25, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 27: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 27, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 29: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 29, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 31: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 31, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    case 33: { dim3 g((W + TB - 1) / TB, (H + RB - 1) / RB);
               morph32_rows_win_kernel<Op, 33, TB, RB><<<g, RB, 0, stream>>>(src, sstep, tmp, tstep, W, H); return true; }
    default: return false;
    }
}

template <typename Op>
bool morph32_cols_win(int k, const float* tmp, size_t tstep, float* dst, size_t dstep,
                      int W, int H, cudaStream_t stream)
{
    constexpr int TB = 16;
    switch (k) {
    case 9:  { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 9,  TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 11: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 11, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 13: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 13, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 15: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 15, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 17: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 17, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 19: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 19, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 21: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 21, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 23: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 23, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 25: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 25, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 27: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 27, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 29: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 29, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 31: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 31, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    case 33: { dim3 g((W + 255) / 256, (H + TB - 1) / TB);
               morph32_cols_win_kernel<Op, 33, TB><<<g, 256, 0, stream>>>(tmp, tstep, dst, dstep, W, H); return true; }
    default: return false;
    }
}

template <typename Op>
void morph32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
             int width, int height, int kw, int kh, cudaStream_t stream)
{
    // 关键：列遍必须读「行遍的输出」而不是 dst 原地读写 —— 行分块后，上方 chunk 的
    // 输出行会被下方 chunk 当输入读，block 间无同步 → 竞态（u8 版列遍读 tmp 同理）
    const size_t tstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(tstep * height + 1024);
    // r54：大核逐轴分派到滑窗寄存器内核（k∈[9,33] 奇数），其余仍走旧直加/ring 路径
    if (!morph32_rows_win<Op>(kw, d_src, srcStep, tmp, tstep, width, height, stream)) {
        dim3 g1((width + 255) / 256, height), t1(256);
        morph32_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);
    }
    if (!morph32_cols_win<Op>(kh, tmp, tstep, d_dst, dstStep, width, height, stream)) {
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2((width + 255) / 256, (height + rpt - 1) / rpt), t2(256);
        morph32_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                       width, height, kh, rpt);
    }
}

} // namespace

void gpu_erode_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                   int width, int height, int kw, int kh, cudaStream_t stream)
{
    morph32<MinOpF>(d_src, srcStep, d_dst, dstStep, width, height, kw, kh, stream);
}

void gpu_dilate_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                    int width, int height, int kw, int kh, cudaStream_t stream)
{
    morph32<MaxOpF>(d_src, srcStep, d_dst, dstStep, width, height, kw, kh, stream);
}
