// 均值滤波：分离滑动窗口，O(1)/像素（与核尺寸无关），总访存 ~6B/像素
// 行遍：u8 -> u16 行和（reflect101 边界在取数时镜像）；列遍：u16 -> u8 归一化
// 边界语义与 cv::cuda::boxFilter 默认 BORDER_DEFAULT(=REFLECT_101) 一致
#include "ops.h"
#include "halo_stash.cuh"
#include <algorithm>
#include <cstdlib>

typedef unsigned int uint;   // Windows 下 CUDA 头不保证提供 uint

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

// ---------------- 大核行遍重写（2026-11-24 r64）----------------
// 归因（docs/probes/probe_meanbig.cu asym，交错 min，四尺寸）：
//   4K    (1,1) 1.506 | (31,1) 4.893（行遍 +3.39）| (1,31) 1.861（列遍 +0.36）| (31,31) 4.461
//   1080p (1,1) 0.482 | (31,1) 1.619（行遍 +1.14）| (1,31) 0.819（列遍 +0.34）
//   ⇒ 行遍是唯一大头；列遍增量已贴同尺寸搬运地板（u16->u8 @4K 0.261ms）。
// 病灶：旧 mean_rows_kernel 的 k>15 分支每 block 只做**一行的一个 256 列段**，且用 512 元
//   Hillis-Steele smem 扫描 = 9 轮 × 2 次 __syncthreads = 18 次屏障；4K 有 32400 个块。
//   4K 行遍增量 3.39ms / 3N(24.9MB) = 7.3 GB/s，u8->u16 搬运地板是 94.6 GB/s ⇒ 13 倍差距。
// 定案 mean_rows_slide_kernel<C>（probe_mr31 check 42 用例 vs CPU int64 精确窗口和，逐值全 0）：
//   ① **一行一 block**（4K 2160 块）→ 每 block 只 1 次 __syncthreads（18 → 1）；
//   ② 整段 [x0-half, x0+bw+half) 一次搬进 smem，REFLECT_101 在载入时解决；
//   ③ 每线程 C=16 个连续像素，初始窗口 O(k) 直加一次，之后每像素「+新 -旧」滑动 O(1)；
//   ④ **写回必须 16B 向量化**：16 个 u16 标量存时 lane 相隔 32B，一条 store 的 32 lane
//      落在 32 个 sector、每个只写 2/32 字节（16 倍写放大）—— 与 r50 integral pass1 同源的坑。
//      打包成 uint4 后 4K k=31 行遍 1.13 → 0.48ms（v1 → v3，2.33x）。
// 实测（probe_mr31 ab，纯内核）：4K k=31 3.54→0.48（7.36x）、1080p 1.00→0.143（6.99x）、
//   720p 0.499→0.073、640 0.180→0.036；k=15 同档 2.5~2.8x。输出与旧内核逐位一致。
template <int C>
__global__ void mean_rows_slide_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                       uint16_t* __restrict__ tmp, size_t tstep, int W, int k)
{
    extern __shared__ uint8_t sm[];
    const int half = k >> 1;
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nld  = min(bw, W - x0) + 2 * half;
    const uint8_t* srow = src + (size_t)y * sstep;

    for (int i = tid; i < nld; i += blockDim.x)
        sm[i] = srow[mirror101<true>(x0 - half + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;
    const int n  = min(C, W - gx);
    uint16_t* trow = tmp + (size_t)y * (tstep >> 1);

    uint32_t s = 0;
#pragma unroll 4
    for (int j = 0; j < k; ++j) s += sm[li + j];

    uint16_t p[C];
    for (int i = 0; i < n; ++i) {
        p[i] = (uint16_t)s;
        if (i + 1 < n) s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
    }
    if (n == C && ((tstep & 15) == 0)) {
        uint4* q = (uint4*)(trow + gx);          // gx = tid*C，C=16 ⇒ 16B 对齐
#pragma unroll
        for (int i = 0; i < C / 8; ++i)
            q[i] = make_uint4(p[8*i] | ((uint32_t)p[8*i+1] << 16),
                              p[8*i+2] | ((uint32_t)p[8*i+3] << 16),
                              p[8*i+4] | ((uint32_t)p[8*i+5] << 16),
                              p[8*i+6] | ((uint32_t)p[8*i+7] << 16));
    } else {
        for (int i = 0; i < n; ++i) trow[gx + i] = p[i];
    }
}

// 大核行遍分派：一行一 block，th 按 W 选（每线程 16 像素）
void mean_rows_slide(const uint8_t* d_src, size_t srcStep, uint16_t* tmp, size_t tstep,
                     int width, int height, int k, cudaStream_t stream)
{
    constexpr int C = 16;
    int th = (width + C - 1) / C;
    th = (th + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    const int bw  = th * C;
    const int nbx = (width + bw - 1) / bw;
    const size_t smem = (size_t)(bw + 2 * (k / 2));
    dim3 g(height, nbx);
    mean_rows_slide_kernel<C><<<g, th, smem, stream>>>(d_src, srcStep, tmp, tstep, width, k);
}

// ---------------- 大核列遍重写（2026-11-24 r65）----------------
// 归因（probe_mcol2/mcol3 sweep，交错 min，四尺寸；u16->u8 地板 640 52.9 / 720p 75.0 /
//   1080p 84.4 / 4K 95.7 GB/s）：
//   4K k=15 old_blocked 3.11ms(8.0 GB/s) | old_slide 0.92(27.0) | 新 0.325(76.7)
//   4K k=31 old_blocked 6.20(4.0)        | old_slide 0.93(26.6) | 新 0.373(66.8)
//   ⇒ 两个旧内核各有硬伤，新内核把两者一起替掉：
//     ① mean_cols_blocked_kernel 每输出行 O(k) 直加 ⇒ k=15 时 tmp 重读 15 倍（45B/px）；
//     ② mean_cols_kernel 每线程一列沿 y 走 ⇒ 4K 只有 W=3840 线程（6 SM 上 30 warp），
//        并行度严重不足，折 27 GB/s。
// 定案 mean_cols_vec_kernel<C=4>：**每线程 C 个连续列 + 行分块 rpt + 窗口滑动**。
//   关键认识（PROGRESS 要求先量清的「列方向 16 行不连续」问题）：**这根本不是问题**——
//   一行之内的 C 个列在内存里是连续的，问题只出在「一列 C 个连续行」那种拆法上。
//   本内核沿 y 分块、块内逐行推进，同一 warp 的 32 个 lane 在同一行上覆盖 32*4 = 128 个
//   连续列 ⇒ 读 256B / 写 128B 都是整段合并，与行遍的 16B 甜点结构同源。
//   **不要**改成「一列 C 行」（那才是 16 倍 sector 放大的坑，r50 integral pass1 同源）。
// 初始窗口 O(k) 直加一次被 rpt 行摊薄；之后每行「+新 −旧」O(1)。
// 为什么**不做**寄存器 ring（每行只读一次，流量 3B/px 而非 5B/px）：
//   probe_mcol3 实测 ring 在 k=31 反而慢 —— v[31][4] = 124 个寄存器溢出，4K 0.373→0.87+；
//   k=15 的 ringC4 也只与 vecC4 打平（4K 0.3256 vs 0.3246）。第二次读的是 half 行之前
//   访问过的同一物理行，已被 L1/L2 吸收，实测不构成代价。**ring 路线已否决，勿重开。**
// rpt 扫参（probe_mcol3 rpt，9 档 × 3 轮轮转交错 min，排序稳定）：
//   k=15 → rpt=8 全尺寸最优（4K 0.3246 / 1080p 0.0979 / 720p 0.0594 / 640 +6% 于最优）；
//   k=31 → rpt=12 全尺寸最优（4K 0.3727 / 1080p 0.1126 / 720p 0.0635 / 640 最优）。
//   ⇒ 一条规则 `rpt = (k <= 15) ? 8 : 12`，不做分尺寸档（避免 r49 那种噪声拟合）。
// 验证：probe_mcol2 check 132 用例 + probe_mcol3 check 48 用例，新/旧/CPU int64 三方逐值全 0
//   （含 16x1/16x2/17x3/8x1/1x1/1x7/12x5 极扁、4096x64、1281x721 非 4 倍数宽、四档 bench 尺寸）。
// 约束：width % 4 == 0 且 dst 4B 对齐（其余回退旧路径，结果一致）。
template <int C>
__device__ __forceinline__ void mcol_ld4(const uint16_t* __restrict__ p, uint32_t v[C])
{
    const uint2 w = *(const uint2*)p;      // C=4：8B = 4 个 u16
    v[0] = w.x & 0xffffu; v[1] = w.x >> 16;
    v[2] = w.y & 0xffffu; v[3] = w.y >> 16;
}

__device__ __forceinline__ uint mcol_pack4(uint32_t a, uint32_t b, uint32_t c, uint32_t d, float inv)
{
    uint w = 0;
    const uint32_t s[4] = {a, b, c, d};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int q = __float2int_rn((float)s[j] * inv);
        q = max(0, min(255, q));
        w |= ((uint32_t)q) << (8 * j);
    }
    return w;
}

template <int C>
__global__ void mean_cols_vec_kernel(const uint16_t* __restrict__ tmp, size_t tstep,
                                     uint8_t* __restrict__ dst, size_t dstep,
                                     int W, int H, int kh, float inv_area, int rpt)
{
    const int c = (blockIdx.x * blockDim.x + threadIdx.x) * C;
    if (c >= W) return;
    const int y0 = blockIdx.y * rpt;
    if (y0 >= H) return;
    const int y1 = min(H, y0 + rpt);
    const int half = kh >> 1;
    const size_t trows = tstep >> 1;

    if (c + C > W) {                       // 尾列（W%4!=0 时不会进本内核，这里只是兜底）
        for (int y = y0; y < y1; ++y) {
            for (int i = c; i < W; ++i) {
                uint32_t s = 0;
                for (int j = -half; j <= half; ++j)
                    s += tmp[(size_t)mirror101<false>(y + j, H) * trows + i];
                const int v = __float2int_rn((float)s * inv_area);
                dst[(size_t)y * dstep + i] = (uint8_t)max(0, min(255, v));
            }
        }
        return;
    }

    uint32_t acc[C];
#pragma unroll
    for (int i = 0; i < C; ++i) acc[i] = 0;
    {
        uint32_t v[C];
        for (int j = -half; j <= half; ++j) {
            mcol_ld4<C>(tmp + (size_t)mirror101<false>(y0 + j, H) * trows + c, v);
#pragma unroll
            for (int i = 0; i < C; ++i) acc[i] += v[i];
        }
    }

    for (int y = y0; y < y1; ++y) {
        *(uint*)(dst + (size_t)y * dstep + c) =
            mcol_pack4(acc[0], acc[1], acc[2], acc[3], inv_area);
        if (y + 1 < y1) {
            uint32_t vn[C], vo[C];
            mcol_ld4<C>(tmp + (size_t)mirror101<false>(y + 1 + half, H) * trows + c, vn);
            mcol_ld4<C>(tmp + (size_t)mirror101<false>(y - half, H) * trows + c, vo);
#pragma unroll
            for (int i = 0; i < C; ++i) acc[i] += vn[i] - vo[i];
        }
    }
}

// 列遍分派：对齐且 W%4==0、kh<=31 走 vec 版；其余回退旧内核（行为与 r64 完全一致）。
// 为什么把 vec 卡在 kh<=31（probe_mcol3 bigk 实测）：kh>=33 时初始 O(k) 直加需要更大的 rpt
// 才摊得平（k=255/4K 要 rpt=256 才 0.486ms，而 rpt=12 是 1.556ms > old_slide 1.053ms），
// 且 k=33 的 4K 最优点在 r12 与 r32 之间来回跳（离群、不稳）。bench 只测 k=3/15/31，
// 不为非 bench 路径引入未扫稳的 rpt 规则 —— kh>31 保留 old_slide（无回归）。
void mean_cols_dispatch(const uint16_t* tmp, size_t tstep, uint8_t* d_dst, size_t dstStep,
                        int width, int height, int kw, int kh, float inv_area,
                        cudaStream_t stream)
{
    constexpr int C = 4;
    const bool vec_ok = (width % 4 == 0) && width >= 4 && kh <= 31 &&
                        (dstStep % 4 == 0) && ((uintptr_t)d_dst % 4 == 0);
    if (vec_ok) {
        const int rpt = (kh <= 15) ? 8 : 12;    // probe_mcol3 rpt 扫参：两档全尺寸最优
        const int ng = width / C;
        int th = (ng + 31) / 32 * 32;
        if (th > MF_THREADS) th = MF_THREADS;
        if (th < 32) th = 32;
        dim3 g((ng + th - 1) / th, (height + rpt - 1) / rpt);
        mean_cols_vec_kernel<C><<<g, th, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                     width, height, kh, inv_area, rpt);
        return;
    }
    if (kw <= 15 && kh <= 15) {
        // 小核列遍走行分块（全场 W*ceil(H/rpt) 线程）；4K 实测 rpt=64 最优，小图 rpt=16 更好
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2((width + MF_THREADS - 1) / MF_THREADS, (height + rpt - 1) / rpt), t2(MF_THREADS);
        mean_cols_blocked_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                        width, height, kh, inv_area, rpt);
    } else {
        // 大核：每列一线程滑动窗口 O(1)/像素（行分块会让每线程重复 O(k) 直加，不划算）
        dim3 g2((width + MF_THREADS - 1) / MF_THREADS), t2(MF_THREADS);
        mean_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh, inv_area);
    }
}

// ---------------- k=3 单遍融合（2026-10-27） ----------------
// 旧 mean3_direct_kernel 是 32x8 smem tile：每像素 9 次 smem 字节读、32400 个小 block，
// 4K 1.10ms 只有 7.5 GB/s；更要命的是它的 tile halo 从 global 读，原地调用（benchmark 口径）
// 存在跨 block 竞态 —— probe_m3a 实测 4K 69 万像素错、maxdiff 48（smoke 只测 out-of-place 所以没抓到）。
// 本内核：16 像素/线程、uint4 取数、水平 3 邻域用 16-bit lane 在寄存器里做
// （__byte_perm 把 4 字节展开成 2 个 u16 lane 字，__funnelshift 取左右邻，__vadd2 加），
// 垂直 3 行滚动；原地路径先融合写 workspace（16 对齐紧排）再整图 uint4 拷回，与 morph k=3 同构。
// 语义：窗口 clamp 到图内 = BORDER_REFLECT_101 对 3x3 中心锚点等价（halo 行/列镜像后与本像素相同）。
// 输出 = (sum*7282+32768)>>16，sum ≤ 9*255 = 2295 时与 round(sum/9) 逐值等价（probe 全量验证）。

// 4 字节 → 2 个 u16 lane（lane0=b0|lane1=b1<<16 形式）；两次 PRMT
__device__ __forceinline__ void m3_expand(uint w, uint& lo, uint& hi)
{
    lo = __byte_perm(w, 0, 0x4140);   // bytes [b0, 0, b1, 0]
    hi = __byte_perm(w, 0, 0x4342);   // bytes [b2, 0, b3, 0]
}

// 两个 u16 lane（≤2295）→ 两个字节打包在低 16 位
__device__ __forceinline__ uint m3_div9_pack(uint v)
{
    const uint a = v & 0xffffu, b = v >> 16;
    const uint qa = (a * 7282u + 32768u) >> 16;
    const uint qb = (b * 7282u + 32768u) >> 16;
    return qa | (qb << 8);
}

template <int TH, bool STASH>
__global__ void mean3_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  uint8_t* __restrict__ dst, size_t dstep, int W, int H,
                                  const uint8_t* __restrict__ stash, size_t stashStep)
{
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int j1   = STASH ? (y1 / TH) : 0;
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : ((W - 16) & ~15);   // 非活跃线程也给 16B 对齐安全地址参与 shuffle

    uint hp[8], hc[8], hn[8];

    // 一行 16 字节的水平 3 邻域和（u16 lane，≤765）
    auto hrow = [&](int ry, uint o[8]) {
        const int sl = STASH ? halo_stash::slot<1>(ry, y0, blockIdx.y, y1, j1, H, gridDim.y) : -1;
        const uint8_t* rp = (sl < 0) ? (src + (size_t)ry * sstep)
                                     : (stash + (size_t)sl * stashStep);
        const uint4 w = *(const uint4*)(rp + gxl);
        uint e[8];
        m3_expand(w.x, e[0], e[1]);
        m3_expand(w.y, e[2], e[3]);
        m3_expand(w.z, e[4], e[5]);
        m3_expand(w.w, e[6], e[7]);
        uint prv = __shfl_up_sync(0xffffffffu, e[7] >> 16, 1);      // 左邻线程最后一像素
        uint nxt = __shfl_down_sync(0xffffffffu, e[0] & 0xffffu, 1); // 右邻线程第一像素
        // 边界必须按 BORDER_REFLECT_101 镜像（求和不能用 clamp：3 邻域 clamp≠镜像，
        // min/max 能糊弄过去是因为 min(a,a,b)==min(a,b)，加法不行）
        if (lane == 0) prv = (gxl == 0) ? (e[0] >> 16) : (uint)rp[gxl - 1];
        if (lane == 31 || gxl + 16 >= W)
            nxt = (gxl + 16 >= W) ? (e[7] & 0xffffu) : (uint)rp[gxl + 16];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint P = (k == 0) ? ((e[0] << 16) | prv) : __funnelshift_r(e[k - 1], e[k], 16);
            const uint N = (k == 7) ? ((nxt << 16) | (e[7] >> 16)) : __funnelshift_r(e[k], e[k + 1], 16);
            o[k] = __vadd2(__vadd2(P, e[k]), N);
        }
    };

    // 垂直方向同样要 reflect101：y0==0 时上一行是 row 1（不是 row 0）；末行下一行是 H-2
    hrow((y0 == 0) ? min(1, H - 1) : (y0 - 1), hp);
    hrow(y0, hc);
    // 原地：预载读的是本块自己的行，快的 warp 可能已经写完行 y0 → 同步一次
    if (STASH) __syncthreads();
    for (int y = y0; y < y1; ++y) {
        const int yn = (y + 1 < H) ? (y + 1) : max(0, H - 2);
        hrow(yn, hn);
        if (act) {
            uint4 o;
            uint p[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) p[k] = m3_div9_pack(__vadd2(__vadd2(hp[k], hc[k]), hn[k]));
            o.x = p[0] | (p[1] << 16);
            o.y = p[2] | (p[3] << 16);
            o.z = p[4] | (p[5] << 16);
            o.w = p[6] | (p[7] << 16);
            *(uint4*)(dst + (size_t)y * dstep + gx) = o;
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) { hp[i] = hc[i]; hc[i] = hn[i]; }
        // 原地：块内 warp 会互相超车（warp 间无顺序保证），快 warp 在迭代 y+1 已经写掉行
        // y+1，慢 warp 在迭代 y 预取的正是它；每行一次 __syncthreads 把 skew 压到 0。
        if (STASH) __syncthreads();
    }
}

// 整图 uint4 拷贝（原地路径用）
__global__ void m3_copy16_kernel(const uint8_t* __restrict__ s, size_t sstep,
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

template <bool STASH>
void mean3_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                 int width, int height, int TH, const uint8_t* stash, size_t stashStep,
                 cudaStream_t stream)
{
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    const int gxx = (width + 16 * th - 1) / (16 * th);
    const int th3 = (TH <= 8) ? 8 : (TH <= 16 ? 16 : (TH <= 32 ? 32 : 64));
    dim3 g(gxx, (height + th3 - 1) / th3);
    if (th3 == 8)       mean3_fuse_kernel<8,  STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else if (th3 == 16) mean3_fuse_kernel<16, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else if (th3 == 32) mean3_fuse_kernel<32, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
    else                mean3_fuse_kernel<64, STASH><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, stash, stashStep);
}

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
void* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) { if (b.p) cudaFree(b.p); CUDA_CHECK(cudaMalloc(&b.p, need)); b.sz = need; }
    return b.p;
}
inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

// 原地路径选择：大图（H>=1000）走 stash（只暂存 halo 行），小图仍走整图拷回 —— barrier 的
// 固定代价在小图上反超收益（gaussian r46 probe 实测 640x480 0.047→0.060）。
// CK_FORCE_STASH=1 强制开启：给 probe 覆盖 TH/nby/尾块 chunk 的角落组合用。
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

} // anonymous namespace

void gpu_mean(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
              int width, int height, int kw, int kh, cudaStream_t stream)
{
    const float inv_area = 1.0f / (float)(kw * kh);
    if (kw == 3 && kh == 3) {
        // k=3 单遍融合：out-of-place 单内核；in-place 融合写 workspace 再整图拷回（4B/px）。
        // 仅接受 W%16==0 且 16B 对齐；其它宽度/对齐回退两遍分离路径（结果一致）。
        const bool inplace = ((const void*)d_src == (const void*)d_dst);
        const bool fuse3 = (width >= 16 && width <= 4096 && (width % 16 == 0) &&
                            (srcStep % 16 == 0) && (dstStep % 16 == 0) &&
                            ((uintptr_t)d_src % 16 == 0) && ((uintptr_t)d_dst % 16 == 0));
        if (fuse3) {
            const int TH = th3_override(height <= 1200 ? 16 : 64);
            if (inplace) {
                if (stash_enabled(height)) {
                    // 大图原地：只暂存 halo 行（4K TH=64 → 71 行 = 273KB vs 整图 8.3MB），
                    // 融合内核读 stash 代替整图拷回
                    size_t sstep = ((size_t)srcStep + 15) & ~(size_t)15;
                    const int rows = halo_stash::rows_needed<1>(height, TH);
                    uint8_t* buf = (uint8_t*)ws_grow((size_t)rows * sstep + 1024);
                    const uint8_t* st = halo_stash::prepare<1>(d_src, srcStep, width, height, TH,
                                                               sstep, buf, stream);
                    mean3_fused<true>(d_src, srcStep, d_dst, dstStep, width, height, TH, st, sstep, stream);
                } else {
                    const int tstep = (width + 15) & ~15;
                    uint8_t* tmp = (uint8_t*)ws_grow((size_t)tstep * height + 1024);
                    mean3_fused<false>(d_src, srcStep, tmp, tstep, width, height, TH, nullptr, 0, stream);
                    int th = ((width + 15) / 16 + 31) / 32 * 32;
                    if (th > 256) th = 256;
                    dim3 gc((width + 16 * th - 1) / (16 * th), height);
                    m3_copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
                }
            } else {
                mean3_fused<false>(d_src, srcStep, d_dst, dstStep, width, height, TH, nullptr, 0, stream);
            }
            return;
        }
    }
    const size_t tstep = (size_t)width * 2;
    uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);

    if (kw >= 15 && kw <= 33) {
        // 大核行遍（2026-11-24 r64）：一行一 block + 每线程 16 像素滑动，见 mean_rows_slide 注释。
        // 门槛 k>=15 的由来：probe_mr31 ab 里 k=15 与 k=31 都测到 2.5~7.4x（k=15 的旧路径虽走
        // O(k) 直加，但 18 次屏障的结构开销与块数问题一样存在）；k<15 未测，保持旧路径。
        mean_rows_slide(d_src, srcStep, tmp, tstep, width, height, kw, stream);
    } else {
        dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
        mean_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);
    }

    // 列遍（2026-11-24 r65 重写）：对齐且 W%4==0 走 vec 版（4 列/线程 + 行分块 + 滑动），
    // 实测 4K k=15 3.11→0.325ms（9.6x）、k=31 6.20→0.373（16.6x）；否则回退旧路径。
    mean_cols_dispatch(tmp, tstep, d_dst, dstStep, width, height, kw, kh, inv_area, stream);
}

// ================= CV_32F 输入均值滤波 =================
// 分离两遍（与 u8 版同构，元素改 float）：行遍 tile 前缀和，列遍滑动窗口。
// 窗口和 ≤255×255×255 ≈ 1.66e7 < 2^24，float 精确；输出 = 和×(1/area)，
// 与 OpenCV boxFilter(CV_32FC1) 的 float 累加同序，逐位一致（smoke 容差 1e-2 兜底）。
namespace {

__global__ void mean32_rows_kernel(const float* __restrict__ src, size_t sstep,
                                   float* __restrict__ tmp, size_t tstep,
                                   int W, int k)
{
    __shared__ float sp[MF_PAD];
    const int y   = blockIdx.y;
    const int x0  = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid = threadIdx.x;
    const int N   = MF_THREADS + k - 1;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < N; i += MF_THREADS)
        sp[i] = srow[mirror101<true>(x0 - half + i, W)];
    for (int i = N + tid; i < MF_PAD; i += MF_THREADS) sp[i] = 0.f;
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    float sum;
    if (k <= 15) {
        sum = 0.f;
#pragma unroll 4
        for (int j = 0; j < k; ++j) sum += sp[tid + j];
    } else {
        // 大核：Hillis-Steele inclusive scan（每线程负责相邻 2 元），P[i]=Σ_{j<=i} sp[j]
        for (int d = 1; d < MF_PAD; d <<= 1) {
            const int i0 = 2 * tid, i1 = 2 * tid + 1;
            float a = 0.f, b = 0.f;
            if (i0 >= d) a = sp[i0 - d];
            if (i1 >= d) b = sp[i1 - d];
            __syncthreads();
            if (i0 >= d) sp[i0] += a;
            if (i1 >= d) sp[i1] += b;
            __syncthreads();
        }
        sum = sp[tid + k - 1] - (tid ? sp[tid - 1] : 0.f);
    }
    tmp[(size_t)y * (tstep >> 2) + x] = sum;
}

__global__ void mean32_cols_blocked_kernel(const float* __restrict__ tmp, size_t tstep,
                                           float* __restrict__ dst, size_t dstep,
                                           int W, int H, int k, float inv_area, int rpt)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k >> 1;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol = tmp + x;

    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mirror101<false>(j - half + y0, H) * rows];
    for (int r = y0; r < y1; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        if (r + 1 < y1) {
            acc += scol[(size_t)mirror101<false>(r + 1 + half, H) * rows]
                 - scol[(size_t)mirror101<false>(r - half, H) * rows];
        }
    }
}

__global__ void mean32_cols_kernel(const float* __restrict__ tmp, size_t tstep,
                                   float* __restrict__ dst, size_t dstep,
                                   int W, int H, int k, float inv_area)
{
    const int x = blockIdx.x * MF_THREADS + threadIdx.x;
    if (x >= W) return;
    const int half = k >> 1;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol = tmp + x;

    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc += scol[(size_t)mirror101<false>(j - half, H) * rows];
    for (int r = 0; r < H; ++r) {
        dst[(size_t)r * orows + x] = acc * inv_area;
        acc += scol[(size_t)mirror101<false>(r + 1 + half, H) * rows]
             - scol[(size_t)mirror101<false>(r - half, H) * rows];
    }
}

// ---------------- f32 均值 k=3 单遍融合（2026-11-02，r45）----------------
// 病灶同 gaussian-f32：两遍分离 16B/px（读 4 + 写 tmp 4 + 读 tmp 4 + 写 4），
// 4K 1.9528ms = 68 GB/s，纯拷贝天花板 93 GB/s → 砍流量到 8B/px。
// 结构：4 px/线程（float4），寄存器滚动 3 行水平 3-tap 和；水平 halo 用 __shfl。
// 算术次序与旧两遍路径逐位同序（行遍 sum=0 起顺序加 3 项；列遍 acc 初值顺序加 3 行，
// 之后 acc += 新行 - 旧行）→ 与旧输出逐位一致。
// 约束：out-of-place、W%4==0、8<=W<=4096、16B 对齐；其余回退两遍路径。
__device__ __forceinline__ int m32_mirror(int r, int H)
{
    int rr = r;
    if (rr < 0)  rr = -rr;
    if (rr >= H) rr = 2 * H - 2 - rr;
    return max(0, min(H - 1, rr));
}

template <int TH>
__global__ void mean32_fuse3_kernel(const float* __restrict__ src, size_t sstep,
                                    float* __restrict__ dst, size_t dstep,
                                    int W, int H, float inv_area)
{
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 4;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 4) & ~3);
    const int rows = (int)(sstep >> 2), orows = (int)(dstep >> 2);

    auto hrow = [&](int ry, float o[4]) {
        const float* srow = src + (size_t)ry * rows;
        const float4 v = *(const float4*)(srow + xa);
        float l1 = __shfl_up_sync(0xffffffffu, v.w, 1);
        float r1 = __shfl_down_sync(0xffffffffu, v.x, 1);
        if (lane == 0 || xa == 0) {
            l1 = (xa == 0) ? v.y : srow[xa - 1];             // p(-1)=p(1)
        }
        if (lane == 31 || xa + 4 >= W) {
            r1 = (xa + 4 >= W) ? v.z : srow[xa + 4];         // p(W)=p(W-2)
        }
        const float pb[6] = { l1, v.x, v.y, v.z, v.w, r1 };
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float s = 0.f;
#pragma unroll
            for (int q = 0; q < 3; ++q) s += pb[i + q];
            o[i] = s;
        }
    };

    float h[3][4];
#pragma unroll
    for (int i = 0; i < 3; ++i) hrow(m32_mirror(y0 - 1 + i, H), h[i]);

    float acc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float s = 0.f;
        s += h[0][j]; s += h[1][j]; s += h[2][j];
        acc[j] = s;
    }

    for (int y = y0; y < y1; ++y) {
        if (act) {
            float4 o;
            o.x = acc[0] * inv_area; o.y = acc[1] * inv_area;
            o.z = acc[2] * inv_area; o.w = acc[3] * inv_area;
            *(float4*)(dst + (size_t)y * orows + x0) = o;
        }
        if (y + 1 < y1) {
            float hn[4];
            hrow(m32_mirror(y + 2, H), hn);
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[j] += hn[j] - h[0][j];
#pragma unroll
            for (int j = 0; j < 4; ++j) { h[0][j] = h[1][j]; h[1][j] = h[2][j]; h[2][j] = hn[j]; }
        }
    }
}

} // anonymous namespace

void gpu_mean_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                  int width, int height, int kw, int kh, cudaStream_t stream)
{
    const float inv_area = 1.0f / (float)(kw * kh);

    // k=3 单遍融合（2026-11-02）：两遍 16B/px → 8B/px。
    const bool inplace32 = ((const void*)d_src == (const void*)d_dst);
    const bool fuse32 = (kw == 3 && kh == 3 && !inplace32 && width >= 8 && width <= 4096 &&
                         (width % 4 == 0) && (srcStep % 16 == 0) && (dstStep % 16 == 0) &&
                         ((uintptr_t)d_src % 16 == 0) && ((uintptr_t)d_dst % 16 == 0));
    if (fuse32) {
        const int TH = (height >= 1600) ? 64 : (height > 800 ? 32 : (height > 400 ? 16 : 8));
        int th = ((width + 3) / 4 + 31) / 32 * 32;
        if (th > 256) th = 256;
        if (th < 32)  th = 32;
        dim3 g(idiv_up(width, 4 * th), idiv_up(height, TH));
        switch (TH) {
        case 8:  mean32_fuse3_kernel<8> <<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, inv_area); break;
        case 16: mean32_fuse3_kernel<16><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, inv_area); break;
        case 32: mean32_fuse3_kernel<32><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, inv_area); break;
        default: mean32_fuse3_kernel<64><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height, inv_area); break;
        }
        return;
    }

    const size_t tstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(tstep * height + 1024);

    dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
    mean32_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);

    if (kh <= 15) {
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2(idiv_up(width, MF_THREADS), idiv_up(height, rpt)), t2(MF_THREADS);
        mean32_cols_blocked_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                          width, height, kh, inv_area, rpt);
    } else {
        dim3 g2(idiv_up(width, MF_THREADS)), t2(MF_THREADS);
        mean32_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh, inv_area);
    }
}
