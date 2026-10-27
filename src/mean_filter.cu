// 均值滤波：分离滑动窗口，O(1)/像素（与核尺寸无关），总访存 ~6B/像素
// 行遍：u8 -> u16 行和（reflect101 边界在取数时镜像）；列遍：u16 -> u8 归一化
// 边界语义与 cv::cuda::boxFilter 默认 BORDER_DEFAULT(=REFLECT_101) 一致
#include "ops.h"
#include <algorithm>

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

template <int TH>
__global__ void mean3_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  uint8_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : ((W - 16) & ~15);   // 非活跃线程也给 16B 对齐安全地址参与 shuffle

    uint hp[8], hc[8], hn[8];

    // 一行 16 字节的水平 3 邻域和（u16 lane，≤765）
    auto hrow = [&](const uint8_t* rp, uint o[8]) {
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
    hrow(src + (size_t)((y0 == 0) ? min(1, H - 1) : (y0 - 1)) * sstep, hp);
    hrow(src + (size_t)y0 * sstep, hc);
    for (int y = y0; y < y1; ++y) {
        const int yn = (y + 1 < H) ? (y + 1) : max(0, H - 2);
        hrow(src + (size_t)yn * sstep, hn);
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

void mean3_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                 int width, int height, int TH, cudaStream_t stream)
{
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    const int gxx = (width + 16 * th - 1) / (16 * th);
    const int th3 = (TH <= 8) ? 8 : (TH <= 16 ? 16 : (TH <= 32 ? 32 : 64));
    dim3 g(gxx, (height + th3 - 1) / th3);
    if (th3 == 8)       mean3_fuse_kernel<8><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
    else if (th3 == 16) mean3_fuse_kernel<16><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
    else if (th3 == 32) mean3_fuse_kernel<32><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
    else                mean3_fuse_kernel<64><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
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
    const float inv_area = 1.0f / (float)(kw * kh);
    if (kw == 3 && kh == 3) {
        // k=3 单遍融合：out-of-place 单内核；in-place 融合写 workspace 再整图拷回（4B/px）。
        // 仅接受 W%16==0 且 16B 对齐；其它宽度/对齐回退两遍分离路径（结果一致）。
        const bool inplace = ((const void*)d_src == (const void*)d_dst);
        const bool fuse3 = (width >= 16 && width <= 4096 && (width % 16 == 0) &&
                            (srcStep % 16 == 0) && (dstStep % 16 == 0) &&
                            ((uintptr_t)d_src % 16 == 0) && ((uintptr_t)d_dst % 16 == 0));
        if (fuse3) {
            const int TH = height <= 1200 ? 16 : 64;
            if (inplace) {
                const int tstep = (width + 15) & ~15;
                uint8_t* tmp = (uint8_t*)ws_grow((size_t)tstep * height + 1024);
                mean3_fused(d_src, srcStep, tmp, tstep, width, height, TH, stream);
                int th = ((width + 15) / 16 + 31) / 32 * 32;
                if (th > 256) th = 256;
                dim3 gc((width + 16 * th - 1) / (16 * th), height);
                m3_copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
            } else {
                mean3_fused(d_src, srcStep, d_dst, dstStep, width, height, TH, stream);
            }
            return;
        }
    }
    const size_t tstep = (size_t)width * 2;
    uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);

    dim3 g1(idiv_up(width, MF_THREADS), height), t1(MF_THREADS);
    mean_rows_kernel<<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);

    if (kw <= 15 && kh <= 15) {
        // 小核列遍走行分块（全场 W*ceil(H/rpt) 线程）；4K 实测 rpt=64 最优，小图 rpt=16 更好
        const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
        dim3 g2(idiv_up(width, MF_THREADS), idiv_up(height, rpt)), t2(MF_THREADS);
        mean_cols_blocked_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                        width, height, kh, inv_area, rpt);
    } else {
        // 大核：每列一线程滑动窗口 O(1)/像素（行分块会让每线程重复 O(k) 直加，不划算）
        dim3 g2(idiv_up(width, MF_THREADS)), t2(MF_THREADS);
        mean_cols_kernel<<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep, width, height, kh, inv_area);
    }
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

} // anonymous namespace

void gpu_mean_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                  int width, int height, int kw, int kh, cudaStream_t stream)
{
    const float inv_area = 1.0f / (float)(kw * kh);
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
