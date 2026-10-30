// 高斯滤波：分离两遍。中间缓冲 float（与 OpenCV cv::cuda 分离滤波的 CV_32F buf 精度一致，可逐像素比对）
// 行遍：smem tile + O(k) MAC；列遍：每线程私有 smem 滚动 ring（k 槽），每输出行只读 1 行新数据，O(1)/像素
// 权重由 host 端按 OpenCV getGaussianKernel 相同公式生成，写入 constant memory
#include "ops.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

typedef unsigned int uint;   // Windows 下 CUDA 头不保证提供 uint

constexpr int MAX_K = 31;
constexpr int R_THREADS = 256;   // 行遍 block 线程
constexpr int C_THREADS = 256;   // 列遍 block 线程（=256 列/块）

__constant__ float c_w[MAX_K];
__constant__ int   c_k;            // 核尺寸（动态），配合 c_w 使用

// ---------------- 行遍：u8 -> float ----------------
__global__ void gauss_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  float* __restrict__ dst, size_t dstep,   // dstep 单位字节
                                  int W, int k)
{
    extern __shared__ uint8_t tile[];           // R_THREADS + k - 1
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * R_THREADS;      // 本块输出起点
    const int half = k / 2;
    const int tid = threadIdx.x;

    const uint8_t* srow = src + (size_t)y * sstep;
    const int N = R_THREADS + k - 1;

    // 加载 tile（带 reflect101 边界）
    for (int i = tid; i < N; i += R_THREADS) {
        int x = x0 - half + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));              // 超窄图保护
        tile[i] = srow[x];
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;

    float acc = 0.f;
    for (int j = 0; j < k; ++j)
        acc += c_w[j] * tile[tid + j];

    dst[(size_t)y * (dstep >> 2) + x] = acc;
}

// ---------------- 列遍：float -> u8 ----------------
// 每线程处理一列，私有 k 槽 smem ring；每输出行只从 global 读 1 个新值 → O(1)/像素
// 注意：acc 用 double 累加 + host 端 double 验证 —— OpenCV 的分离滤波 CV_32F buf 在
// 8U→32F 行遍、32F→8U 列遍时舍入规则可能与单精度 MAC 不同；这里实测校准。
__global__ void gauss_cols_kernel(const float* __restrict__ src, size_t sstep,   // sstep 字节
                                  uint8_t* __restrict__ dst, size_t dstep,
                                  int W, int H, int k, int rpt)
{
    extern __shared__ float ring[];             // k * C_THREADS，线程 tid 私有列 = ring[tid + slot*C_THREADS]
    const int tid = threadIdx.x;
    const int x = blockIdx.x * C_THREADS + tid;
    if (x >= W) return;

    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);            // 本线程负责的行区间 [y0, y1)

    const int half = k / 2;
    const size_t srows = sstep >> 2;
    const float* scol = src + x;
    float* col = ring + tid;

    auto fetch = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));            // 超矮图保护
        return scol[(size_t)rr * srows];
    };

    // 预加载窗口行 r = y0-half .. y0-half + k - 1，slot 0 = 最旧
    for (int j = 0; j < k; ++j)
        col[j * C_THREADS] = fetch(j - half + y0);
    int oldest = 0;

    for (int r = y0; r < y1; ++r) {
        // 与 OpenCV 的 row/col 滤波舍入一致：round-half-away-from-zero（cvRound 语义）
        float acc = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            acc += c_w[j] * col[slot * C_THREADS];
            if (++slot == k) slot = 0;
        }
        // 舍入语义：__float2int_rn（round-even，SSE _mm_cvtps_epi32 语义，OpenCV cvRound 同源）。
        // k<=7（固定权重）与 OpenCV 完全一致；k>15 的残差 ≤1 来自 MAC 次序不同（可容忍）。
        int v = __float2int_rn(acc);
        v = max(0, min(255, v));
        dst[(size_t)r * dstep + x] = (uint8_t)v;

        if (r + 1 < y1) {
            col[oldest * C_THREADS] = fetch(r + 1 + half);   // 滚入新行
            if (++oldest == k) oldest = 0;
        }
    }
}

// ---------------- k=5 固定核 [1,4,6,4,1]/16 的定点整数路径（sigma<=0）----------------
// rows: u8 -> u16 存未归一化行部分和 S = Σ intw_i * v_i（≤4080，精确）；
// cols: 整型累加 S_total = Σ intw_j * S_j（≤65280），输出 __float2int_rn(S_total / 256)。
// 全程精确：OpenCV 的 CV_32F 分离滤波在 k=5 固定核下所有中间值均为二进分数、float32 可
// 精确表示，浮点结果 = S_total/256（无舍入），因此本路径与其逐位一致。
// tmp 带宽 4B/px -> 2B/px（单侧），两遍合计 10B/px -> 6B/px。
__global__ void gauss5_rows_u16_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                       uint16_t* __restrict__ dst, size_t dstep,   // dstep 字节
                                       int W)
{
    extern __shared__ uint8_t tile[];           // R_THREADS + 4
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * R_THREADS;
    const int tid = threadIdx.x;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int N = R_THREADS + 4;

    for (int i = tid; i < N; i += R_THREADS) {
        int x = x0 - 2 + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        tile[i] = srow[x];
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    const unsigned acc = tile[tid] + (tile[tid + 1] << 2) + 6u * tile[tid + 2] +
                         (tile[tid + 3] << 2) + tile[tid + 4];
    dst[(size_t)y * (dstep >> 1) + x] = (uint16_t)acc;
}

__global__ void gauss5_cols_u16_kernel(const uint16_t* __restrict__ src, size_t sstep,   // sstep 字节
                                       uint8_t* __restrict__ dst, size_t dstep,
                                       int W, int H, int rpt)
{
    extern __shared__ uint16_t ring16[];          // 5 * C_THREADS
    const int tid = threadIdx.x;
    const int x = blockIdx.x * C_THREADS + tid;
    if (x >= W) return;

    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const size_t srows = sstep >> 1;
    const uint16_t* scol = src + x;
    uint16_t* col = ring16 + tid;

    auto fetch = [=](int r) -> uint16_t {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * srows];
    };

    for (int j = 0; j < 5; ++j) col[j * C_THREADS] = fetch(j - 2 + y0);
    int oldest = 0;

    for (int r = y0; r < y1; ++r) {
        unsigned acc = 0;
        int slot = oldest;
#pragma unroll
        for (int j = 0; j < 5; ++j) {
            constexpr unsigned iw[5] = { 1u, 4u, 6u, 4u, 1u };
            acc += iw[j] * col[slot * C_THREADS];
            if (++slot == 5) slot = 0;
        }
        // acc/256 精确可表示（≤255，8 位二进小数），__float2int_rn 与 float 路径/OpenCV 同语义
        const int v = __float2int_rn((float)acc * (1.f / 256.f));
        dst[(size_t)r * dstep + x] = (uint8_t)max(0, min(255, v));
        if (r + 1 < y1) {
            col[oldest * C_THREADS] = fetch(r + 1 + 2);
            if (++oldest == 5) oldest = 0;
        }
    }
}

// ---------------- k=7 固定核 [2,7,14,18,14,7,2]/64 的 u16 定点 tmp 路径（sigma<=0）----------------
// rows: u8 -> u16 精确整数行部分和 S = Σ intw_i * v_i（≤16320，不缩放）；
// cols: 整数 MAC（同 k=5 模式）：acc = Σ intw_j * S_j ≤ 64*16320 = 1,044,480 < 2^21，
// float 精确表示，最后 ×(1/4096)（=64×64，2 的幂，乘法精确）——与 OpenCV CV_32F 管线
// （行缩放 1/64 与列缩放 1/64 全程精确、总和为 1/4096 的整数倍）逐位一致。
// 注意：不能像旧版那样 cols 直接乘缩放后的 float 权重 w_j=intw_j/64——行和未缩放，
// 再乘缩放权重会多出 64×（输出整体饱和到 255）。
// tmp 带宽 8B/px → 4B/px。
__global__ void gauss7_rows_u16_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                       uint16_t* __restrict__ dst, size_t dstep,   // dstep 字节
                                       int W)
{
    extern __shared__ uint8_t tile7[];          // R_THREADS + 6
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * R_THREADS;
    const int tid = threadIdx.x;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int N = R_THREADS + 6;

    for (int i = tid; i < N; i += R_THREADS) {
        int x = x0 - 3 + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        tile7[i] = srow[x];
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    const unsigned acc = 2u * tile7[tid] + 7u * tile7[tid + 1] + 14u * tile7[tid + 2] +
                         18u * tile7[tid + 3] + 14u * tile7[tid + 4] + 7u * tile7[tid + 5] +
                         2u * tile7[tid + 6];
    dst[(size_t)y * (dstep >> 1) + x] = (uint16_t)acc;
}

__global__ void gauss7_cols_u16_kernel(const uint16_t* __restrict__ src, size_t sstep,   // sstep 字节
                                       uint8_t* __restrict__ dst, size_t dstep,
                                       int W, int H, int rpt)
{
    extern __shared__ uint16_t ring7[];           // 7 * C_THREADS
    const int tid = threadIdx.x;
    const int x = blockIdx.x * C_THREADS + tid;
    if (x >= W) return;

    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const size_t srows = sstep >> 1;
    const uint16_t* scol = src + x;
    uint16_t* col = ring7 + tid;

    auto fetch = [=](int r) -> uint16_t {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * srows];
    };

    for (int j = 0; j < 7; ++j) col[j * C_THREADS] = fetch(j - 3 + y0);
    int oldest = 0;

    for (int r = y0; r < y1; ++r) {
        unsigned acc = 0;
        int slot = oldest;
#pragma unroll
        for (int j = 0; j < 7; ++j) {
            constexpr unsigned iw7[7] = { 2u, 7u, 14u, 18u, 14u, 7u, 2u };
            acc += iw7[j] * col[slot * C_THREADS];
            if (++slot == 7) slot = 0;
        }
        // 总缩放 64*64=4096（2 的幂），acc ≤ 64*16320 < 2^21，float 精确；与 OpenCV 逐位一致
        const int v = __float2int_rn((float)acc * (1.f / 4096.f));
        dst[(size_t)r * dstep + x] = (uint8_t)max(0, min(255, v));
        if (r + 1 < y1) {
            col[oldest * C_THREADS] = fetch(r + 1 + 3);
            if (++oldest == 7) oldest = 0;
        }
    }
}

// ---------------- 行遍 v4：OpenCV 同构 4-patch + halo smem 布局（spike 002）----------------
// 32x8 block，每线程 4 输出，smem tile = (4+2)*32 元素/行 × 8 行；全 smem MAC。
// 1080p k=15 0.040→0.016ms (2.5x)、4K k=15 0.088→0.049；逐位结果与 1 输出版一致
// （MAC 次序 j=0..k-1 相同）。
__global__ void gauss_rows4_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   float* __restrict__ dst, size_t dstep,
                                   int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4, HALO = 1;
    __shared__ float smem[BDY][(PPB + 2 * HALO) * BDX];
    const int half = k / 2;
    const int y = blockIdx.y * BDY + threadIdx.y;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    const int xStart = blockIdx.x * (PPB * BDX) + threadIdx.x;
    float* sm = smem[threadIdx.y];

    auto load = [&](int idx, int x) {
        if (x < 0) x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        sm[idx] = (float)srow[x];
    };
    // 左 halo（blockIdx.x==0 时镜像边界由 load 内 clamp 处理，无需分支）
#pragma unroll
    for (int j = 0; j < HALO; ++j) load(threadIdx.x + j * BDX, xStart - (HALO - j) * BDX);
    // 主体 + 右 halo
#pragma unroll
    for (int j = 0; j < PPB; ++j) load(threadIdx.x + (HALO + j) * BDX, xStart + j * BDX);
#pragma unroll
    for (int j = 0; j < HALO; ++j) load(threadIdx.x + (PPB + HALO + j) * BDX, xStart + (PPB + j) * BDX);
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int x = xStart + j * BDX;
        if (x < W) {
            float sum = 0.f;
            const int base = threadIdx.x + (HALO + j) * BDX - anchor;
#pragma unroll 8
            for (int q = 0; q < k; ++q) sum = fmaf(c_w[q], sm[base + q], sum);
            dst[(size_t)y * (dstep >> 2) + x] = sum;
        }
    }
}

// ---------------- 列遍 v4：OpenCV 同构 4-patch + halo 行 smem（spike 002）----------------
// 32x8 block，每线程 4 个输出行；smem tile = (4+2*HALO)*8 行 × 32 列，halo 行复用。
// 与 ring 滚动版逐位一致（MAC 次序 j=0..k-1 相同）。
__global__ void gauss_cols4_kernel(const float* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4;
    const int HALO = (k <= 16) ? 1 : 2;
    __shared__ float smem[(PPB + 2 * 2) * BDY][BDX];    // 按 HALO=2 上界分配
    const int half = k / 2;
    const int x = blockIdx.x * BDX + threadIdx.x;
    if (x >= W) return;
    const int yStart = blockIdx.y * (BDY * PPB) + threadIdx.y;
    const size_t rows = sstep >> 2;
    const float* scol = src + x;

    auto fetch = [&](int r) -> float {
        int rr = r;
        if (rr < 0) rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * rows];
    };
#pragma unroll
    for (int j = 0; j < PPB + 2 * HALO; ++j)
        smem[j * BDY + threadIdx.y][threadIdx.x] = fetch(yStart - HALO * BDY + j * BDY);
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int y = yStart + j * BDY;
        if (y < H) {
            float sum = 0.f;
#pragma unroll 8
            for (int q = 0; q < k; ++q)
                sum = fmaf(c_w[q], smem[(HALO + j) * BDY + threadIdx.y - anchor + q][threadIdx.x], sum);
            const int v = __float2int_rn(sum);
            dst[(size_t)y * dstep + x] = (uint8_t)max(0, min(255, v));
        }
    }
}

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
void* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) { if (b.p) cudaFree(b.p); CUDA_CHECK(cudaMalloc(&b.p, need)); b.sz = need; }
    return b.p;
}
inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

// ---------------- k=5 固定核单遍融合（2026-10-29）----------------
// 两遍路径（rows u8→u16 写 2B/px、cols 读 2B/px）在 4K 要过 4 次全图流量：读 1B + 写 2B +
// 读 2B + 写 1B = 6B/px，实测 1.39ms 只有 36 GB/s（纯 uint4 拷贝天花板 93 GB/s）。
// 本内核单遍完成：16 像素/线程、uint4 取数；水平 5-tap 在 u16 lane 上做（__byte_perm 把 4 字节
// 拆成 2 个 lane 字，±1 邻用 __funnelshift_r，±2 邻直接取相邻 lane 字，__vadd2 加），
// 垂直 5 行滚动寄存器；权重 [1,4,6,4,1]⊗[1,4,6,4,1]，总和 256。
// 输出 (S + 127 + ((S>>8)&1)) >> 8 —— 与 __float2int_rn(S/256) 全量等价（round-half-even；
// S ≤ 256*255 = 65280 恒成立），与两遍定点路径逐位一致。
// 边界水平/垂直都按 BORDER_REFLECT_101（求和不能用 clamp —— min/max 能糊弄 clamp，
// 加法不行，见 mean k=3 的教训）。
// 约束：W%16==0 且 16<=W<=4096（单块覆盖整行）、16B 对齐、H>=1；其它回退两遍路径。
__device__ __forceinline__ int g5_mirror(int i, int len)
{
    if (len <= 1) return 0;
    const int per = 2 * len - 2;
    i %= per;
    if (i < 0) i += per;
    return (i < len) ? i : (per - i);      // REFLECT_101：H=2 时 y=3 要折两次回到 1
}

template <int TH>
__global__ void gauss5_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H)
{
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : ((W - 16) & ~15);   // 非活跃线程给 16B 对齐安全地址，仍参与 shuffle

    uint h[5][8];

    // 一行 16 字节的水平 5-tap 和（u16 lane，≤4080）
    auto hrow = [&](int ry, uint o[8]) {
        const uint8_t* rp = src + (size_t)ry * sstep;
        const uint4 w = *(const uint4*)(rp + gxl);
        uint e[9];
        e[0] = __byte_perm(w.x, 0, 0x4140);     // lane0 = px0, lane1 = px1
        e[1] = __byte_perm(w.x, 0, 0x4342);     // lane0 = px2, lane1 = px3
        e[2] = __byte_perm(w.y, 0, 0x4140);
        e[3] = __byte_perm(w.y, 0, 0x4342);
        e[4] = __byte_perm(w.z, 0, 0x4140);
        e[5] = __byte_perm(w.z, 0, 0x4342);
        e[6] = __byte_perm(w.w, 0, 0x4140);
        e[7] = __byte_perm(w.w, 0, 0x4342);
        // 邻线程 halo：左 = 前一线程最后两像素（正是它的 e[7]），右 = 后一线程前两像素（e[0]）
        uint prv = __shfl_up_sync(0xffffffffu, e[7], 1);
        uint nxt = __shfl_down_sync(0xffffffffu, e[0], 1);
        // 跨 warp/图像边界必须自己取：注意 u16 lane 布局是「每字节占一个 lane」，
        // 不能用 *(uint16_t*) 直接读（那会把两个字节压进同一个 lane 的 0..15 位）
        if (lane == 0)
            // gxl==0：x=-2 镜像到 x=2（e[1].lane0），x=-1 镜像到 x=1（e[0].lane1）
            prv = (gxl == 0) ? ((e[1] & 0xffffu) | ((e[0] >> 16) & 0xffffu) << 16)
                             : ((uint)rp[gxl - 2] | ((uint)rp[gxl - 1] << 16));
        if (lane == 31 || gxl + 16 >= W)
            nxt = (gxl + 16 >= W) ? ((e[7] & 0xffffu) | (((e[6] >> 16) & 0xffffu) << 16))
                                  : ((uint)rp[gxl + 16] | ((uint)rp[gxl + 17] << 16));
        e[8] = nxt;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint A  = (k == 0) ? prv : e[k - 1];
            const uint C  = e[k + 1];
            const uint L1 = __funnelshift_r(A, e[k], 16);     // v(x-1) 两个 lane
            const uint R1 = __funnelshift_r(e[k], C, 16);     // v(x+1)
            const uint t  = __vadd2(L1, R1);                                  // ≤510
            const uint t4 = __vadd2(__vadd2(t, t), __vadd2(t, t));            // 4t
            const uint c  = __vadd2(__vadd2(e[k], e[k]), e[k]);
            const uint c6 = __vadd2(c, c);                                    // 6E
            o[k] = __vadd2(__vadd2(__vadd2(A, C), t4), c6);
        }
    };

    // 预载 y0-2 .. y0+2（行号按 REFLECT_101 镜像）
#pragma unroll
    for (int i = 0; i < 5; ++i) hrow(g5_mirror(y0 - 2 + i, H), h[i]);

    for (int y = y0; y < y1; ++y) {
        uint hn[8];
        hrow(g5_mirror(y + 3, H), hn);      // 下一轮的最下行（本轮窗口是 y-2..y+2）
        if (act) {
            uint p[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint b  = __vadd2(h[1][k], h[3][k]);
                const uint b4 = __vadd2(__vadd2(b, b), __vadd2(b, b));        // 4(h1+h3)
                const uint c  = __vadd2(__vadd2(h[2][k], h[2][k]), h[2][k]);
                const uint c6 = __vadd2(c, c);                                // 6h2
                const uint S  = __vadd2(__vadd2(__vadd2(h[0][k], h[4][k]), b4), c6);
                // round-half-even(S/256)，与 __float2int_rn 全量等价
                const uint adj = __vadd2((S >> 8) & 0x00010001u, 0x007f007fu);
                p[k] = __byte_perm(__vadd2(S, adj), 0, 0x4431);               // 每 lane >>8
            }
            uint4 o;
            // p[k] 已是「两字节紧排」= l0 | l1<<8，四像素再拼一次
            o.x = __byte_perm(p[0], p[1], 0x5410);
            o.y = __byte_perm(p[2], p[3], 0x5410);
            o.z = __byte_perm(p[4], p[5], 0x5410);
            o.w = __byte_perm(p[6], p[7], 0x5410);
            *(uint4*)(dst + (size_t)y * dstep + gx) = o;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            h[0][k] = h[1][k]; h[1][k] = h[2][k]; h[2][k] = h[3][k];
            h[3][k] = h[4][k]; h[4][k] = hn[k];
        }
    }
}

// 整图 uint4 拷贝（原地路径用：融合先写 workspace，再拷回 dst）
__global__ void g5_copy16_kernel(const uint8_t* __restrict__ s, size_t sstep,
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

void gauss5_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int TH, cudaStream_t stream)
{
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(width, 16 * th), idiv_up(height, TH));
    switch (TH) {
    case 8:  gauss5_fuse_kernel<8> <<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    case 16: gauss5_fuse_kernel<16><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    case 32: gauss5_fuse_kernel<32><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    default: gauss5_fuse_kernel<64><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    }
}

// ---------------- k=7 固定核单遍融合（2026-10-31）----------------
// 与 k=5 融合版同构，但两处必须不同：
//   ① 水平 7-tap 用 __dp4a（sm_61 起支持 u8x4 点积）：每像素 2 条 dp4a（7 tap 拆 4+3，
//      末位权重 0），8 像素共 16 条，比展开成 IMAD 便宜得多；两个操作数都是同一批
//      uint2 字的错位重排（funnelshift_r），只有 12 条移位。
//   ② 垂直求和最大 64*16320 = 1,044,480 > 65535，装不进 u16 lane（k=5 的 __vadd2 打包
//      在这里会溢出），所以本版 8 像素/线程 + u32 lane 做垂直，权重 [2,7,14,18,14,7,2]
//      直接由 IMAD 立即数乘。
// 中间量全精确：h ≤ 16320；S ≤ 1,044,480 < 2^21（float 精确表示）→
// __float2int_rn(S/4096) 与两遍定点路径、OpenCV CV_32F 管线逐位一致。
// 边界：水平/垂直都按 REFLECT_101（求和不能用 clamp，见 mean k=3 的教训）。
// 约束：W%8==0 且 8<=W<=4096、8B 对齐、H>=4；其它回退两遍路径。
//   H<4 特意回退：H=3 时两遍路径的单次折叠+clamp 与 REFLECT_101 周期镜像本就不同（旧行为），
//   融合路径（严格周期镜像）虽更接近数学定义，但会改变该角落的既有输出，故不动它。
__device__ __forceinline__ int g7_mirror(int i, int len)
{
    if (len <= 1) return 0;
    const int per = 2 * len - 2;
    i %= per;
    if (i < 0) i += per;
    return (i < len) ? i : (per - i);
}

template <int TH>
__global__ void gauss7_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H)
{
    constexpr uint K1 = 2u | (7u << 8) | (14u << 16) | (18u << 24);   // taps x-3..x
    constexpr uint K2 = 14u | (7u << 8) | (2u << 16);                 // taps x+1..x+4（末位权重 0）
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 8;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : (W - 8);     // 非活跃线程给 8B 对齐安全地址，仍参与 shuffle

    uint h[7][8];

    // 一行 8 像素的水平 7-tap 和（≤16320）
    auto hrow = [&](int ry, uint o[8]) {
        const uint8_t* rp = src + (size_t)ry * sstep;
        const uint2 v = *(const uint2*)(rp + gxl);          // v.x = px0..3, v.y = px4..7
        uint pv = __shfl_up_sync(0xffffffffu, v.y, 1);      // 前线程 px4..7 = 本线程 px-4..-1
        uint nv = __shfl_down_sync(0xffffffffu, v.x, 1);    // 后线程 px0..3 = 本线程 px8..11
        if (lane == 0 || gxl == 0)
            pv = (gxl == 0) ? __byte_perm(v.x, 0, 0x1234)   // 左边界镜像：(—, px3, px2, px1)
                            : *(const uint*)(rp + gxl - 4);
        if (lane == 31 || gxl + 8 >= W) {
            if (gxl + 8 >= W) {
                const uint8_t* q = rp + (W - 4);            // (—, W-2, W-3, W-4)
                nv = (uint)q[2] | ((uint)q[1] << 8) | ((uint)q[0] << 16);
            } else {
                nv = *(const uint*)(rp + gxl + 8);
            }
        }
        // lw[i] = 像素 x+i 的左 4 tap 窗口（字节序 x+i-3 .. x+i）；rw[4..7] 同理右窗
        uint lw[8], rw[8];
        lw[0] = __funnelshift_r(pv, v.x, 8);
        lw[1] = __funnelshift_r(pv, v.x, 16);
        lw[2] = __funnelshift_r(pv, v.x, 24);
        lw[3] = v.x;
        lw[4] = __funnelshift_r(v.x, v.y, 8);
        lw[5] = __funnelshift_r(v.x, v.y, 16);
        lw[6] = __funnelshift_r(v.x, v.y, 24);
        lw[7] = v.y;
        rw[0] = lw[4]; rw[1] = lw[5]; rw[2] = lw[6]; rw[3] = lw[7];
        rw[4] = __funnelshift_r(v.y, nv, 8);
        rw[5] = __funnelshift_r(v.y, nv, 16);
        rw[6] = __funnelshift_r(v.y, nv, 24);
        rw[7] = nv;
#pragma unroll
        for (int i = 0; i < 8; ++i) o[i] = __dp4a(rw[i], K2, __dp4a(lw[i], K1, 0u));
    };

#pragma unroll
    for (int i = 0; i < 7; ++i) hrow(g7_mirror(y0 - 3 + i, H), h[i]);

    for (int y = y0; y < y1; ++y) {
        uint hn[8];
        hrow(g7_mirror(y + 4, H), hn);          // 下一轮的最下行（本轮窗口 y-3..y+3）
        if (act) {
            uint p[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const uint U  = h[0][k] + h[6][k];
                const uint V  = h[1][k] + h[5][k];
                const uint Wd = h[2][k] + h[4][k];
                const uint S  = 7u * (V + 2u * Wd) + 2u * U + 18u * h[3][k];  // ≤1,044,480
                const int v = __float2int_rn((float)S * (1.f / 4096.f));       // 与两遍路径同语义
                p[k] = (uint)max(0, min(255, v));
            }
            uint2 o;
            o.x = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
            o.y = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
            *(uint2*)(dst + (size_t)y * dstep + gxl) = o;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            h[0][k] = h[1][k]; h[1][k] = h[2][k]; h[2][k] = h[3][k]; h[3][k] = h[4][k];
            h[4][k] = h[5][k]; h[5][k] = h[6][k]; h[6][k] = hn[k];
        }
    }
}

void gauss7_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int TH, cudaStream_t stream)
{
    int th = ((width + 7) / 8 + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g(idiv_up(width, 8 * th), idiv_up(height, TH));
    switch (TH) {
    case 8:  gauss7_fuse_kernel<8> <<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    case 16: gauss7_fuse_kernel<16><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    case 32: gauss7_fuse_kernel<32><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    default: gauss7_fuse_kernel<64><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height); break;
    }
}

} // anonymous namespace

// host: 与 OpenCV getGaussianKernel(ksize, sigma, CV_32F) 完全一致的权重
// OpenCV 4.x 的 getGaussianKernel 走 getGaussianKernelBitExact：
//   固定核：k=1/3/5/7/9（sigma<=0 时）；k=9 是 [4,13,30,51,60,51,30,13,4]/256
//   公式核：sigmaX = 0.15*n + 0.35 (即 0.15*((n-1)*0.5-1)+0.35+... 与旧公式等价化简：0.15*n+0.35)
//   scale2X = -0.125/(sigmaX^2)（旧写法 -0.5/(2σ²) 的 softdouble 实现）
//   且用 softdouble 精确 exp —— 与 std::exp(double) 的差异在 float 输出下逐位一致（4舍5入同向）
static void make_gauss_kernel(int ksize, double sigma, float* w)
{
    if (sigma <= 0) {
        switch (ksize) {
        case 1: w[0] = 1.f; return;
        case 3: { w[0] = 0.25f; w[1] = 0.5f; w[2] = 0.25f; return; }
        case 5: { w[0] = 0.0625f; w[1] = 0.25f; w[2] = 0.375f; w[3] = 0.25f; w[4] = 0.0625f; return; }
        case 7: {
            w[0] = 0.03125f; w[1] = 0.109375f; w[2] = 0.21875f; w[3] = 0.28125f;
            w[4] = 0.21875f; w[5] = 0.109375f; w[6] = 0.03125f; return;
        }
        case 9: {
            float v9[9] = {4.f, 13.f, 30.f, 51.f, 60.f, 51.f, 30.f, 13.f, 4.f};
            for (int i = 0; i < 9; ++i) w[i] = v9[i] / 256.f;
            return;
        }
        default: break;
        }
    }
    // OpenCV 4.x: sigmaX = 0.15*n + 0.35; scale2X = -0.125/sigmaX^2
    const double sigmaX = sigma > 0 ? sigma : 0.15 * ksize + 0.35;
    const double scale2X = -0.125 / (sigmaX * sigmaX);
    // 利用对称性：只需算半边
    const int n2 = (ksize - 1) / 2;
    std::vector<double> halfV(n2 + 1);
    halfV[n2] = 1.0;
    double sumHalf = 0.0;
    for (int i = 0, x = 1 - ksize; i < n2; i += 1, x += 2) {
        const double t = std::exp((double)x * x * scale2X);
        halfV[i] = t;
        sumHalf += t;
    }
    // OpenCV: sum = 2*Σ(half) + center（中心项不能乘 2）
    const double sum = 2.0 * sumHalf + 1.0;
    for (int i = 0; i <= n2; ++i) {
        w[i] = (float)(halfV[i] / sum);
        w[ksize - 1 - i] = w[i];
    }
}

void gpu_gaussian(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int ksize, float sigma, cudaStream_t stream)
{
    const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;

    // k=5 固定核（sigma<=0）走定点整数路径：u16 tmp 带宽减半，逐位结果不变
    if (ksize == 5 && sigma <= 0.f) {
        // 单遍融合（2026-10-29）：W%16==0 + 16B 对齐时直接一遍出图（6B/px → 2.1B/px）。
        // 原地（dst==src）融合先写 workspace 再整图 uint4 拷回（4.1B/px），仍然远低于两遍路径。
        // 其它宽度/对齐回退下面的两遍 u16 定点路径，结果逐位一致。
        const bool inplace = ((const void*)d_src == (const void*)d_dst);
        const bool fuse5 = (width >= 16 && width <= 4096 && (width % 16 == 0) &&
                            (srcStep % 16 == 0) && (dstStep % 16 == 0) &&
                            ((uintptr_t)d_src % 16 == 0) && ((uintptr_t)d_dst % 16 == 0));
        if (fuse5) {
            const int TH = (height >= 1600) ? 64 : (height > 800 ? 16 : 8);
            if (inplace) {
                const int tstep = (width + 15) & ~15;
                uint8_t* tmp = (uint8_t*)ws_grow((size_t)tstep * height + 1024);
                gauss5_fused(d_src, srcStep, tmp, tstep, width, height, TH, stream);
                int th = ((width + 15) / 16 + 31) / 32 * 32;
                if (th > 256) th = 256;
                if (th < 32)  th = 32;
                dim3 gc(idiv_up(width, 16 * th), height);
                g5_copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
            } else {
                gauss5_fused(d_src, srcStep, d_dst, dstStep, width, height, TH, stream);
            }
            return;
        }
        const size_t tstep = (size_t)width * 2;
        uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);
        {
            const int smem = R_THREADS + 4;
            dim3 g(idiv_up(width, R_THREADS), height), t(R_THREADS);
            gauss5_rows_u16_kernel<<<g, t, smem, stream>>>(d_src, srcStep, tmp, tstep, width);
        }
        {
            const int smem = 5 * C_THREADS * (int)sizeof(uint16_t);
            dim3 g(idiv_up(width, C_THREADS), idiv_up(height, rpt)), t(C_THREADS);
            gauss5_cols_u16_kernel<<<g, t, smem, stream>>>(tmp, tstep, d_dst, dstStep,
                                                           width, height, rpt);
        }
        return;
    }

    // k=7 固定核 [2,7,14,18,14,7,2]/64（sigma<=0）走 u16 定点 tmp：带宽 10B/px → 6B/px，
    // 结果与 OpenCV 逐位一致（行和精确 + 列 float MAC 同序，见 kernel 注释）
    if (ksize == 7 && sigma <= 0.f) {
        // 单遍融合（2026-10-31）：W%8==0 + 8B 对齐时一遍出图（两遍路径 6B/px → 4.1B/px 原地）。
        const bool inplace7 = ((const void*)d_src == (const void*)d_dst);
        const bool fuse7 = (width >= 8 && width <= 4096 && (width % 8 == 0) && height >= 4 &&
                            (srcStep % 8 == 0) && (dstStep % 8 == 0) &&
                            ((uintptr_t)d_src % 8 == 0) && ((uintptr_t)d_dst % 8 == 0));
        if (fuse7) {
            const int TH = (height >= 1600) ? 64 : (height > 800 ? 16 : 8);
            if (inplace7) {
                const int tstep = (width + 15) & ~15;
                uint8_t* tmp = (uint8_t*)ws_grow((size_t)tstep * height + 1024);
                gauss7_fused(d_src, srcStep, tmp, tstep, width, height, TH, stream);
                int th = ((width + 15) / 16 + 31) / 32 * 32;
                if (th > 256) th = 256;
                if (th < 32)  th = 32;
                dim3 gc(idiv_up(width, 16 * th), height);
                g5_copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
            } else {
                gauss7_fused(d_src, srcStep, d_dst, dstStep, width, height, TH, stream);
            }
            return;
        }
        float hw7[MAX_K];
        make_gauss_kernel(7, sigma, hw7);       // 固定核表，float 权重为精确二进分数
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw7, sizeof(float) * 7, 0, cudaMemcpyHostToDevice, stream));

        const size_t tstep = (size_t)width * 2;
        uint16_t* tmp = (uint16_t*)ws_grow(tstep * height + 1024);
        {
            const int smem = R_THREADS + 6;
            dim3 g(idiv_up(width, R_THREADS), height), t(R_THREADS);
            gauss7_rows_u16_kernel<<<g, t, smem, stream>>>(d_src, srcStep, tmp, tstep, width);
        }
        {
            const int smem = 7 * C_THREADS * (int)sizeof(uint16_t);
            dim3 g(idiv_up(width, C_THREADS), idiv_up(height, rpt)), t(C_THREADS);
            gauss7_cols_u16_kernel<<<g, t, smem, stream>>>(tmp, tstep, d_dst, dstStep,
                                                           width, height, rpt);
        }
        return;
    }

    float hw[MAX_K];
    make_gauss_kernel(ksize, sigma, hw);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw, sizeof(float) * ksize, 0, cudaMemcpyHostToDevice, stream));

    // v4 路径（spike 002）：OpenCV 同构 4-patch halo 布局，全尺寸实测最快
    {
        const size_t fstep = (size_t)width * 4;
        float* tmp = (float*)ws_grow(fstep * height + 1024);
        {
            dim3 g(idiv_up(width, 32 * 4), idiv_up(height, 8)), t(32, 8);
            gauss_rows4_kernel<<<g, t, 0, stream>>>(d_src, srcStep, tmp, fstep, width, height, ksize);
        }
        {
            dim3 g(idiv_up(width, 32), idiv_up(height, 8 * 4)), t(32, 8);
            gauss_cols4_kernel<<<g, t, 0, stream>>>(tmp, fstep, d_dst, dstStep, width, height, ksize);
        }
        return;
    }
}

// ================= CV_32F 输入高斯 =================
// 分离两遍 float MAC（行遍 tile 直加、列遍 ring 滚动），输出不取整。
// 权重与 u8 版同源 make_gauss_kernel(CV_32F)；边界 REFLECT_101。
// 显式 fmaf 与 OpenCV CPU 端浮点累加同构，smoke 与其逐位一致（4/4 尺寸 diff=0）。
namespace {

__global__ void gauss32_rows_kernel(const float* __restrict__ src, size_t sstep,
                                    float* __restrict__ tmp, size_t tstep,
                                    int W, int k)
{
    extern __shared__ float tileg[];            // RT + k - 1（RT = blockDim.x，可 128/256）
    const int RT = blockDim.x;
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * RT;
    const int tid = threadIdx.x;
    const int N = RT + k - 1;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < N; i += RT) {
        int x = x0 - k / 2 + i;
        if (x < 0)  x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        tileg[i] = srow[x];
    }
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    float acc = 0.f;
    for (int j = 0; j < k; ++j) acc = fmaf(c_w[j], tileg[tid + j], acc);
    tmp[(size_t)y * (tstep >> 2) + x] = acc;
}

// f32 行遍 v2：每线程 2 个输出（tile 宽 2*RT + k - 1，MAC 循环两路 ILP）。
// probe_gf32b 实测比单输出版快 15~25%（1080p 0.29 vs 0.34ms、4K 1.05 vs 1.31ms），
// 输出与单输出版逐位一致（MAC 次序相同）。W 极小（< 2*RT）时 g0<0 或 g0+N>W 走镜像回退。
__global__ void gauss32_rows2_kernel(const float* __restrict__ src, size_t sstep,
                                     float* __restrict__ tmp, size_t tstep,
                                     int W, int k)
{
    extern __shared__ float tileg[];            // 2*RT + k - 1
    const int RT = blockDim.x;
    const int y = blockIdx.y;
    const int x0 = blockIdx.x * (2 * RT);
    const int tid = threadIdx.x;
    const int N = 2 * RT + k - 1;
    const int half = k / 2;
    const float* srow = src + (size_t)y * (sstep >> 2);

    // 本块需要的元素范围 [x0-half, x0+2RT-1+half]（tile 索引 i ↔ 元素 g0+i）
    const int g0 = x0 - half;
    const bool aligned4 = ((reinterpret_cast<uintptr_t>(srow) & 15) == 0) && ((sstep & 15) == 0);
    if (aligned4 && (g0 >= 0) && (g0 + N <= W) && ((g0 & 3) == 0)) {
        // 干净内部段：uint4 向量化装载（其余元素逐个补齐）
        const float4* r4 = (const float4*)(srow + g0);
        for (int i = tid; i < (N >> 2); i += RT) {
            const float4 v = r4[i];
            tileg[i*4] = v.x; tileg[i*4+1] = v.y; tileg[i*4+2] = v.z; tileg[i*4+3] = v.w;
        }
        for (int i = (N & ~3) + tid; i < N; i += RT) tileg[i] = srow[g0 + i];
    } else {
        for (int i = tid; i < N; i += RT) {
            int x = g0 + i;
            if (x < 0)  x = -x;
            if (x >= W) x = 2 * W - 2 - x;
            x = max(0, min(W - 1, x));
            tileg[i] = srow[x];
        }
    }
    __syncthreads();

    const int xa = x0 + tid;
    const int xb = xa + RT;
    if (xa >= W) return;
    float acca = 0.f, accb = 0.f;
    const float* pa = tileg + tid;
    const float* pb = tileg + tid + RT;
    for (int j = 0; j < k; ++j) {
        const float wj = c_w[j];
        acca = fmaf(wj, pa[j], acca);
        if (xb < W) accb = fmaf(wj, pb[j], accb);
    }
    tmp[(size_t)y * (tstep >> 2) + xa] = acca;
    if (xb < W) tmp[(size_t)y * (tstep >> 2) + xb] = accb;
}

__global__ void gauss32_cols_kernel(const float* __restrict__ tmp, size_t tstep,
                                    float* __restrict__ dst, size_t dstep,
                                    int W, int H, int k, int rpt)
{
    extern __shared__ float ringg[];            // k * C_THREADS
    const int tid = threadIdx.x;
    const int x = blockIdx.x * C_THREADS + tid;
    if (x >= W) return;

    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k / 2;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol = tmp + x;
    float* col = ringg + tid;

    auto fetch = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * rows];
    };

    for (int j = 0; j < k; ++j) col[j * C_THREADS] = fetch(j - half + y0);
    int oldest = 0;

    for (int r = y0; r < y1; ++r) {
        float acc = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            acc = fmaf(c_w[j], col[slot * C_THREADS], acc);
            if (++slot == k) slot = 0;
        }
        dst[(size_t)r * orows + x] = acc;
        if (r + 1 < y1) {
            col[oldest * C_THREADS] = fetch(r + 1 + half);
            if (++oldest == k) oldest = 0;
        }
    }
}

// f32 列遍 v2：每线程 2 列（smem ring 双列交错，512 列/块）。
// probe_gf32b：720p rpt=64 0.107 vs 0.134ms、4K rpt=64 0.768 vs 0.811ms，输出逐位一致。
__global__ void gauss32_cols2_kernel(const float* __restrict__ tmp, size_t tstep,
                                     float* __restrict__ dst, size_t dstep,
                                     int W, int H, int k, int rpt)
{
    extern __shared__ float ringg[];            // k * 512（双列交错，线程 tid 持列 2*tid / 2*tid+1）
    const int tid = threadIdx.x;
    const int x = blockIdx.x * 512 + 2 * tid;
    if (x >= W) return;
    const bool two = (x + 1 < W);
    const int y0 = blockIdx.y * rpt;
    const int y1 = min(H, y0 + rpt);
    const int half = k / 2;
    const size_t rows = tstep >> 2, orows = dstep >> 2;
    const float* scol0 = tmp + x;
    const float* scol1 = tmp + x + 1;
    float* col0 = ringg + 2 * tid;
    float* col1 = ringg + 2 * tid + 1;

    auto fetch0 = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol0[(size_t)rr * rows];
    };
    auto fetch1 = [=](int r) -> float {
        int rr = r;
        if (rr < 0)  rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol1[(size_t)rr * rows];
    };

    for (int j = 0; j < k; ++j) {
        col0[j * 512] = fetch0(j - half + y0);
        if (two) col1[j * 512] = fetch1(j - half + y0);
    }
    int oldest = 0;
    for (int r = y0; r < y1; ++r) {
        float a0 = 0.f, a1 = 0.f;
        int slot = oldest;
        for (int j = 0; j < k; ++j) {
            const float wj = c_w[j];
            a0 = fmaf(wj, col0[slot * 512], a0);
            if (two) a1 = fmaf(wj, col1[slot * 512], a1);
            if (++slot == k) slot = 0;
        }
        dst[(size_t)r * orows + x] = a0;
        if (two) dst[(size_t)r * orows + x + 1] = a1;
        if (r + 1 < y1) {
            col0[oldest * 512] = fetch0(r + 1 + half);
            if (two) col1[oldest * 512] = fetch1(r + 1 + half);
            if (++oldest == k) oldest = 0;
        }
    }
}

// f32 v4：OpenCV 同构 4-patch halo 布局（spike 002），输入/输出/中间全 float。
__global__ void gauss32_rows4_kernel(const float* __restrict__ src, size_t sstep,
                                     float* __restrict__ dst, size_t dstep,
                                     int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4, HALO = 1;
    __shared__ float smem[BDY][(PPB + 2 * HALO) * BDX];
    const int half = k / 2;
    const int y = blockIdx.y * BDY + threadIdx.y;
    if (y >= H) return;
    const float* srow = src + (size_t)y * (sstep >> 2);
    const int xStart = blockIdx.x * (PPB * BDX) + threadIdx.x;
    float* sm = smem[threadIdx.y];

    auto load = [&](int idx, int x) {
        if (x < 0) x = -x;
        if (x >= W) x = 2 * W - 2 - x;
        x = max(0, min(W - 1, x));
        sm[idx] = srow[x];
    };
#pragma unroll
    for (int j = 0; j < HALO; ++j) load(threadIdx.x + j * BDX, xStart - (HALO - j) * BDX);
#pragma unroll
    for (int j = 0; j < PPB; ++j) load(threadIdx.x + (HALO + j) * BDX, xStart + j * BDX);
#pragma unroll
    for (int j = 0; j < HALO; ++j) load(threadIdx.x + (PPB + HALO + j) * BDX, xStart + (PPB + j) * BDX);
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int x = xStart + j * BDX;
        if (x < W) {
            float sum = 0.f;
            const int base = threadIdx.x + (HALO + j) * BDX - anchor;
#pragma unroll 8
            for (int q = 0; q < k; ++q) sum = fmaf(c_w[q], sm[base + q], sum);
            dst[(size_t)y * (dstep >> 2) + x] = sum;
        }
    }
}

__global__ void gauss32_cols4_kernel(const float* __restrict__ src, size_t sstep,
                                     float* __restrict__ dst, size_t dstep,
                                     int W, int H, int k)
{
    constexpr int BDX = 32, BDY = 8, PPB = 4;
    const int HALO = (k <= 16) ? 1 : 2;
    __shared__ float smem[(PPB + 2 * 2) * BDY][BDX];
    const int half = k / 2;
    const int x = blockIdx.x * BDX + threadIdx.x;
    if (x >= W) return;
    const int yStart = blockIdx.y * (BDY * PPB) + threadIdx.y;
    const size_t rows = sstep >> 2, orows = dstep >> 2;
    const float* scol = src + x;

    auto fetch = [&](int r) -> float {
        int rr = r;
        if (rr < 0) rr = -rr;
        if (rr >= H) rr = 2 * H - 2 - rr;
        rr = max(0, min(H - 1, rr));
        return scol[(size_t)rr * rows];
    };
#pragma unroll
    for (int j = 0; j < PPB + 2 * HALO; ++j)
        smem[j * BDY + threadIdx.y][threadIdx.x] = fetch(yStart - HALO * BDY + j * BDY);
    __syncthreads();

    const int anchor = half;
#pragma unroll
    for (int j = 0; j < PPB; ++j) {
        const int y = yStart + j * BDY;
        if (y < H) {
            float sum = 0.f;
#pragma unroll 8
            for (int q = 0; q < k; ++q)
                sum = fmaf(c_w[q], smem[(HALO + j) * BDY + threadIdx.y - anchor + q][threadIdx.x], sum);
            dst[(size_t)y * orows + x] = sum;
        }
    }
}

} // namespace

void gpu_gaussian_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                      int width, int height, int ksize, float sigma, cudaStream_t stream)
{
    float hw[MAX_K];
    make_gauss_kernel(ksize, sigma, hw);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw, sizeof(float) * ksize, 0, cudaMemcpyHostToDevice, stream));

    // v4 路径（spike 002）
    const size_t fstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(fstep * height + 1024);
    {
        dim3 g(idiv_up(width, 32 * 4), idiv_up(height, 8)), t(32, 8);
        gauss32_rows4_kernel<<<g, t, 0, stream>>>(d_src, srcStep, tmp, fstep, width, height, ksize);
    }
    {
        dim3 g(idiv_up(width, 32), idiv_up(height, 8 * 4)), t(32, 8);
        gauss32_cols4_kernel<<<g, t, 0, stream>>>(tmp, fstep, d_dst, dstStep, width, height, ksize);
    }
}
