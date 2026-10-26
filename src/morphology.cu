// 腐蚀/膨胀（二值与灰度统一，min/max 语义）
// 核心优化：van Herk/Gil-Werman 分离算法 —— 任意核尺寸 O(1)/像素（OpenCV CUDA 是每轴 O(k) 的暴力窗）
// 水平/垂直各一遍。边界语义与 OpenCV morphologyDefaultBorderValue 一致：
//   腐蚀(最小值滤波)边界视为 255（恒等元），膨胀(最大值滤波)边界视为 0 —— 等价于窗口 clamp 到图像内。
#include "ops.h"
#include <algorithm>

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

template <typename V, int TH>
__global__ void morph3_fuse_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint8_t* __restrict__ dst, size_t dstep,
                                   int W, int H)
{
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (gx < W);
    // 非活跃线程仍要参与 shuffle，给一个 16B 对齐的安全取数地址
    // （W 非 16 倍数时向下取整对齐，否则 uint4 取数会 misaligned）；
    // 活跃的尾块线程照旧读自己的 gx（16 对齐，pitch 保证不越界），这样邻居拿到的是真数据
    const int gxl  = act ? gx : ((W - 16) & ~15);


    // 原地（dst==src）不在这里处理：host 端先融合写 workspace 再整图拷贝回 dst，
    // 这样内核保持最瘦（31 regs / 高占用），比在内核里做 halo + warp 边界预存快一倍
    uint hp[4], hc[4], hn[4];

    // 一行内的水平 3 邻域 min/max（4 个 32 位字打包 16 字节）
    auto hrow = [&](const uint8_t* rp, uint o[4]) {
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

    hrow(src + (size_t)max(0, y0 - 1) * sstep, hp);
    hrow(src + (size_t)y0 * sstep, hc);
    for (int y = y0; y < y1; ++y) {
        hrow(src + (size_t)min(H - 1, y + 1) * sstep, hn);
        if (act) {
            uint o[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = V::op4(hp[i], V::op4(hc[i], hn[i]));
            *(uint4*)(dst + (size_t)y * dstep + gx) = make_uint4(o[0], o[1], o[2], o[3]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) { hp[i] = hc[i]; hc[i] = hn[i]; }
    }
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

template <typename Op>
void morph3_fused(const uint8_t* d_src, size_t srcStep, uint8_t* d_dst, size_t dstStep,
                  int width, int height, int TH, cudaStream_t stream)
{
    using V = typename VOpOf<Op>::V;
    int th = ((width + 15) / 16 + 31) / 32 * 32;
    if (th > 256) th = 256;
    const int gxx = (width + 16 * th - 1) / (16 * th);
    const int th3 = (TH <= 8) ? 8 : (TH <= 16 ? 16 : 64);
    dim3 g(gxx, (height + th3 - 1) / th3);
    if (th3 == 8)       morph3_fuse_kernel<V, 8><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
    else if (th3 == 16) morph3_fuse_kernel<V, 16><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
    else                morph3_fuse_kernel<V, 64><<<g, th, 0, stream>>>(d_src, srcStep, d_dst, dstStep, width, height);
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
        const int TH = height <= 800 ? 8 : (height <= 1200 ? 16 : 64);
        if (inplace) {
            // 原地：先融合写到 workspace（每行按 16 对齐的紧排），再整图 uint4 拷回 dst。
            // 内核里做 halo/warp 边界预存能省掉这次拷贝，但实测寄存器从 31 涨到 52、
            // 4K 反而慢 2x（0.19 → 0.38ms），所以选「拷一遍」。
            const int tstep = (width + 15) & ~15;
            uint8_t* tmp = ws_grow((size_t)tstep * height + 1024);
            morph3_fused<Op>(d_src, srcStep, tmp, tstep, width, height, TH, stream);
            int th = ((width + 15) / 16 + 31) / 32 * 32;
            if (th > 256) th = 256;
            dim3 gc((width + 16 * th - 1) / (16 * th), height);
            copy16_kernel<<<gc, th, 0, stream>>>(tmp, tstep, d_dst, dstStep, width);
        } else {
            morph3_fused<Op>(d_src, srcStep, d_dst, dstStep, width, height, TH, stream);
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

template <typename Op>
void morph32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
             int width, int height, int kw, int kh, cudaStream_t stream)
{
    // 关键：列遍必须读「行遍的输出」而不是 dst 原地读写 —— 行分块后，上方 chunk 的
    // 输出行会被下方 chunk 当输入读，block 间无同步 → 竞态（u8 版列遍读 tmp 同理）
    const size_t tstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(tstep * height + 1024);
    dim3 g1((width + 255) / 256, height), t1(256);
    morph32_rows_kernel<Op><<<g1, t1, 0, stream>>>(d_src, srcStep, tmp, tstep, width, kw);
    const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
    dim3 g2((width + 255) / 256, (height + rpt - 1) / rpt), t2(256);
    morph32_cols_kernel<Op><<<g2, t2, 0, stream>>>(tmp, tstep, d_dst, dstStep,
                                                   width, height, kh, rpt);
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
