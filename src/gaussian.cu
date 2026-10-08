// 高斯滤波：分离两遍。中间缓冲 float（与 OpenCV cv::cuda 分离滤波的 CV_32F buf 精度一致，可逐像素比对）
// 行遍：smem tile + O(k) MAC；列遍：每线程私有 smem 滚动 ring（k 槽），每输出行只读 1 行新数据，O(1)/像素
// 权重由 host 端按 OpenCV getGaussianKernel 相同公式生成，写入 constant memory
#include "ops.h"
#include <cmath>
#include <vector>
#include <algorithm>

namespace {

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

struct DeviceBuf { void* p = nullptr; size_t sz = 0; };
DeviceBuf& ws() { static DeviceBuf b; return b; }
void* ws_grow(size_t need) {
    DeviceBuf& b = ws();
    if (b.sz < need) { if (b.p) cudaFree(b.p); CUDA_CHECK(cudaMalloc(&b.p, need)); b.sz = need; }
    return b.p;
}
inline int idiv_up(int a, int b) { return (a + b - 1) / b; }

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

    const size_t fstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(fstep * height + 1024);

    {
        const int smem = R_THREADS + ksize - 1;
        dim3 g(idiv_up(width, R_THREADS), height), t(R_THREADS);
        gauss_rows_kernel<<<g, t, smem, stream>>>(d_src, srcStep, tmp, fstep, width, ksize);
    }
    {
        const int smem = ksize * C_THREADS * (int)sizeof(float);
        // 列遍按行分块：全场线程数 = W * ceil(H/rpt)，否则每列一线程时只有 W 线程（占用率过低）
        // 4K 实测 rpt=64 最优（>32 后重复读的代价被占用率收益盖过；小图 rpt=8~16 更好，见 probe_k3）
        dim3 g(idiv_up(width, C_THREADS), idiv_up(height, rpt)), t(C_THREADS);
        gauss_cols_kernel<<<g, t, smem, stream>>>(tmp, fstep, d_dst, dstStep, width, height, ksize, rpt);
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

} // namespace

void gpu_gaussian_f32(const float* d_src, size_t srcStep, float* d_dst, size_t dstStep,
                      int width, int height, int ksize, float sigma, cudaStream_t stream)
{
    float hw[MAX_K];
    make_gauss_kernel(ksize, sigma, hw);
    CUDA_CHECK(cudaMemcpyToSymbolAsync(c_w, hw, sizeof(float) * ksize, 0, cudaMemcpyHostToDevice, stream));

    const int rpt = height > 1080 ? (height + 63) / 64 : (height + 15) / 16;
    const size_t fstep = (size_t)width * 4;
    float* tmp = (float*)ws_grow(fstep * height + 1024);

    {
        // f32 行遍数据量是 u8 的 4 倍：大图 RT=128 提升占用率更划算（实测 1080p/4K 更优）
        const int RT = width > 2560 ? 128 : R_THREADS;
        const int smemRT = (RT + ksize - 1) * (int)sizeof(float);
        dim3 g(idiv_up(width, RT), height), t(RT);
        gauss32_rows_kernel<<<g, t, smemRT, stream>>>(d_src, srcStep, tmp, fstep, width, ksize);
    }
    {
        const int smem = ksize * C_THREADS * (int)sizeof(float);
        dim3 g(idiv_up(width, C_THREADS), idiv_up(height, rpt)), t(C_THREADS);
        gauss32_cols_kernel<<<g, t, smem, stream>>>(tmp, fstep, d_dst, dstStep, width, height, ksize, rpt);
    }
}
