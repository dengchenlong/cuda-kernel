// 积分图：两遍法
// Pass1: 行内前缀和 —— 每 warp 处理一行，uint4 向量化读入(每 lane 16B)，warp shuffle 扫描
// Pass2: 列方向扫描 —— 分层：先扫 chunk，再用 carry 矩阵加前缀（对 W%4!=0 有标量回退路径）
#include "ops.h"
#include <vector>
#include <algorithm>

namespace {

constexpr int SEG_BYTES = 512;      // 32 lanes * 16B
constexpr int CHUNK_H   = 512;      // 列扫描 chunk 高度
constexpr int CT_THREADS = 128;     // 列扫描 block 线程数 (每线程 4 列 → 512 列/block)

// ---------------- Pass 1: 行扫描 ----------------
// v2 launch shape（spike 001 实测）：每 block 8 行、每 warp 独占一行，grid=(H+7)/8。
// 旧版固定 12 block×256（96 warp）在 sm_120（170 SM）上只占一小部分 SM，
// 4K 0.53ms；新 shape 4K 0.19ms（2.8x），全尺寸超 OpenCV（720p 4.1x/4K 2.2x）。
// 内核逐指令与旧版 grid-stride 版一致，数值路径零改动（sm_61 同样受益或持平）。
__global__ void integral_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                     uint32_t* __restrict__ dst, size_t dstep,
                                     int W, int H)
{
    const int lane  = threadIdx.x & 31;
    const int warp  = threadIdx.x >> 5;
    constexpr int SEG_BYTES = 512;      // 32 lanes * 16B
    const bool vecOK = ((sstep & 15) == 0) &&
                       ((reinterpret_cast<uintptr_t>(src) & 15) == 0);

    const int y = blockIdx.x * 8 + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;

    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;          // 本 lane 负责 16 字节
        uint32_t vals[16];
        uint32_t partial = 0;

        if (my0 + 16 <= W && vecOK) {
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
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            run += vals[k];
            const int x = my0 + k;
            if (x < W) drow[x] = run;
        }
    }
}

// ---------------- Pass 2a: chunk 内列扫描 + 写 carry 矩阵 ----------------
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

    if (vecOK) {
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
