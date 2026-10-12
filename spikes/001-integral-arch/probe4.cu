// probe4: bad rows=5 定位 —— Pass1(v2) + Pass2(现版) 全链路 vs CPU 双重前缀（不经过 OpenCV）
// 排除 probe1 里 OpenCV 对照/convertTo/download step 的干扰，直接 CPU 算 2D prefix（u32 行内前缀 + 行间累加）
#include <opencv2/core.hpp>
#include <cstdio>
#include <random>
#include <vector>

#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

__global__ void p1_rows_v2_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int rowBase = blockIdx.x * 8;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    constexpr int SEG_BYTES = 512;
    const bool vecOK = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0);
    const int y = rowBase + warp;
    if (y >= H) return;
    const uint8_t* srow = src + (size_t)y * sstep;
    uint32_t* drow = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep);
    uint32_t carry = 0;
    for (int x0 = 0; x0 < W; x0 += SEG_BYTES) {
        const int my0 = x0 + lane * 16;
        uint32_t vals[16]; uint32_t partial = 0;
        if (my0 + 16 <= W && vecOK) {
            union { uint4 v; uint8_t b[16]; } u;
            u.v = *reinterpret_cast<const uint4*>(srow + my0);
#pragma unroll
            for (int k = 0; k < 16; ++k) { vals[k] = u.b[k]; partial += vals[k]; }
        } else {
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const int x = my0 + k; uint32_t v = 0;
                if (x < W) v = srow[x];
                vals[k] = v; partial += v;
            }
        }
        uint32_t inc = partial;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            uint32_t t = __shfl_up_sync(0xffffffffu, inc, d);
            if (lane >= d) inc += t;
        }
        const uint32_t excl = inc - partial + carry;
        carry += __shfl_sync(0xffffffffu, inc, 31);
        uint32_t run = excl;
#pragma unroll
        for (int k = 0; k < 16; ++k) { run += vals[k]; const int x = my0 + k; if (x < W) drow[x] = run; }
    }
}

__global__ void p2_cols_kernel(const uint32_t* __restrict__ in, size_t istep,
                               uint32_t* __restrict__ out, size_t ostep,
                               uint32_t* __restrict__ carry, int W, int H, int nChunks)
{
    constexpr int CHUNK_H = 512;
    const int chunk = blockIdx.y;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;
    const uint4* ip = (const uint4*)((const uint8_t*)in  + (size_t)r0 * istep) + c0 / 4;
    uint4*       op = (uint4*)((uint8_t*)out + (size_t)r0 * ostep) + c0 / 4;
    uint4 car = make_uint4(0,0,0,0);
    for (int r = r0; r < r1; ++r) {
        uint4 v = *ip;
        v.x += car.x; v.y += car.y; v.z += car.z; v.w += car.w;
        *op = v; car = v;
        ip = (const uint4*)((const uint8_t*)ip + istep);
        op = (uint4*)((uint8_t*)op + ostep);
    }
    if (nChunks > 1) {
        uint32_t* cr = carry + (size_t)chunk * W + c0;
        cr[0] = car.x; cr[1] = car.y; cr[2] = car.z; cr[3] = car.w;
    }
}
__global__ void p2_carry_kernel(uint32_t* __restrict__ out, size_t ostep,
                                const uint32_t* __restrict__ carry, int W, int H)
{
    constexpr int CHUNK_H = 512;
    const int chunk = blockIdx.y + 1;
    const int r0 = chunk * CHUNK_H;
    const int r1 = min(H, r0 + CHUNK_H);
    const int c0 = blockIdx.x * (blockDim.x * 4) + threadIdx.x * 4;
    if (c0 >= W) return;
    uint4 exc = make_uint4(0,0,0,0);
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

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int W = 3840, H = 2160;
    cv::Mat m(H, W, CV_8UC1);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)d(rng);

    uint8_t* s; size_t ss;
    CUDA_CHECK(cudaMallocPitch(&s, &ss, W, H));
    CUDA_CHECK(cudaMemcpy2D(s, ss, m.data, m.step, W, H, cudaMemcpyHostToDevice));
    uint32_t* dst; size_t ds;
    CUDA_CHECK(cudaMallocPitch(&dst, &ds, (size_t)W * 4, H));
    const int nChunks = (H + 511) / 512;
    uint32_t* carry;
    CUDA_CHECK(cudaMalloc(&carry, (size_t)nChunks * W * 4));

    p1_rows_v2_kernel<<<(H + 7) / 8, 256>>>(s, ss, dst, ds, W, H);
    p2_cols_kernel<<<dim3((W + 511) / 512, nChunks), 128>>>(dst, ds, dst, ds, carry, W, H, nChunks);
    p2_carry_kernel<<<dim3((W + 511) / 512, nChunks - 1), 128>>>(dst, ds, carry, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    // CPU 双重前缀参考
    cv::Mat ref(H, W, CV_32SC1);
    for (int y = 0; y < H; ++y) {
        uint32_t run = 0;
        for (int x = 0; x < W; ++x) {
            run += m.at<uchar>(y, x);
            ref.at<int>(y, x) = (int)run + (y > 0 ? ref.at<int>(y-1, x) : 0);
        }
    }

    std::vector<uint32_t> full((size_t)W * H);
    CUDA_CHECK(cudaMemcpy2D(full.data(), (size_t)W * 4, dst, ds, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
    int badRows = 0;
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x)
            if ((int)full[(size_t)y * W + x] != ref.at<int>(y, x)) {
                if (badRows < 5) printf("row %d x %d: got %u ref %d\n", y, x, full[(size_t)y*W+x], ref.at<int>(y,x));
                ++badRows; break;
            }
    }
    printf("v2 全链路 bad rows: %d / %d\n", badRows, H);
    return badRows > 0;
}
