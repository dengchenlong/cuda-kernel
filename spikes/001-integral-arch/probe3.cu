// probe3: 定位 v2 integral 的 bad rows —— 大图上 v2 Pass1 输出 vs CPU 参考（Pass2 不参与）
// 结论候选：bad 来自 Pass2 carry（v2 没改 Pass2），或 bad rows=5 是 probe1 里
// cvr.convertTo 之后的比较 bug（OpenCV integral 输出是 CV_32SC1，convertTo 到 CV_32SC1 是恒等，
// 但 dOut GpuMat 的 step 对齐 4 字节对齐 pad —— download 后 cvr.step 可能 != W*4）
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
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
    p1_rows_v2_kernel<<<(H + 7) / 8, 256>>>(s, ss, dst, ds, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 只抽 3 行下载比对（第 0、1、2159 行），避免 33MB 拷贝时间
    std::vector<uint32_t> row(W);
    int bad = 0;
    for (int y : {0, 1, 7, 1000, 2159}) {
        CUDA_CHECK(cudaMemcpy2D(row.data(), (size_t)W * 4,
                                (uint8_t*)dst + (size_t)y * ds, ds, (size_t)W * 4, 1,
                                cudaMemcpyDeviceToHost));
        uint32_t run = 0;
        int rowBad = -1;
        for (int x = 0; x < W; ++x) {
            run += m.at<uchar>(y, x);
            if (row[x] != run) { rowBad = x; break; }
        }
        if (rowBad >= 0) { printf("row %d bad @x=%d\n", y, rowBad); ++bad; }
    }
    printf("pass1 spot-check bad: %d/5\n", bad);
    return bad;
}
