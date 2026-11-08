// probe_hgray2: staged timing of the hist-gray path (r52).
// Question: hist-gray 4K is the global min item (1.48x) while hist-binary with identical
// byte traffic is 7.3x -> is the cost the smem atomics, the read, or the launch/tail?
// Variants (all u8, 256 bins, last-block reduction kept identical so the tail is common):
//   full      : copy of the current hist_lb_vec_kernel (16 smem atomics per uint4)
//   read      : same loop, atomics replaced by register checksum  -> read+loop floor
//   a4 / a1   : only 4 / 1 atomics per uint4                      -> atomic cost slope
//   flat      : flat grid-stride over the whole image, 2x unrolled uint4 loads + 16 atomics
//   flatread  : flat + no atomics
//   empty     : empty kernel (grid 48 x 256) -> launch cost floor
//   gridk     : full kernel launched with grid = 48/96/192
// Timing: 3 interleaved rounds, median per round, min across rounds (clock drift lesson r49).
#include "ops.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

constexpr int G = 48;

struct WS { int* partials = nullptr; unsigned* counter = nullptr; };
WS& ws() { static WS w; return w; }

__global__ void k_empty() {}

// ---------- shared tail: block writes its 256 bins, last block reduces ----------
__device__ __forceinline__ void lb_store(int tid, int val, int* partials, unsigned* counter, int* d_hist)
{
    __shared__ bool amLast;
    partials[blockIdx.x * 256 + tid] = val;
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridDim.x - 1);
        amLast = (ticket == gridDim.x - 1);
    }
    __syncthreads();
    if (!amLast) return;
    int a0 = 0, a1 = 0, a2 = 0, a3 = 0, q = 0;
    for (; q + 4 <= (int)gridDim.x; q += 4) {
        a0 += partials[(q + 0) * 256 + tid];
        a1 += partials[(q + 1) * 256 + tid];
        a2 += partials[(q + 2) * 256 + tid];
        a3 += partials[(q + 3) * 256 + tid];
    }
    for (; q < (int)gridDim.x; ++q) a0 += partials[q * 256 + tid];
    d_hist[tid] = a0 + a1 + a2 + a3;
}

// MODE: 0 = 16 atomics/uint4 (full), 1 = no atomics (checksum), 2 = 4 atomics, 3 = 1 atomic
template <int MODE>
__global__ void k_row(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                      int* __restrict__ d_hist, int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    unsigned acc = 0;
    const int n4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row = (const uint4*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < n4; x += blockDim.x) {
            const uint4 v = row[x];
            const unsigned w4[4] = { v.x, v.y, v.z, v.w };
            if (MODE == 1) {
                acc += v.x + v.y + v.z + v.w;
            } else {
#pragma unroll
                for (int u = 0; u < 4; ++u) {
                    if (MODE == 2 && u > 0) break;
#pragma unroll
                    for (int b = 0; b < 32; b += 8) {
                        if (MODE == 3 && b > 0) break;
                        atomicAdd(&shist[(w4[u] >> b) & 0xff], 1);
                    }
                }
            }
        }
    }
    __syncthreads();
    lb_store(tid, MODE == 1 ? (int)acc : shist[tid], partials, counter, d_hist);
}

// flat grid-stride, 2x unrolled uint4 loads
template <int MODE>
__global__ void k_flat(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                       int* __restrict__ d_hist, int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const unsigned total = (unsigned)((size_t)W * H) >> 4;      // uint4 count (W%16==0 assumed)
    const int stride = gridDim.x * blockDim.x * blockDim.y;
    const unsigned base = blockIdx.x * blockDim.x * blockDim.y + tid;
    unsigned acc = 0;
    for (unsigned i = base; i < total; i += stride) {
        const uint4* p = ((const uint4*)src) + i;
        const uint4 v = p[0];
        if (MODE == 1) { acc += v.x + v.y + v.z + v.w; }
        else {
            const unsigned w4[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w4[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    lb_store(tid, MODE == 1 ? (int)acc : shist[tid], partials, counter, d_hist);
}

// ---------- host side ----------
double median_run(const std::function<void()>& f, int warmup, int reps)
{
    for (int i = 0; i < warmup; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> ms(reps);
    cudaEvent_t s, e; cudaEventCreate(&s); cudaEventCreate(&e);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(s); f(); cudaEventRecord(e);
        cudaEventSynchronize(e); cudaEventElapsedTime(&ms[i], s, e);
    }
    cudaEventDestroy(s); cudaEventDestroy(e);
    std::nth_element(ms.begin(), ms.begin() + reps / 2, ms.end());
    return ms[reps / 2];
}

} // namespace

int main()
{
    CUDA_CHECK(cudaFree(0));
    CUDA_CHECK(cudaMalloc(&ws().partials, (size_t)256 * 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ws().counter, sizeof(unsigned)));
    CUDA_CHECK(cudaMemset(ws().counter, 0, sizeof(unsigned)));

    const int sizes[][2] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const char* names[] = {"full    ", "read    ", "a4      ", "a1      ", "flat    ", "flatread",
                           "grid48  ", "grid96  ", "grid192 ", "empty   "};
    const int NV = 10;

    printf("%-12s%10s%10s%10s%10s%10s%10s%10s%10s%10s%10s\n", "size", names[0], names[1], names[2],
           names[3], names[4], names[5], names[6], names[7], names[8], names[9]);
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t step = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &step, (size_t)W, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(7);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, step, host.data(), W, W, H, cudaMemcpyHostToDevice));
        int* dh = nullptr; CUDA_CHECK(cudaMalloc(&dh, 256 * sizeof(int)));

        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);
        const dim3 t(32, 8);
        double best[NV];
        for (int v = 0; v < NV; ++v) best[v] = 1e30;

        auto run_variant = [&](int v) {
            switch (v) {
            case 0: return median_run([&] { k_row<0><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 1: return median_run([&] { k_row<1><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 2: return median_run([&] { k_row<2><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 3: return median_run([&] { k_row<3><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 4: return median_run([&] { k_flat<0><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 5: return median_run([&] { k_flat<1><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 6: return median_run([&] { k_row<0><<<48, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 7: return median_run([&] { k_row<0><<<96, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            case 8: return median_run([&] { k_row<0><<<192, t>>>(src, step, W, H, dh, ws().partials, ws().counter); }, warmup, reps);
            default: return median_run([&] { k_empty<<<G, t>>>(); }, warmup, reps);
            }
        };
        // 3 interleaved rounds
        for (int round = 0; round < 3; ++round)
            for (int v = 0; v < NV; ++v) best[v] = std::min(best[v], run_variant(v));

        printf("%5dx%-6d", W, H);
        for (int v = 0; v < NV; ++v) printf("%10.4f", best[v]);
        printf("\n");
        printf("%-12s", "  GB/s");
        for (int v = 0; v < NV; ++v) printf("%10.1f", (double)W * H / 1e6 / (best[v] * 1e-3) / 1000.0);
        printf("\n");
        cudaFree(src); cudaFree(dh);
    }
    return 0;
}
