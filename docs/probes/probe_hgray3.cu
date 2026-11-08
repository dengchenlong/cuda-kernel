// probe_hgray3 (r52): hist-gray 剩余 ~10% 空间（smem 原子）与固定启动开销。
// probe_hgray2 结论（4K）：full 0.1034 / read-floor 0.0932 / empty(纯启动) 0.0102 ms
//   → 16 条 smem 原子/uint4 只值 0.0102ms（10%），读已达 100 GB/s（天花板）。
//   4 条原子/uint4 与 0 条原子同速（被访存延迟完全掩盖）⇒ 只有「16 条」这一档有代价。
// 本探针对原子代价做归因与压制尝试：
//   base    : shist[256]，warp 内 32 lane 随机 bin（bank = bin&31 有冲突，同地址偶发）
//   rep4    : 4 份私有 hist（副本号 = warp&3），跨 warp 同 bin 不再串行
//   rep4pad : 同上但 stride 257（副本错开 bank）
//   bybyte  : 4 份副本按「uint4 内的字节序号 u」分配（同 warp 内 4 组字节各自独立）
//   gridN   : base 换更小的 grid（固定开销随块数变化）
//   notail  : base 去掉 last-block 归约尾（隔离尾块成本）
// 计时 3 轮交错取 min；base 之外的变体都与 CPU 参照比对（全 256 bin 相等才算过）。
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

template <int MODE>   // 0 base, 1 rep4, 2 rep4pad, 3 bybyte
__global__ void k_atom(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                       int* __restrict__ d_hist, int* __restrict__ partials, unsigned* __restrict__ counter)
{
    __shared__ int shist[4 * 257];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp = tid >> 5;
    constexpr int N = (MODE == 0) ? 256 : (MODE == 2 ? 4 * 257 : 1024);
#pragma unroll
    for (int i = tid; i < N; i += 256) shist[i] = 0;
    __syncthreads();
    int* myh;
    if (MODE == 0)      myh = shist;
    else if (MODE == 2) myh = shist + (warp & 3) * 257;
    else                myh = shist + (warp & 3) * 256;

    const int n4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row = (const uint4*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < n4; x += blockDim.x) {
            const uint4 v = row[x];
            const unsigned w4[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                int* h = (MODE == 3) ? (shist + u * 256) : myh;
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&h[(w4[u] >> b) & 0xff], 1);
            }
        }
    }
    __syncthreads();
    __shared__ bool amLast;
    int s = 0;
    if (MODE == 0) {
        s = shist[tid];
    } else if (MODE == 2) {
#pragma unroll
        for (int r = 0; r < 4; ++r) s += shist[r * 257 + tid];
    } else {
#pragma unroll
        for (int r = 0; r < 4; ++r) s += shist[r * 256 + tid];
    }
    partials[blockIdx.x * 256 + tid] = s;
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

// 无 last-block 尾（只写 partials）—— 隔离尾块成本，故意不写 d_hist
__global__ void k_notail(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                         int* __restrict__ partials)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int n4 = W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridDim.x * blockDim.y) {
        const uint4* row = (const uint4*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < n4; x += blockDim.x) {
            const uint4 v = row[x];
            const unsigned w4[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w4[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    partials[blockIdx.x * 256 + tid] = shist[tid];
}

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
    const int NV = 8;
    const char* vn[NV] = {"base", "rep4", "rep4pad", "bybyte", "grid24", "grid12", "notail", "base2"};

    printf("%-12s", "size");
    for (int v = 0; v < NV; ++v) printf("%10s", vn[v]);
    printf("\n");
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t step = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &step, (size_t)W, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(7);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, step, host.data(), W, W, H, cudaMemcpyHostToDevice));
        int* dh = nullptr; CUDA_CHECK(cudaMalloc(&dh, 256 * sizeof(int)));

        // 参照（CPU）
        int want[256] = {0};
        for (size_t i = 0; i < host.size(); ++i) want[host[i]]++;

        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);
        const dim3 t(32, 8);
        double best[NV];
        int bad[NV] = {0};
        for (int v = 0; v < NV; ++v) best[v] = 1e30;

        auto launch = [&](int v) {
            switch (v) {
            case 0: k_atom<0><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 1: k_atom<1><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 2: k_atom<2><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 3: k_atom<3><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 4: k_atom<0><<<24, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 5: k_atom<0><<<12, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            case 6: k_notail<<<G, t>>>(src, step, W, H, ws().partials); break;
            default: k_atom<0><<<G, t>>>(src, step, W, H, dh, ws().partials, ws().counter); break;
            }
        };

        for (int v = 0; v < NV; ++v) {
            CUDA_CHECK(cudaMemset(dh, 0x5A, 256 * sizeof(int)));
            launch(v); launch(v);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (v != 6) {
                int got[256];
                CUDA_CHECK(cudaMemcpy(got, dh, sizeof(got), cudaMemcpyDeviceToHost));
                for (int i = 0; i < 256; ++i) if (got[i] != want[i]) ++bad[v];
            }
        }
        for (int round = 0; round < 3; ++round)
            for (int v = 0; v < NV; ++v) best[v] = std::min(best[v], median_run([&] { launch(v); }, warmup, reps));

        printf("%5dx%-6d", W, H);
        for (int v = 0; v < NV; ++v) printf("%10.4f", best[v]);
        printf("\n%-12s", "  bins_bad");
        for (int v = 0; v < NV; ++v) printf("%10d", bad[v]);
        printf("\n");
        cudaFree(src); cudaFree(dh);
    }
    return 0;
}
