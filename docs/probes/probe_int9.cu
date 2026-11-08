// probe_int9 (r52): integral 融合内核的**硬件地板**测量。
// 目的：r51 一直拿「纯 uint4 拷贝 93 GB/s」当天花板，但融合内核是**读 1N(u8) + 写 4N(u32)**，
// 读写比 1:4，与拷贝（1:1）不同。若混合读写的地板本就只有 ~80 GB/s，那 4K 76.6 GB/s
// 已经贴住物理上限，继续调 ROWS/屏障都是白费。
// 变体（全部无扫描、无屏障、无 smem，只做 u8->u32 扩展搬运）：
//   A exp   : 每线程读 1 个 uint4(u8x16)，写 4 个 uint4(u32x16)
//   B expst : 同上，写用 __stcs
//   C exp2  : 每线程 2 行（同 R2 的访存形状）
//   D copy  : 纯 u8->u8 uint4 拷贝（1N:1N，对照 r51 引用的 93 GB/s）
//   E rdonly: 只读 u8（1N）+ 累加，不写（纯读上限）
//   F u32cp : 纯 u32 uint4 拷贝（4N:4N，对照大块宽总线）
// 计时 3 轮交错取 min（r49 时钟漂移）。
#include "ops.h"
#include <cstdio>
#include <vector>
#include <algorithm>
#include <functional>
#include <random>

namespace {

__global__ void k_exp(const uint8_t* __restrict__ src, size_t sstep,
                      uint32_t* __restrict__ dst, size_t dstep, int W, int H, int mode)
{
    const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = W >> 4;
    if (c0 >= n4) return;
    for (int y = blockIdx.y; y < H; y += gridDim.y) {
        const uint4 v = *reinterpret_cast<const uint4*>(src + (size_t)y * sstep + (size_t)c0 * 16);
        uint32_t o[16];
        o[0] = v.x & 0xffu;         o[1] = (v.x >> 8) & 0xffu;
        o[2] = (v.x >> 16) & 0xffu; o[3] = v.x >> 24;
        o[4] = v.y & 0xffu;         o[5] = (v.y >> 8) & 0xffu;
        o[6] = (v.y >> 16) & 0xffu; o[7] = v.y >> 24;
        o[8] = v.z & 0xffu;         o[9] = (v.z >> 8) & 0xffu;
        o[10] = (v.z >> 16) & 0xffu; o[11] = v.z >> 24;
        o[12] = v.w & 0xffu;        o[13] = (v.w >> 8) & 0xffu;
        o[14] = (v.w >> 16) & 0xffu; o[15] = v.w >> 24;
        uint32_t* q = (uint32_t*)((uint8_t*)dst + (size_t)y * dstep) + (size_t)c0 * 16;
        uint4 a0 = make_uint4(o[0], o[1], o[2], o[3]);
        uint4 a1 = make_uint4(o[4], o[5], o[6], o[7]);
        uint4 a2 = make_uint4(o[8], o[9], o[10], o[11]);
        uint4 a3 = make_uint4(o[12], o[13], o[14], o[15]);
        if (mode == 1) {
            __stcs(reinterpret_cast<uint4*>(q),      a0);
            __stcs(reinterpret_cast<uint4*>(q + 4),  a1);
            __stcs(reinterpret_cast<uint4*>(q + 8),  a2);
            __stcs(reinterpret_cast<uint4*>(q + 12), a3);
        } else if (mode == 2) {   // 两行（R2 形状）
            const int y2 = y + 1;
            if (y2 < H) {
                const uint4 w = *reinterpret_cast<const uint4*>(src + (size_t)y2 * sstep + (size_t)c0 * 16);
                uint32_t p[16];
                p[0] = w.x & 0xffu;         p[1] = (w.x >> 8) & 0xffu;
                p[2] = (w.x >> 16) & 0xffu; p[3] = w.x >> 24;
                p[4] = w.y & 0xffu;         p[5] = (w.y >> 8) & 0xffu;
                p[6] = (w.y >> 16) & 0xffu; p[7] = w.y >> 24;
                p[8] = w.z & 0xffu;         p[9] = (w.z >> 8) & 0xffu;
                p[10] = (w.z >> 16) & 0xffu; p[11] = w.z >> 24;
                p[12] = w.w & 0xffu;        p[13] = (w.w >> 8) & 0xffu;
                p[14] = (w.w >> 16) & 0xffu; p[15] = w.w >> 24;
                uint32_t* q2 = (uint32_t*)((uint8_t*)dst + (size_t)y2 * dstep) + (size_t)c0 * 16;
                *reinterpret_cast<uint4*>(q)       = a0;
                *reinterpret_cast<uint4*>(q + 4)   = a1;
                *reinterpret_cast<uint4*>(q + 8)   = a2;
                *reinterpret_cast<uint4*>(q + 12)  = a3;
                *reinterpret_cast<uint4*>(q2)      = make_uint4(p[0], p[1], p[2], p[3]);
                *reinterpret_cast<uint4*>(q2 + 4)  = make_uint4(p[4], p[5], p[6], p[7]);
                *reinterpret_cast<uint4*>(q2 + 8)  = make_uint4(p[8], p[9], p[10], p[11]);
                *reinterpret_cast<uint4*>(q2 + 12) = make_uint4(p[12], p[13], p[14], p[15]);
                y += 1;
            }
        } else {
            *reinterpret_cast<uint4*>(q)      = a0;
            *reinterpret_cast<uint4*>(q + 4)  = a1;
            *reinterpret_cast<uint4*>(q + 8)  = a2;
            *reinterpret_cast<uint4*>(q + 12) = a3;
        }
    }
}

__global__ void k_copy8(const uint8_t* __restrict__ src, size_t sstep,
                        uint8_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = W >> 4;
    if (c0 >= n4) return;
    for (int y = blockIdx.y; y < H; y += gridDim.y)
        *reinterpret_cast<uint4*>(dst + (size_t)y * dstep + (size_t)c0 * 16) =
            *reinterpret_cast<const uint4*>(src + (size_t)y * sstep + (size_t)c0 * 16);
}

__global__ void k_copy32(const uint32_t* __restrict__ src, size_t sstep,
                         uint32_t* __restrict__ dst, size_t dstep, int W, int H)
{
    const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = W >> 2;
    if (c0 >= n4) return;
    for (int y = blockIdx.y; y < H; y += gridDim.y)
        *reinterpret_cast<uint4*>(dst + (size_t)y * (dstep >> 2) + (size_t)c0 * 4) =
            *reinterpret_cast<const uint4*>(src + (size_t)y * (sstep >> 2) + (size_t)c0 * 4);
}

__global__ void k_rdonly(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                         uint32_t* __restrict__ sink)
{
    const int c0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int n4 = W >> 4;
    if (c0 >= n4) return;
    uint32_t s = 0;
    for (int y = blockIdx.y; y < H; y += gridDim.y) {
        const uint4 v = *reinterpret_cast<const uint4*>(src + (size_t)y * sstep + (size_t)c0 * 16);
        s += v.x + v.y + v.z + v.w;
    }
    if (s == 0xdeadbeefu) sink[0] = s;
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
    const int sizes[][2] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int NV = 6;
    const char* vn[NV] = {"exp", "expst", "exp2", "copy8", "rdonly", "u32cp"};
    const double N[NV] = {5, 5, 5, 2, 1, 8};       // 各变体的有效流量倍数（N 字节单位）

    printf("%-12s", "size");
    for (int v = 0; v < NV; ++v) printf("%9s", vn[v]);
    printf("   (ms / GB/s by own traffic)\n");
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        uint8_t* src = nullptr; size_t sstep = 0;
        uint8_t* d8 = nullptr; size_t ds8 = 0;
        uint32_t* d32 = nullptr; size_t ds32 = 0;
        CUDA_CHECK(cudaMallocPitch(&src, &sstep, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&d8, &ds8, (size_t)W, H));
        CUDA_CHECK(cudaMallocPitch(&d32, &ds32, (size_t)W * 4, H));
        std::vector<uint8_t> host((size_t)W * H);
        std::mt19937 rng(11);
        for (auto& c : host) c = (uint8_t)(rng() & 0xFF);
        CUDA_CHECK(cudaMemcpy2D(src, sstep, host.data(), W, W, H, cudaMemcpyHostToDevice));
        uint32_t* sink = nullptr; CUDA_CHECK(cudaMalloc(&sink, 4));

        const dim3 t(128);
        const dim3 g((W / 16 + 127) / 128, 8);
        const dim3 g32((W / 4 + 127) / 128, 8);
        const int reps = std::max(60, std::min(250, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(10, reps / 5);

        auto launch = [&](int v) {
            switch (v) {
            case 0: k_exp<<<g, t>>>(src, sstep, d32, ds32, W, H, 0); break;
            case 1: k_exp<<<g, t>>>(src, sstep, d32, ds32, W, H, 1); break;
            case 2: k_exp<<<g, t>>>(src, sstep, d32, ds32, W, H, 2); break;
            case 3: k_copy8<<<g, t>>>(src, sstep, d8, ds8, W, H); break;
            case 4: k_rdonly<<<g, t>>>(src, sstep, W, H, sink); break;
            default: k_copy32<<<g32, t>>>(reinterpret_cast<const uint32_t*>(d32), ds32,
                                          reinterpret_cast<uint32_t*>(d32), ds32, W * 4, H); break;
            }
        };
        double best[NV];
        for (int v = 0; v < NV; ++v) best[v] = 1e30;
        for (int round = 0; round < 3; ++round)
            for (int v = 0; v < NV; ++v) best[v] = std::min(best[v], median_run([&] { launch(v); }, warmup, reps));

        printf("%5dx%-6d", W, H);
        for (int v = 0; v < NV; ++v) printf("%9.4f", best[v]);
        printf("\n%-12s", "  GB/s");
        for (int v = 0; v < NV; ++v)
            printf("%9.1f", N[v] * W * H / 1e6 / (best[v] * 1e-3) / 1000.0);
        printf("\n");
        cudaFree(src); cudaFree(d8); cudaFree(d32); cudaFree(sink);
    }
    return 0;
}
