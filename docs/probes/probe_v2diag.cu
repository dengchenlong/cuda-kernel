// probe_v2diag：V2 hrow 的最小诊断 —— 直接调 V2 内核跑一行、dump 16 个水平输出，
// 与 CPU 逐像素 REFLECT_101 精确参照比对。目标是把 lw/rw/Warr 逐项对出。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "ops.h"
using uint = unsigned int;
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)

// 与 probe_g7px V2 完全相同的 hrow 逻辑，抽成 device 函数 dump 版
__global__ void dump_hrow(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                          int ry, uint* out /* [16] x nblocks */)
{
    constexpr uint K1 = 2u | (7u << 8) | (14u << 16) | (18u << 24);
    constexpr uint K2 = 14u | (7u << 8) | (2u << 16);
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : ((W - 16) & ~15);
    if (!act) return;

    uint lw[16], rw[16];
    const uint8_t* rp = src + (size_t)ry * sstep;
    const uint4 w = *(const uint4*)(rp + gxl);
    uint pv4 = __shfl_up_sync(0xffffffffu, w.w, 1);
    if (gxl == 0) {
        pv4 = (uint)rp[4] | ((uint)rp[3] << 8) | ((uint)rp[2] << 16) | ((uint)rp[1] << 24);
    } else if (lane == 0) {
        pv4 = *(const uint*)(rp + gxl - 4);
    }
    uint nv = __shfl_down_sync(0xffffffffu, w.x, 1);
    if (gxl + 16 >= W) {
        const int q0 = 2 * (W - 1) - (gxl + 16);
        const uint8_t* q = rp + q0;
        nv = (uint)q[0] | ((uint)q[-1] << 8) | ((uint)q[-2] << 16) | ((uint)q[-3] << 24);
    } else if (lane == 31) {
        nv = *(const uint*)(rp + gxl + 16);
    }
    const uint Warr[6] = {
        (pv4 >> 8) | (w.x << 24),
        __funnelshift_r(w.x, w.y, 8),
        __funnelshift_r(w.y, w.z, 8),
        __funnelshift_r(w.z, w.w, 8),
        __funnelshift_r(w.w, nv, 8),
        nv >> 8,
    };
#pragma unroll
    for (int a = 0; a < 4; ++a) {
        const uint wa = Warr[a], wb = Warr[a + 1], wc = Warr[a + 2];
        lw[4 * a + 0] = wa;
        lw[4 * a + 1] = __funnelshift_r(wa, wb, 8);
        lw[4 * a + 2] = __funnelshift_r(wa, wb, 16);
        lw[4 * a + 3] = __funnelshift_r(wa, wb, 24);
        rw[4 * a + 0] = wb;
        rw[4 * a + 1] = __funnelshift_r(wb, wc, 8);
        rw[4 * a + 2] = __funnelshift_r(wb, wc, 16);
        rw[4 * a + 3] = __funnelshift_r(wb, wc, 24);
    }
#pragma unroll
    for (int i = 0; i < 16; ++i)
        out[gx + i] = __dp4a(rw[i], K2, __dp4a(lw[i], K1, 0u));
}

int main()
{
    CK(cudaFree(0));
    // 用 W=1920 的图 + 一行 ry=0，warp 内 lane 全部活跃（16px/线程 → 128 线程 → 1 warp...不，
    // 128 线程 = 4 warps。取 W=256 保证单 block 且覆盖 lane0/31 边界
    const int W = 256, H = 16, ry = 0;
    std::vector<uint8_t> img((size_t)W * H);
    unsigned st = 42;
    for (size_t i = 0; i < img.size(); ++i) { st = st * 1664525u + 1013904223u; img[i] = (uint8_t)(st >> 24); }
    uint8_t* dA; size_t astep;
    CK(cudaMallocPitch(&dA, &astep, W, H));
    CK(cudaMemcpy2D(dA, astep, img.data(), W, W, H, cudaMemcpyHostToDevice));
    uint* dOut; CK(cudaMalloc(&dOut, 1024 * sizeof(uint)));
    dump_hrow<<<1, W / 16>>>(dA, astep, W, H, ry, dOut);
    CK(cudaDeviceSynchronize());
    std::vector<uint> hout(256);
    CK(cudaMemcpy(hout.data(), dOut, 256 * sizeof(uint), cudaMemcpyDeviceToHost));

    // CPU 参照：行 ry 的 REFLECT_101 流字节 + 7-tap dp4a 等价加权和
    auto ref = [&](int x) {
        int S = 0;
        static const int kw[7] = {2, 7, 14, 18, 14, 7, 2};
        for (int j = 0; j < 7; ++j) {
            int idx = x + j - 3;
            int per = 2 * W - 2, m = idx % per; if (m < 0) m += per;
            int px = (m < W) ? m : (per - m);
            S += kw[j] * img[(size_t)ry * W + px];
        }
        return S;
    };
    int bad = 0;
    for (int x = 0; x < W; ++x) {
        const uint mine = hout[x];
        const uint r = (uint)ref(x);
        if (mine != r) {
            if (bad < 12) printf("x=%3d mine=%u ref=%u\n", x, mine, r);
            ++bad;
        }
    }
    printf(bad ? "V2_HROW_FAIL bad=%d\n" : "V2_HROW_OK\n", bad);
    return 0;
}
