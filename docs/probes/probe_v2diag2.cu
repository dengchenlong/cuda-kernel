// probe_v2diag2：把 probe_g7px 的 V2 内核**原样**复制进来跑 W=32 H=8，
// 若复现 FAIL，再 dump 线程 1 的 h[7][16] 与 CPU 参照逐项对。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "ops.h"
using uint = unsigned int;
#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while(0)

static void fillPat(std::vector<uint8_t>& h, int W, int H, unsigned seed)
{
    unsigned st = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            st = st * 1664525u + 1013904223u;
            h[(size_t)y * W + x] = (uint8_t)((st >> 16) ^ (x * 37 + y * 91));
        }
}

__host__ __device__ __forceinline__ int v2mirror(int i, int len)
{
    if (len <= 1) return 0;
    const int per = 2 * len - 2;
    i %= per;
    if (i < 0) i += per;
    return (i < len) ? i : (per - i);
}

// ===== 与 probe_g7px g7px_v2_kernel 主体相同（r58 修复后），加 DBG dump =====
template <int TH, bool DBG>
__global__ void v2_kernel(const uint8_t* __restrict__ src, size_t sstep,
                          uint8_t* __restrict__ dst, size_t dstep, int W, int H,
                          uint* dbg)
{
    constexpr uint K1 = 2u | (7u << 8) | (14u << 16) | (18u << 24);
    constexpr uint K2 = 14u | (7u << 8) | (2u << 16);
    const int tid  = threadIdx.x;
    const int gx   = (blockIdx.x * blockDim.x + tid) * 16;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (gx < W);
    const int gxl  = act ? gx : ((W - 16) & ~15);

    uint h[7][16];

    auto hrow = [&](int ry, uint o[16]) {
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
        for (int i = 0; i < 16; ++i) o[i] = __dp4a(rw[i], K2, __dp4a(lw[i], K1, 0u));
    };

#pragma unroll
    for (int i = 0; i < 7; ++i) hrow(v2mirror(y0 - 3 + i, H), h[i]);

    for (int y = y0; y < y1; ++y) {
        uint hn[16];
        hrow(v2mirror(y + 4, H), hn);
        if (DBG && tid == 1 && blockIdx.x == 0 && blockIdx.y == 0) {
            for (int r = 0; r < 7; ++r)
                for (int k = 0; k < 16; ++k)
                    dbg[(size_t)y * 112 + r * 16 + k] = h[r][k];
        }
        if (act) {
            uint p[16];
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                const uint U  = h[0][k] + h[6][k];
                const uint V  = h[1][k] + h[5][k];
                const uint Wd = h[2][k] + h[4][k];
                const uint S  = 7u * (V + 2u * Wd) + 2u * U + 18u * h[3][k];
                const int v = __float2int_rn((float)S * (1.f / 4096.f));
                p[k] = (uint)max(0, min(255, v));
            }
            uint4 o;
            o.x = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
            o.y = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
            o.z = p[8] | (p[9] << 8) | (p[10] << 16) | (p[11] << 24);
            o.w = p[12] | (p[13] << 8) | (p[14] << 16) | (p[15] << 24);
            *(uint4*)(dst + (size_t)y * dstep + gx) = o;
        }
#pragma unroll
        for (int k = 0; k < 16; ++k) {
            h[0][k] = h[1][k]; h[1][k] = h[2][k]; h[2][k] = h[3][k]; h[3][k] = h[4][k];
            h[4][k] = h[5][k]; h[5][k] = h[6][k]; h[6][k] = hn[k];
        }
    }
}

int main()
{
    CK(cudaFree(0));
    const int W = 32, H = 8;
    const int TH = 16;
    std::vector<uint8_t> img((size_t)W * H);
    fillPat(img, W, H, 12345u + (unsigned)W * 7u + (unsigned)H);
    uint8_t *dA, *dB; size_t astep, bstep;
    CK(cudaMallocPitch(&dA, &astep, W, H));
    CK(cudaMallocPitch(&dB, &bstep, W, H));
    CK(cudaMemcpy2D(dA, astep, img.data(), W, W, H, cudaMemcpyHostToDevice));
    uint* dbg; CK(cudaMalloc(&dbg, (size_t)H * 7 * 16 * 4));
    const int th = ((W + 15) / 16 + 31) / 32 * 32;
    dim3 g((unsigned)((W + 16 * th - 1) / (16 * th)), (unsigned)((H + TH - 1) / TH));
    v2_kernel<TH, true><<<g, th>>>(dA, astep, dB, bstep, W, H, dbg);
    CK(cudaDeviceSynchronize());
    std::vector<uint8_t> out((size_t)W * H);
    CK(cudaMemcpy2D(out.data(), W, dB, bstep, W, H, cudaMemcpyDeviceToHost));
    std::vector<uint> hdbg((size_t)H * 7 * 16);
    CK(cudaMemcpy(hdbg.data(), dbg, hdbg.size() * 4, cudaMemcpyDeviceToHost));

    // CPU 参照：REFLECT_101 全 2D 窗
    auto px = [&](int y, int x) {
        int per = 2 * H - 2; int m = y % per; if (m < 0) m += per;
        int yy = (m < H) ? m : (per - m);
        per = 2 * W - 2; m = x % per; if (m < 0) m += per;
        int xx = (m < W) ? m : (per - m);
        return img[(size_t)yy * W + xx];
    };
    static const int kw[7] = {2, 7, 14, 18, 14, 7, 2};
    int bad = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            long long S = 0;
            for (int j = 0; j < 7; ++j)
                for (int i2 = 0; i2 < 7; ++i2)
                    S += (long long)kw[j] * kw[i2] * px(y + j - 3, x + i2 - 3);
            double d = (double)S / 4096.0;
            int v = (int)lrint(d);
            v = v < 0 ? 0 : (v > 255 ? 255 : v);
            if (out[(size_t)y * W + x] != (uint8_t)v) {
                if (bad < 8) printf("y=%d x=%d mine=%d ref=%d\n", y, x, out[(size_t)y * W + x], v);
                ++bad;
            }
        }
    printf(bad ? "V2_FULL_FAIL bad=%d\n" : "V2_FULL_OK\n", bad);

    // dump 线程 1（x=16..31）的 h[r][k] 与 CPU 水平和对比
    if (bad) {
        for (int y = 0; y < 3; ++y) {
            printf("== y=%d ==\n", y);
            for (int r = 0; r < 7; ++r) {
                const int ry = v2mirror(y + r - 3, H);
                for (int k = 0; k < 16; ++k) {
                    const int x = 16 + k;
                    int S = 0;
                    for (int j = 0; j < 7; ++j) {
                        int per = 2 * W - 2; int m = (x + j - 3) % per; if (m < 0) m += per;
                        int xx = (m < W) ? m : (per - m);
                        S += kw[j] * img[(size_t)ry * W + xx];
                    }
                    const uint mine = hdbg[(size_t)y * 112 + r * 16 + k];
                    if (mine != (uint)S)
                        printf("  h[%d][%d] mine=%u ref=%d (row %d)\n", r, k, mine, S, ry);
                }
            }
        }
    }
    return 0;
}
