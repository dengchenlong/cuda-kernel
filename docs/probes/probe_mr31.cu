// probe_mr31 (r64): mean 大核**行遍**重写 —— Hillis-Steele smem 扫描 → 每线程一段连续像素滑动
//
// 归因（probe_meanbig asym，交错 min，四尺寸）：
//   4K      (1,1) 1.506 | (31,1) 4.893 (行遍 +3.39) | (1,31) 1.861 (列遍 +0.36) | (31,31) 4.461
//   1080p   (1,1) 0.482 | (31,1) 1.619 (行遍 +1.14) | (1,31) 0.819 (列遍 +0.34) | (31,31) 1.461
//   ⇒ **行遍是唯一的大头**；列遍增量 0.36ms 已贴同尺寸地板（u16->u8 @4K 0.261）。
// 行遍现状（src/mean_filter.cu mean_rows_kernel，k>15 分支）：每 block 只做**一行的一个 256 列段**，
//   512 元 Hillis-Steele 扫描 = 9 轮 × 2 次 __syncthreads = 18 次屏障；4K 有 W/256×H = 32400 个块。
//   行遍增量 3.39ms / 3N(24.9MB) = 7.3 GB/s，而 u8->u16 地板 94.6 GB/s ⇒ 13 倍差距。
// 新内核 rows_slide_kernel<C>：**一行一个 block**（4K 2160 块 = 1/15），整段
//   [x0-half, x0+bw+half) 一次搬进 smem（REFLECT_101 镜像在载入时解决），每线程负责 C 个连续像素：
//   初始窗口和 O(k) 直加一次，之后每像素滑动「+新 -旧」= O(1)/像素。**每 block 只 1 次 __syncthreads**。
// 模式：check（vs CPU int64 精确窗口和，逐值） / ab（老内核 vs 新内核交错 min）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include "ops.h"

#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

namespace {

constexpr int MF_THREADS = 256;
constexpr int MF_PAD     = 512;

static __device__ __forceinline__ int mir(int i, int len)
{
    if (i < 0) i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// ---------------- 老内核：照抄 src/mean_filter.cu 的 mean_rows_kernel（A/B 用） ----------------
__global__ void old_rows_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                uint16_t* __restrict__ tmp, size_t tstep, int W, int k)
{
    __shared__ uint32_t sp[MF_PAD];
    const int y    = blockIdx.y;
    const int x0   = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid  = threadIdx.x;
    const int N    = MF_THREADS + k - 1;
    const uint8_t* srow = src + (size_t)y * sstep;

    for (int i = tid; i < N; i += MF_THREADS) sp[i] = srow[mir(x0 - half + i, W)];
    for (int i = N + tid; i < MF_PAD; i += MF_THREADS) sp[i] = 0;
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    uint32_t sum;
    if (k <= 15) {
        sum = 0;
#pragma unroll 4
        for (int j = 0; j < k; ++j) sum += sp[tid + j];
    } else {
#pragma unroll
        for (int d = 1; d < MF_PAD; d <<= 1) {
            const int i0 = 2 * tid, i1 = 2 * tid + 1;
            uint32_t a = 0, b = 0;
            if (i0 >= d) a = sp[i0 - d];
            if (i1 >= d) b = sp[i1 - d];
            __syncthreads();
            if (i0 >= d) sp[i0] += a;
            if (i1 >= d) sp[i1] += b;
            __syncthreads();
        }
        sum = sp[tid + k - 1] - (tid > 0 ? sp[tid - 1] : 0);
    }
    tmp[(size_t)y * (tstep >> 1) + x] = (uint16_t)sum;
}

// ---------------- 新内核：一行一 block + 每线程 C 个连续像素滑动 ----------------
template <int C>
__global__ void rows_slide_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                  uint16_t* __restrict__ tmp, size_t tstep, int W, int k)
{
    extern __shared__ uint8_t sm[];
    const int half = k >> 1;
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nld  = min(bw, W - x0) + 2 * half;
    const uint8_t* srow = src + (size_t)y * sstep;

    for (int i = tid; i < nld; i += blockDim.x)
        sm[i] = srow[mir(x0 - half + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;                    // sm 中对应全局列 gx-half
    const int n  = min(C, W - gx);

    uint32_t s = 0;
#pragma unroll 4
    for (int j = 0; j < k; ++j) s += sm[li + j];

    uint16_t* trow = tmp + (size_t)y * (tstep >> 1);
    for (int i = 0; i < n; ++i) {
        trow[gx + i] = (uint16_t)s;
        if (i + 1 < n) s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
    }
}

// v3：真正的 16 字节粒度（8 个 u16 打成一个 uint4）
template <int C>
__global__ void rows_slide3_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint16_t* __restrict__ tmp, size_t tstep, int W, int k,
                                   bool vecstore)
{
    extern __shared__ uint8_t sm[];
    const int half = k >> 1;
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nld  = min(bw, W - x0) + 2 * half;
    const uint8_t* srow = src + (size_t)y * sstep;

    for (int i = tid; i < nld; i += blockDim.x)
        sm[i] = srow[mir(x0 - half + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;
    const int n  = min(C, W - gx);

    uint32_t s = 0;
#pragma unroll 4
    for (int j = 0; j < k; ++j) s += sm[li + j];

    uint16_t p[C];
    for (int i = 0; i < n; ++i) {
        p[i] = (uint16_t)s;
        if (i + 1 < n) s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
    }
    uint16_t* trow = tmp + (size_t)y * (tstep >> 1);
    if (vecstore && n == C) {
        uint4* q = (uint4*)(trow + gx);
#pragma unroll
        for (int i = 0; i < C / 8; ++i) {
            q[i] = make_uint4(p[8*i] | ((uint32_t)p[8*i+1] << 16),
                              p[8*i+2] | ((uint32_t)p[8*i+3] << 16),
                              p[8*i+4] | ((uint32_t)p[8*i+5] << 16),
                              p[8*i+6] | ((uint32_t)p[8*i+7] << 16));
        }
    } else {
        for (int i = 0; i < n; ++i) trow[gx + i] = p[i];
    }
}

// v4：v3 + **向量化载入 + 共享内存排布修正**
// v3 诊断：4K k=31 0.4823ms / 3N(24.9MB) = 51.6 GB/s，地板 u8->u16 是 94.6 —— 还差 1.8 倍。
// 剩两个可疑点：① 载入是逐字节（`sm[i] = srow[mir(...)]`，两操作数都是标量字节）⇒ 16 个 lane
// 读同一 32B sector 只用 1 字节，读放大 16 倍；② sm 起点按 1 字节对齐，li = tid*16 处的
// 4 字节行和跨 sector。修法：镜像只在首尾两次（整段内部是连续列），主体走 uint4 载入；
// sm 前面留 16B pad 让 `sm + half` 16B 对齐。
template <int C>
__global__ void rows_slide4_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint16_t* __restrict__ tmp, size_t tstep, int W, int k,
                                   bool vecstore)
{
    extern __shared__ uint8_t smraw[];
    const int half = k >> 1;
    const int pad  = (16 - (half & 15)) & 15;  // 使 sm+half 16B 对齐（uint4 载入需要）
    uint8_t* sm = smraw + pad;                // sm[i] ↔ 全局列 x0-half+i
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nx   = min(bw, W - x0);         // 本块负责的列数（不含 halo）
    const uint8_t* srow = src + (size_t)y * sstep;

    // 左 halo（可能镜像）：最多 half 个字节，标量
    for (int i = tid; i < half; i += blockDim.x) sm[i] = srow[mir(x0 - half + i, W)];
    // 主体（连续列，不镜像；x0 是 bw 的倍数 = 16 的倍数，故两端 16B 对齐）
    for (int i = tid * 16; i < nx; i += blockDim.x * 16) {
        if (i + 16 <= nx) *(uint4*)(sm + half + i) = *(const uint4*)(srow + x0 + i);
        else for (int j = i; j < nx; ++j) sm[half + j] = srow[x0 + j];
    }
    // 右 halo（可能镜像）
    for (int i = tid; i < half; i += blockDim.x) sm[half + nx + i] = srow[mir(x0 + nx + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;
    const int n  = min(C, W - gx);

    uint32_t s = 0;
#pragma unroll 4
    for (int j = 0; j < k; ++j) s += sm[li + j];

    uint16_t p[C];
    for (int i = 0; i < n; ++i) {
        p[i] = (uint16_t)s;
        if (i + 1 < n) s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
    }
    uint16_t* trow = tmp + (size_t)y * (tstep >> 1);
    if (vecstore && n == C) {
        uint4* q = (uint4*)(trow + gx);
#pragma unroll
        for (int i = 0; i < C / 8; ++i) {
            q[i] = make_uint4(p[8*i] | ((uint32_t)p[8*i+1] << 16),
                              p[8*i+2] | ((uint32_t)p[8*i+3] << 16),
                              p[8*i+4] | ((uint32_t)p[8*i+5] << 16),
                              p[8*i+6] | ((uint32_t)p[8*i+7] << 16));
        }
    } else {
        for (int i = 0; i < n; ++i) trow[gx + i] = p[i];
    }
}


// v5：v3 + **整块路径编译期展开**。v3 的 `for (i < n)` 里 n 是运行期值，p[C] 无法寄存器化
// （落 local memory = 每线程 32B 的 local 流量）。整块（n==C）走编译期 unroll，p[] 才进寄存器。
template <int C>
__global__ void rows_slide5_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                   uint16_t* __restrict__ tmp, size_t tstep, int W, int k,
                                   bool vecstore)
{
    extern __shared__ uint8_t sm[];
    const int half = k >> 1;
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nld  = min(bw, W - x0) + 2 * half;
    const uint8_t* srow = src + (size_t)y * sstep;

    for (int i = tid; i < nld; i += blockDim.x)
        sm[i] = srow[mir(x0 - half + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;
    uint16_t* trow = tmp + (size_t)y * (tstep >> 1);

    if (gx + C <= W) {
        uint32_t s = 0;
#pragma unroll 4
        for (int j = 0; j < k; ++j) s += sm[li + j];
        uint16_t p[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            p[i] = (uint16_t)s;
            s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
        }
        if (vecstore) {
            uint4* q = (uint4*)(trow + gx);
#pragma unroll
            for (int i = 0; i < C / 8; ++i)
                q[i] = make_uint4(p[8*i] | ((uint32_t)p[8*i+1] << 16),
                                  p[8*i+2] | ((uint32_t)p[8*i+3] << 16),
                                  p[8*i+4] | ((uint32_t)p[8*i+5] << 16),
                                  p[8*i+6] | ((uint32_t)p[8*i+7] << 16));
        } else {
#pragma unroll
            for (int i = 0; i < C; ++i) trow[gx + i] = p[i];
        }
    } else {
        const int n = W - gx;
        uint32_t s = 0;
#pragma unroll 4
        for (int j = 0; j < k; ++j) s += sm[li + j];
        for (int i = 0; i < n; ++i) {
            trow[gx + i] = (uint16_t)s;
            if (i + 1 < n) s += (uint32_t)sm[li + i + k] - (uint32_t)sm[li + i];
        }
    }
}

// host 侧启动（C=16）
static void launch_new(const uint8_t* src, size_t sstep, uint16_t* tmp, size_t tstep,
                       int W, int H, int k, int ver = 3)
{
    constexpr int C = 16;
    int th = (W + C - 1) / C;
    th = (th + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    const int bw = th * C;
    const int nbx = (W + bw - 1) / bw;
    const size_t smem = (size_t)(bw + 2 * (k / 2));
    dim3 g(H, nbx);
    const bool vs = ((tstep % 16) == 0) && (((size_t)tmp % 16) == 0);
    if (ver == 1)      rows_slide_kernel<C><<<g, th, smem>>>(src, sstep, tmp, tstep, W, k);
    else if (ver == 3) rows_slide3_kernel<C><<<g, th, smem>>>(src, sstep, tmp, tstep, W, k, vs);
    else if (ver == 4) rows_slide4_kernel<C><<<g, th, smem + 16>>>(src, sstep, tmp, tstep, W, k, vs);
    else               rows_slide5_kernel<C><<<g, th, smem>>>(src, sstep, tmp, tstep, W, k, vs);
}

template <typename F>
static double timeMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a); f(); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms = 0.f; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

template <typename FA, typename FB>
static void abMin(FA fa, FB fb, int warm, int reps, double& oa, double& ob)
{
    for (int i = 0; i < warm; ++i) { fa(); fb(); }
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ba = 1e9f, bb = 1e9f;
    for (int i = 0; i < reps; ++i) {
        float ms = 0.f;
        cudaEventRecord(a); fa(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < ba) ba = ms;
        cudaEventRecord(a); fb(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < bb) bb = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    oa = ba; ob = bb;
}

struct Img {
    uint8_t* d = nullptr; size_t step = 0; int W = 0, H = 0;
    std::vector<uint8_t> h;
    void make(int w, int hh, unsigned seed) {
        W = w; H = hh; h.resize((size_t)w * hh);
        unsigned st = seed;
        for (auto& v : h) { st = st * 1664525u + 1013904223u; v = (uint8_t)(st >> 24); }
        CKV(cudaMallocPitch(&d, &step, (size_t)w, hh));
        CKV(cudaMemcpy2D(d, step, h.data(), (size_t)w, (size_t)w, hh, cudaMemcpyHostToDevice));
    }
    ~Img() { if (d) cudaFree(d); }
};

} // namespace

int main(int argc, char** argv)
{
    CKV(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "check";
    const int reps = (argc > 2) ? atoi(argv[2]) : 15;

    if (mode == "check") {
        struct S { int W, H; };
        const S shapes[] = {{7,1},{15,1},{16,1},{17,3},{31,3},{32,2},{33,17},{63,5},{64,64},{100,100},
                            {127,129},{333,257},{640,480},{720,479},{1280,720},{1920,1080},{3840,2160},
                            {4096,64},{7,400},{1281,721},{512,512}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            for (int k : {15, 31}) {
                Img src; src.make(s.W, s.H, 4242u + (unsigned)(s.W * 3 + s.H + k));
                const int tstep = s.W * 2;
                uint16_t* tmp = nullptr;
                CKV(cudaMalloc(&tmp, (size_t)tstep * s.H));
                CKV(cudaMemset(tmp, 0, (size_t)tstep * s.H));
                launch_new(src.d, src.step, tmp, tstep, s.W, s.H, k, 4);
                CKV(cudaDeviceSynchronize());
                std::vector<uint16_t> got((size_t)s.W * s.H);
                for (int y = 0; y < s.H; ++y)
                    CKV(cudaMemcpy(got.data() + (size_t)y * s.W, tmp + (size_t)y * s.W,
                                   (size_t)s.W * 2, cudaMemcpyDeviceToHost));
                // CPU 精确参考（int64 窗口和）
                const int half = k >> 1;
                size_t bad = 0, first = (size_t)-1;
                for (int y = 0; y < s.H; ++y) {
                    for (int x = 0; x < s.W; ++x) {
                        long long acc = 0;
                        for (int j = -half; j <= half; ++j) {
                            int xx = x + j;
                            if (xx < 0) xx = -xx;
                            if (xx >= s.W) xx = 2 * s.W - 2 - xx;
                            xx = std::max(0, std::min(s.W - 1, xx));
                            acc += src.h[(size_t)y * s.W + xx];
                        }
                        const uint16_t want = (uint16_t)acc;
                        const uint16_t g = got[(size_t)y * s.W + x];
                        if (g != want) { if (first == (size_t)-1) first = (size_t)y * s.W + x; ++bad; }
                    }
                }
                ++total;
                const bool ok = (bad == 0);
                printf("  %4dx%-4d k=%-2d %s", s.W, s.H, k, ok ? "[OK]" : "[FAIL]");
                if (!ok) printf("  bad=%zu first=(%d,%d) got=%u", bad,
                                (int)(first % s.W), (int)(first / s.W), got[first]);
                printf("\n");
                if (!ok) ++fails;
                cudaFree(tmp);
            }
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MR31_FAIL" : "PROBE_MR31_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "ab") {
        struct C2 { const char* n; int W, H; };
        const C2 cs[] = {{"640x480",640,480},{"1280x720",1280,720},
                         {"1920x1080",1920,1080},{"3840x2160",3840,2160}};
        for (const C2& c : cs) {
            Img src; src.make(c.W, c.H, 777u);
            const int tstep = c.W * 2;
            uint16_t* t1 = nullptr; uint16_t* t2 = nullptr;
            CKV(cudaMalloc(&t1, (size_t)tstep * c.H));
            CKV(cudaMalloc(&t2, (size_t)tstep * c.H));
            for (int k : {15, 31}) {
                double o = 0, n = 0;
                dim3 g1((c.W + MF_THREADS - 1) / MF_THREADS, c.H);
                abMin([&] { old_rows_kernel<<<g1, MF_THREADS>>>(src.d, src.step, t1, tstep, c.W, k); },
                      [&] { launch_new(src.d, src.step, t2, tstep, c.W, c.H, k, 1); },
                      3, reps, o, n);
                double d1 = 0, d3 = 0, d4 = 0, d4b = 0;
                abMin([&] { launch_new(src.d, src.step, t1, tstep, c.W, c.H, k, 1); },
                      [&] { launch_new(src.d, src.step, t2, tstep, c.W, c.H, k, 3); },
                      3, reps, d1, d3);
                abMin([&] { launch_new(src.d, src.step, t1, tstep, c.W, c.H, k, 3); },
                      [&] { launch_new(src.d, src.step, t2, tstep, c.W, c.H, k, 5); },
                      3, reps, d4, d4b);
                // 同时校验各版本输出一致
                CKV(cudaDeviceSynchronize());
                std::vector<uint16_t> a((size_t)c.W * c.H), b((size_t)c.W * c.H);
                CKV(cudaMemcpy2D(a.data(), (size_t)c.W * 2, t1, (size_t)tstep, (size_t)c.W * 2, c.H, cudaMemcpyDeviceToHost));
                CKV(cudaMemcpy2D(b.data(), (size_t)c.W * 2, t2, (size_t)tstep, (size_t)c.W * 2, c.H, cudaMemcpyDeviceToHost));
                const bool same = (memcmp(a.data(), b.data(), (size_t)c.W * c.H * 2) == 0);
                printf("%-10s k=%-2d old %8.4f | v1 %8.4f (%.2fx) | v3 %8.4f (%.2fx vs old, %.2fx vs v1) %s\n",
                       c.n, k, o, n, o / n, d3, o / d3, d1 / d3, same ? "identical" : "DIFF!!");
            }
            cudaFree(t1); cudaFree(t2);
        }
        return 0;
    }

    printf("usage: probe_mr31 [check|ab] [reps]\n");
    return 0;
}