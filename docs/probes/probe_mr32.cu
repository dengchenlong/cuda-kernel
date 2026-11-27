// probe_mr32 (r68): mean-f32 **行遍**重写 —— 移植 r64 的 u8 行遍结构（一行一 block +
// 每线程 C 个连续像素滑动窗口），替掉 src/mean_filter.cu 里 mean32_rows_kernel 的
// 「每 block 一行 256 列段 + 512 元 Hillis-Steele 扫描（k>15 分支，9 轮 × 2 屏障）」。
//
// 归因（r67b bench 反推）：mean-f32 4K k=15 总 2.1094ms、k=31 总 4.6459ms；
//   r66 的 probe_mcol4 已量出列遍（vec）4K k=15 0.790 / k=31 0.912
//   ⇒ 行遍 ≈ 1.32 / 3.73ms。行遍流量只有 8B/px（读 4 + 写 4 = 33.2MB@4K），
//   地板 93 GB/s 对应 0.36ms ⇒ k=31 至少有 10 倍空间。
// 为什么 r66 漏了它：r66 只换了列遍（u8 侧 r65/r64 是行遍先修、列遍后修，f32 侧顺序反了）。
//
// 新内核 mean32_rows_slide_kernel<C>（照抄 u8 mean_rows_slide_kernel 的结构）：
//   ① 一行一个 block（4K 2160 块，旧路径 15×2160 个块）；
//   ② 整段 [x0-half, x0+bw+half) 一次搬进 smem，REFLECT_101 在载入时解决；
//   ③ 每线程 C 个连续像素：初始窗口 O(k) 直加一次，之后「+新 -旧」O(1)/像素；
//   ④ 整块路径写 float4（C=16 → 4 条），尾块标量。
// 算术次序与旧内核不同（滑动 vs 重算）→ float 结合律差 ~1e-5 量级，check 用容差判据
// （u8 侧是逐位；f32 侧 r66 已有同款先例：列遍 vec 与旧内核容差 1e-3）。
// 模式：check（vs CPU double 精确窗口和，容差）/ sweep（C × 四尺寸 × k=15/31，交错 min）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

namespace {

constexpr int MF_THREADS = 256;
constexpr int MF_PAD     = 512;

__device__ __forceinline__ int mir(int i, int len)
{
    if (i < 0) i = -i;
    if (i >= len) i = 2 * len - 2 - i;
    return max(0, min(len - 1, i));
}

// ---------------- 旧内核：照抄 src/mean_filter.cu 的 mean32_rows_kernel（A/B 用） ----------------
__global__ void old_rows_kernel(const float* __restrict__ src, size_t sstep,
                                float* __restrict__ tmp, size_t tstep, int W, int k)
{
    __shared__ float sp[MF_PAD];
    const int y    = blockIdx.y;
    const int x0   = blockIdx.x * MF_THREADS;
    const int half = k >> 1;
    const int tid  = threadIdx.x;
    const int N    = MF_THREADS + k - 1;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < N; i += MF_THREADS) sp[i] = srow[mir(x0 - half + i, W)];
    for (int i = N + tid; i < MF_PAD; i += MF_THREADS) sp[i] = 0.f;
    __syncthreads();

    const int x = x0 + tid;
    if (x >= W) return;
    float sum;
    if (k <= 15) {
        sum = 0.f;
#pragma unroll 4
        for (int j = 0; j < k; ++j) sum += sp[tid + j];
    } else {
#pragma unroll
        for (int d = 1; d < MF_PAD; d <<= 1) {
            const int i0 = 2 * tid, i1 = 2 * tid + 1;
            float a = 0.f, b = 0.f;
            if (i0 >= d) a = sp[i0 - d];
            if (i1 >= d) b = sp[i1 - d];
            __syncthreads();
            if (i0 >= d) sp[i0] += a;
            if (i1 >= d) sp[i1] += b;
            __syncthreads();
        }
        sum = sp[tid + k - 1] - (tid ? sp[tid - 1] : 0.f);
    }
    tmp[(size_t)y * (tstep >> 2) + x] = sum;
}

// ---------------- 新内核：一行一 block + 每线程 C 个连续像素滑动 ----------------
// STV：整块路径用 float4 写（C%4==0 且行距/基址 16B 对齐时开），否则标量。
template <int C, bool STV>
__global__ void rows_slide_kernel(const float* __restrict__ src, size_t sstep,
                                  float* __restrict__ tmp, size_t tstep, int W, int k)
{
    extern __shared__ float sm[];
    const int half = k >> 1;
    const int y    = blockIdx.x;
    const int bw   = blockDim.x * C;
    const int x0   = blockIdx.y * bw;
    const int tid  = threadIdx.x;
    const int nld  = min(bw, W - x0) + 2 * half;
    const float* srow = src + (size_t)y * (sstep >> 2);

    for (int i = tid; i < nld; i += blockDim.x)
        sm[i] = srow[mir(x0 - half + i, W)];
    __syncthreads();

    const int gx = x0 + tid * C;
    if (gx >= W) return;
    const int li = tid * C;
    float* trow = tmp + (size_t)y * (tstep >> 2);

    if (gx + C <= W) {
        float s = 0.f;
#pragma unroll 4
        for (int j = 0; j < k; ++j) s += sm[li + j];
        float p[C];
#pragma unroll
        for (int i = 0; i < C; ++i) {
            p[i] = s;
            s += sm[li + i + k] - sm[li + i];
        }
        if (STV && (C % 4 == 0)) {
            float4* q = (float4*)(trow + gx);
#pragma unroll
            for (int i = 0; i < C / 4; ++i)
                q[i] = make_float4(p[4*i], p[4*i+1], p[4*i+2], p[4*i+3]);
        } else {
#pragma unroll
            for (int i = 0; i < C; ++i) trow[gx + i] = p[i];
        }
    } else {
        const int n = W - gx;
        float s = 0.f;
#pragma unroll 4
        for (int j = 0; j < k; ++j) s += sm[li + j];
        for (int i = 0; i < n; ++i) {
            trow[gx + i] = s;
            if (i + 1 < n) s += sm[li + i + k] - sm[li + i];
        }
    }
}

template <int C>
static void launch_new(const float* src, size_t sstep, float* tmp, size_t tstep,
                       int W, int H, int k)
{
    int th = (W + C - 1) / C;
    th = (th + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    const int bw   = th * C;
    const int nbx  = (W + bw - 1) / bw;
    const size_t smem = (size_t)(bw + 2 * (k / 2)) * sizeof(float);
    dim3 g(H, nbx);
    const bool stv = ((tstep % 16) == 0) && (((size_t)tmp % 16) == 0) && (C % 4 == 0);
    if (stv) rows_slide_kernel<C, true ><<<g, th, smem>>>(src, sstep, tmp, tstep, W, k);
    else     rows_slide_kernel<C, false><<<g, th, smem>>>(src, sstep, tmp, tstep, W, k);
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

// 图像：float 值 = 均匀随机整数 0..255（与 benchmark 的 f32 输入同分布）
struct Img {
    float* d = nullptr; size_t step = 0; int W = 0, H = 0;
    std::vector<float> h;
    void make(int w, int hh, unsigned seed) {
        W = w; H = hh; h.resize((size_t)w * hh);
        unsigned st = seed;
        for (auto& v : h) { st = st * 1664525u + 1013904223u; v = (float)((st >> 24) & 255); }
        CKV(cudaMallocPitch(&d, &step, (size_t)w * 4, hh));
        CKV(cudaMemcpy2D(d, step, h.data(), (size_t)w * 4, (size_t)w * 4, hh, cudaMemcpyHostToDevice));
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
                            {127,129},{333,257},{640,480},{720,479},{1279,721},{1280,720},
                            {1920,1080},{3840,2160},{4096,64},{7,400},{1281,721},{512,512}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            for (int k : {15, 31}) {
                Img src; src.make(s.W, s.H, 4242u + (unsigned)(s.W * 3 + s.H + k));
                const int tstep = s.W * 4;
                float* t1 = nullptr; float* t2 = nullptr;
                CKV(cudaMalloc(&t1, (size_t)tstep * s.H));
                CKV(cudaMalloc(&t2, (size_t)tstep * s.H));
                CKV(cudaMemset(t1, 0, (size_t)tstep * s.H));
                CKV(cudaMemset(t2, 0, (size_t)tstep * s.H));
                dim3 g1((s.W + MF_THREADS - 1) / MF_THREADS, s.H);
                old_rows_kernel<<<g1, MF_THREADS>>>(src.d, src.step, t1, tstep, s.W, k);
                launch_new<16>(src.d, src.step, t2, tstep, s.W, s.H, k);
                CKV(cudaDeviceSynchronize());
                std::vector<float> a((size_t)s.W * s.H), b((size_t)s.W * s.H);
                CKV(cudaMemcpy2D(a.data(), (size_t)s.W * 4, t1, (size_t)tstep, (size_t)s.W * 4, s.H, cudaMemcpyDeviceToHost));
                CKV(cudaMemcpy2D(b.data(), (size_t)s.W * 4, t2, (size_t)tstep, (size_t)s.W * 4, s.H, cudaMemcpyDeviceToHost));
                // CPU double 精确参考
                const int half = k >> 1;
                size_t bad = 0, first = (size_t)-1;
                double maxAbs = 0.0, maxAbsOld = 0.0;
                for (int y = 0; y < s.H; ++y) {
                    for (int x = 0; x < s.W; ++x) {
                        double acc = 0.0;
                        for (int j = -half; j <= half; ++j) {
                            int xx = x + j;
                            if (xx < 0) xx = -xx;
                            if (xx >= s.W) xx = 2 * s.W - 2 - xx;
                            xx = std::max(0, std::min(s.W - 1, xx));
                            acc += (double)src.h[(size_t)y * s.W + xx];
                        }
                        const double gn = fabs((double)b[(size_t)y * s.W + x] - acc);
                        const double go = fabs((double)a[(size_t)y * s.W + x] - acc);
                        if (gn > maxAbs) maxAbs = gn;
                        if (go > maxAbsOld) maxAbsOld = go;
                        if (gn > 5e-2) { if (first == (size_t)-1) first = (size_t)y * s.W + x; ++bad; }
                    }
                }
                ++total;
                const bool ok = (bad == 0);
                printf("  %4dx%-4d k=%-2d %s newMaxDiff=%.3g oldMaxDiff=%.3g", s.W, s.H, k, ok ? "[OK]" : "[FAIL]", maxAbs, maxAbsOld);
                if (!ok) printf("  bad=%zu first=(%d,%d)", bad, (int)(first % s.W), (int)(first / s.W));
                printf("\n");
                if (!ok) ++fails;
                cudaFree(t1); cudaFree(t2);
            }
            fflush(stdout);
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MR32_FAIL" : "PROBE_MR32_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "sweep") {
        struct C2 { const char* n; int W, H; };
        const C2 cs[] = {{"640x480",640,480},{"1280x720",1280,720},
                         {"1920x1080",1920,1080},{"3840x2160",3840,2160}};
        for (const C2& c : cs) {
            Img src; src.make(c.W, c.H, 777u);
            const int tstep = c.W * 4;
            float* t1 = nullptr; float* t2 = nullptr;
            CKV(cudaMalloc(&t1, (size_t)tstep * c.H));
            CKV(cudaMalloc(&t2, (size_t)tstep * c.H));
            for (int k : {15, 31}) {
                dim3 g1((c.W + MF_THREADS - 1) / MF_THREADS, c.H);
                double o = timeMin([&] { old_rows_kernel<<<g1, MF_THREADS>>>(src.d, src.step, t1, tstep, c.W, k); }, 3, reps);
                double c4 = timeMin([&] { launch_new<4 >(src.d, src.step, t2, tstep, c.W, c.H, k); }, 3, reps);
                double c8 = timeMin([&] { launch_new<8 >(src.d, src.step, t2, tstep, c.W, c.H, k); }, 3, reps);
                double c16 = timeMin([&] { launch_new<16>(src.d, src.step, t2, tstep, c.W, c.H, k); }, 3, reps);
                // 交错复核（min 口径，排序稳定才认）
                double o2 = 0, n2 = 0;
                abMin([&] { old_rows_kernel<<<g1, MF_THREADS>>>(src.d, src.step, t1, tstep, c.W, k); },
                      [&] { launch_new<16>(src.d, src.step, t2, tstep, c.W, c.H, k); },
                      3, reps, o2, n2);
                // GB/s（读 4 + 写 4 = 8B/px）
                const double bytes = 8.0 * c.W * c.H;
                printf("%-10s k=%-2d old %7.4f (%5.1f GB/s) | C4 %7.4f | C8 %7.4f | C16 %7.4f (%.2fx, %5.1f GB/s) | ab old %7.4f new %7.4f (%.2fx)\n",
                       c.n, k, o, bytes / (o * 1e6), c4, c8, c16, o / c16, bytes / (c16 * 1e6), o2, n2, o2 / n2);
            }
            cudaFree(t1); cudaFree(t2);
            fflush(stdout);
        }
        printf("PROBE_MR32_SWEEP_DONE\n");
        return 0;
    }

    printf("usage: probe_mr32 [check|sweep] [reps]\n");
    return 0;
}
