// probe_mf32：morph-f32 大核（k=15/31）病灶归因 + 滑窗寄存器内核原型
// 背景（PROGRESS r53 后）：u8 大核已换「一线程一条线 smem 走查」（k=31 4K 14.06→4.57ms），
// f32 还是「rows 直加 O(k) smem 读 + cols ring O(k) 本地内存读」——ringv[255] 动态下标
// 落 local memory，是 f32 与 u8 的差距主因候选（erode-f32 k=31 4K 14.25ms / 581 MP/s）。
// 1) asym：gpu_erode_f32 用非对称核拆行遍/列遍（(1,1)/(k,1)/(1,k)/(k,k)）→ 归因。
// 2) check：滑窗寄存器内核 vs 旧路径逐位一致（min/max 结合律保证算术序无关）。
// 3) ab：交错 A/B 取 min，旧两遍 vs 新内核（完整 k×k）。
//
// 新内核设计（滑窗寄存器，非 smem 整线走查——f32 的 48KB smem 上限装不下 2 张 float 线表：
// 128 列 × 4B × 2 张 × 96 行 = 98KB；缩尺寸又会把占用率压到 3~5%）：
//   rows：block=128 线程=128 行，协作把 [x0-R, x0+TB-1+R] 列段载入 smem（线性映射、合并读），
//         一线程一行、窗口 K 值常驻寄存器，滑动 = 移位 + 补新值，输出写 smem 暂存（SOUTP=TB+1
//         错 bank）后线性映射合并写回。
//   cols：一线程一列、TB 行一段，窗口寄存器滑动；global 读天然合并（warp 同行邻列），
//         输出攒在寄存器数组、段尾合并写回（避免一列一写的 32 sector 放大）。零 smem、零屏障。
//   边界：clamp 载入 + 恒等元（min/max 恒等元与 clamp 等价），与旧路径语义一致。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct MinOpF { __device__ float operator()(float a, float b) const { return fminf(a, b); } static __device__ float ident() { return __int_as_float(0x7f800000); } };
struct MaxOpF { __device__ float operator()(float a, float b) const { return fmaxf(a, b); } static __device__ float ident() { return __int_as_float(0xff800000); } };

// ---------------------------------------------------------------- 滑窗寄存器内核
// rows：blockDim = RB = 一 block 的输出行数（一线程一行）；每 block 负责一列段 x0..x0+TB-1
template <typename Op, int K, int TB, int RB>
__global__ void mf32_rows_win_kernel(const float* __restrict__ src, size_t sstep,
                                     float* __restrict__ dst, size_t dstep,
                                     int W, int H)
{
    constexpr int RAD   = K >> 1;
    constexpr int SCW   = TB + K - 1;       // 载入列跨度
    constexpr int SCWP  = SCW | 1;          // 奇数行距错 bank（39 % 32 与 32 互质）
    constexpr int SOUTP = TB + 1;           // 输出暂存行距错 bank（8→9）
    __shared__ float s_in[RB * SCWP];
    __shared__ float s_out[RB * SOUTP];

    const int x0  = blockIdx.x * TB;
    const int y0  = blockIdx.y * RB;
    const int tid = threadIdx.x;
    const Op op;

    // 协作载入（线性映射：相邻线程相邻列，行内合并读；越界列/行填恒等元）
    for (int idx = tid; idx < RB * SCW; idx += RB) {
        const int r  = idx / SCW;
        const int c  = idx - r * SCW;
        const int gy = y0 + r;
        const int gx = x0 - RAD + c;
        s_in[(size_t)r * SCWP + c] = (gy < H && gx >= 0 && gx < W)
                                     ? src[(size_t)gy * (sstep >> 2) + gx] : Op::ident();
    }
    __syncthreads();

    if (y0 + tid < H) {                     // 一线程一行：窗口寄存器滑动
        const float* rp = s_in + (size_t)tid * SCWP;
        float* out      = s_out + (size_t)tid * SOUTP;
        float w[K];
#pragma unroll
        for (int j = 0; j < K; ++j) w[j] = rp[j];
#pragma unroll
        for (int i = 0; i < TB; ++i) {
            float v = w[0];
#pragma unroll
            for (int j = 1; j < K; ++j) v = op(v, w[j]);
            out[i] = v;
            if (i + 1 < TB) {
#pragma unroll
                for (int j = 0; j < K - 1; ++j) w[j] = w[j + 1];
                w[K - 1] = rp[i + K];       // 窗口右移一格的新右端（rp 已 clamp 过）
            }
        }
    }
    __syncthreads();

    const int nc = min(TB, W - x0);
    for (int idx = tid; idx < RB * nc; idx += RB) {   // 线性映射合并写回
        const int r  = idx / nc;
        const int c  = idx - r * nc;
        const int gy = y0 + r;
        if (gy < H) dst[(size_t)gy * (dstep >> 2) + x0 + c] = s_out[(size_t)r * SOUTP + c];
    }
}

// cols：一线程一列，TB 行一段；窗口寄存器滑动；输出攒寄存器、段尾合并写回
template <typename Op, int K, int TB>
__global__ void mf32_cols_win_kernel(const float* __restrict__ src, size_t sstep,
                                     float* __restrict__ dst, size_t dstep,
                                     int W, int H)
{
    constexpr int RAD = K >> 1;
    const int x  = blockIdx.x * blockDim.x + threadIdx.x;
    const int y0 = blockIdx.y * TB;
    if (x >= W) return;
    const Op op;
    const size_t rows = sstep >> 2, orows = dstep >> 2;

    float w[K];
#pragma unroll
    for (int j = 0; j < K; ++j) {
        const int gy = y0 - RAD + j;
        w[j] = (gy >= 0 && gy < H) ? src[(size_t)gy * rows + x] : Op::ident();
    }
    float out[TB];
#pragma unroll
    for (int i = 0; i < TB; ++i) {
        float v = w[0];
#pragma unroll
        for (int j = 1; j < K; ++j) v = op(v, w[j]);
        out[i] = v;
        if (i + 1 < TB) {
#pragma unroll
            for (int j = 0; j < K - 1; ++j) w[j] = w[j + 1];
            const int gy = y0 + i + RAD + 1;      // 窗口下移一格的新下端（clamp）
            w[K - 1] = (gy < H) ? src[(size_t)gy * rows + x] : Op::ident();
        }
    }
#pragma unroll
    for (int i = 0; i < TB; ++i) {
        const int y = y0 + i;
        if (y < H) dst[(size_t)y * orows + x] = out[i];
    }
}

template <typename Op, int K>
static void launchWin(const float* s, size_t sstep, float* d, size_t dstep,
                      int W, int H, float* tmp, size_t tstep, cudaStream_t st = 0)
{
    constexpr int TB = 8, RB = 128, TBC = 16;
    dim3 g1((W + TB - 1) / TB, (H + RB - 1) / RB);
    mf32_rows_win_kernel<Op, K, TB, RB><<<g1, RB, 0, st>>>(s, sstep, tmp, tstep, W, H);
    dim3 g2((W + 255) / 256, (H + TBC - 1) / TBC);
    mf32_cols_win_kernel<Op, K, TBC><<<g2, 256, 0, st>>>(tmp, tstep, d, dstep, W, H);
}

// ---------------------------------------------------------------- 工具
static void fillPat(std::vector<float>& h, int W, int H, unsigned seed)
{
    unsigned st = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            st = st * 1664525u + 1013904223u;
            h[(size_t)y * W + x] = (float)((st >> 8) & 0xffff) / 7.0f;
        }
}

template <typename F>
static double timeMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
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
static void abMin(FA fa, FB fb, int warm, int reps, double& outA, double& outB)
{
    for (int i = 0; i < warm; ++i) { fa(); fb(); }
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float bestA = 1e9f, bestB = 1e9f;
    for (int i = 0; i < reps; ++i) {
        float ms = 0.f;
        cudaEventRecord(a); fa(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < bestA) bestA = ms;
        cudaEventRecord(a); fb(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < bestB) bestB = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    outA = bestA; outB = bestB;
}

struct FImg {
    std::vector<float> h;
    float* d = nullptr; size_t step = 0;
    int W = 0, H = 0;
    void make(int w, int hh, unsigned seed) {
        W = w; H = hh;
        h.resize((size_t)w * hh);
        fillPat(h, w, hh, seed);
        CKV(cudaMallocPitch(&d, &step, (size_t)w * 4, hh));
        CKV(cudaMemcpy2D(d, step, h.data(), (size_t)w * 4, (size_t)w * 4, hh, cudaMemcpyHostToDevice));
    }
    ~FImg() { if (d) cudaFree(d); }
};

static float* g_tmp = nullptr;
static size_t g_tstep = 0;

template <typename Op, int K>
static void newFull(const float* s, size_t sstep, float* d, size_t dstep, int W, int H)
{
    launchWin<Op, K>(s, sstep, d, dstep, W, H, g_tmp, g_tstep);
}

template <typename Op>
static void newFullK(int k, const float* s, size_t sstep, float* d, size_t dstep, int W, int H)
{
    if (k == 15)      newFull<Op, 15>(s, sstep, d, dstep, W, H);
    else if (k == 31) newFull<Op, 31>(s, sstep, d, dstep, W, H);
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "asym";
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    // 最大尺寸的紧排暂存（4K float，宽度取 4096 覆盖 check 的 4096x64）
    CK(cudaMallocPitch(&g_tmp, &g_tstep, (size_t)4096 * 4, 2160));

    struct Case { const char* name; int W, H; };
    const Case perf[] = {{"640x480", 640, 480}, {"1280x720", 1280, 720},
                         {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};

    if (mode == "asym") {
        // 行遍/列遍拆分（旧路径归因）
        for (const Case& c : perf) {
            FImg src, dst;
            src.make(c.W, c.H, 999u);
            dst.make(c.W, c.H, 1u);
            printf("=== %s ===\n", c.name);
            for (int k : {15, 31}) {
                double t11 = timeMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, 1, 1); }, 5, reps);
                double tR  = timeMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, 1); }, 5, reps);
                double tC  = timeMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, 1, k); }, 5, reps);
                double tKK = timeMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                printf("  k=%-2d  (1,1) %8.4f  (%d,1) %8.4f  (1,%d) %8.4f  (%d,%d) %8.4f\n",
                       k, t11, k, tR, k, tC, k, k, tKK);
            }
        }
        return 0;
    }

    if (mode == "check") {
        struct S { int W, H; };
        const S shapes[] = {{16, 1}, {16, 2}, {17, 3}, {31, 17}, {33, 33}, {64, 64}, {100, 100},
                            {127, 129}, {160, 17}, {640, 480}, {720, 479}, {1280, 720},
                            {1920, 1080}, {3840, 2160}, {333, 257}, {4096, 64}, {7, 400}};
        int fails = 0, total = 0;
        for (int k : {15, 31}) {
            for (const S& s : shapes) {
                const int W = s.W, H = s.H;
                for (int opn = 0; opn < 2; ++opn) {
                    FImg src, a;
                    src.make(W, H, 4242u + (unsigned)(W * 3 + H + k));
                    a.make(W, H, 1u);
                    // 参照：旧路径（smoke 已 vs OpenCV 验证过）
                    if (opn == 0) gpu_erode_f32(src.d, src.step, a.d, a.step, W, H, k, k);
                    else          gpu_dilate_f32(src.d, src.step, a.d, a.step, W, H, k, k);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> ref((size_t)W * H);
                    CK(cudaMemcpy2D(ref.data(), (size_t)W * 4, a.d, a.step, (size_t)W * 4, H, cudaMemcpyDeviceToHost));

                    CK(cudaMemset2D(a.d, a.step, 0xCD, (size_t)W * 4, H));
                    if (opn == 0) newFullK<MinOpF>(k, src.d, src.step, a.d, a.step, W, H);
                    else          newFullK<MaxOpF>(k, src.d, src.step, a.d, a.step, W, H);
                    CK(cudaDeviceSynchronize());
                    std::vector<float> got((size_t)W * H);
                    CK(cudaMemcpy2D(got.data(), (size_t)W * 4, a.d, a.step, (size_t)W * 4, H, cudaMemcpyDeviceToHost));

                    const size_t bytes = (size_t)W * H * 4;
                    const int bad = memcmp(ref.data(), got.data(), bytes) ? 1 : 0;
                    size_t firstbad = (size_t)-1;
                    if (bad) for (size_t i = 0; i < (size_t)W * H; ++i)
                        if (memcmp(&ref[i], &got[i], 4)) { firstbad = i; break; }
                    ++total;
                    const bool ok = !bad;
                    printf("  k=%-2d %4dx%-4d %s %s", k, W, H, opn ? "dilate" : "erode", ok ? "[OK]" : "[FAIL]");
                    if (!ok) { printf("  firstbad %zu (%.6g vs %.6g)", firstbad, ref[firstbad], got[firstbad]); ++fails; }
                    printf("\n");
                }
            }
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MF32_FAIL" : "PROBE_MF32_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "ab") {
        for (const Case& c : perf) {
            printf("=== %s ===\n", c.name);
            for (int k : {15, 31}) {
                FImg src, dst;
                src.make(c.W, c.H, 999u);
                dst.make(c.W, c.H, 1u);
                double tOldE = 0, tNewE = 0, tOldD = 0, tNewD = 0;
                if (k == 15) {
                    abMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); },
                          [&] { newFull<MinOpF, 15>(src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, tOldE, tNewE);
                    abMin([&] { gpu_dilate_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); },
                          [&] { newFull<MaxOpF, 15>(src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, tOldD, tNewD);
                } else {
                    abMin([&] { gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); },
                          [&] { newFull<MinOpF, 31>(src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, tOldE, tNewE);
                    abMin([&] { gpu_dilate_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); },
                          [&] { newFull<MaxOpF, 31>(src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, tOldD, tNewD);
                }
                printf("  erode  k=%-2d old %8.4f  win %8.4f  (%.2fx)\n", k, tOldE, tNewE, tOldE / tNewE);
                printf("  dilate k=%-2d old %8.4f  win %8.4f  (%.2fx)\n", k, tOldD, tNewD, tOldD / tNewD);
            }
        }
        return 0;
    }

    printf("usage: probe_mf32 [asym|check|ab] [reps]\n");
    return 0;
}
