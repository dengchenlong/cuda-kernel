// probe_mcol：morph 大核（k=15/31）性能归因 + 分块列遍（smem tile）原型
// 1) asym：用非对称核把行遍/列遍拆开计时 —— (1,1) 基线、(k,1) 只放大行遍、(1,k) 只放大列遍。
//    怀疑点：vhg_cols_kernel 的 blockIdx.y = 列号、threadIdx.x = 行号，
//    所以一个 warp 的 32 个 lane 读的是「同一列的 32 个连续行」= 步长 sstep → 每 lane 各自一个
//    32B sector、只用 1 字节 ⇒ 32 倍写放大。行遍是线程沿 x 走，天然合并。
// 2) tile：分块列遍原型（每线程独占一列、把 [A-R, B+R) 行装进 smem，van Herk 双趟 O(1)/px 出结果），
//    与「信任路径」gpu_erode(kw=1, kh=k) 的输出逐位比对，再交错 A/B 计时。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

struct MaxOp { __device__ __host__ uint8_t operator()(uint8_t a, uint8_t b) const { return a > b ? a : b; } static __device__ __host__ uint8_t ident() { return 0; } };
struct MinOp { __device__ __host__ uint8_t operator()(uint8_t a, uint8_t b) const { return a < b ? a : b; } static __device__ __host__ uint8_t ident() { return 255; } };

// ---------------------------------------------------------------- 分块列遍原型
// 每线程一列；smem 里放本列 [A-R, B+R) 的输入（sx）与 van Herk 的 U（su，块内前缀）。
// U[j] = op over [blockstart(j) .. j]（块长 = k，按全局行号对齐）
// D[j] = op over [j .. blockend(j)]，输出 y 的窗口 = [y-R, y-R+k-1]，
// 结果 = op(D[y-R], U[y-R+k-1])，两者落在同一 k 块内时都恰等于整块 → 精确。
// 越界行按 clamp 载入（min/max 的恒等元与 clamp 等价，与原内核语义一致）。
template <typename Op, int TB, int TC, int RMAX>
__global__ void cols_tile_kernel(const uint8_t* __restrict__ src, size_t sstep,
                                 uint8_t* __restrict__ dst, size_t dstep,
                                 int W, int H, int k)
{
    constexpr int TR  = TB + 2 * RMAX;
    constexpr int STR = TR + 4;

    __shared__ uint8_t sx[TC * STR];
    __shared__ uint8_t su[TC * STR];

    const int t = threadIdx.x;
    const int x = blockIdx.x * TC + t;
    if (x >= W) return;

    const Op op;
    const int R  = k >> 1;
    const int A  = blockIdx.y * TB;
    const int B  = min(H, A + TB);
    const int r0 = A - R;                     // 局部行 0 对应的全局行
    uint8_t* cx = &sx[t * STR];
    uint8_t* cu = &su[t * STR];

    for (int r = 0; r < TR; ++r) {
        int gy = r0 + r;
        gy = min(max(gy, 0), H - 1);
        cx[r] = src[(size_t)gy * sstep + x];
    }

    // U：升序单趟（块首重置）
    {
        uint8_t u = Op::ident();
        for (int r = 0; r < TR; ++r) {
            const int j = r0 + r;
            int m = j % k; if (m < 0) m += k;
            u = (m == 0) ? cx[r] : op(u, cx[r]);
            cu[r] = u;
        }
    }

    // D[y-R+1] 的初值（第一个输出 y=B-1 需要它）：从局部位 bA 直接滚到块尾
    uint8_t d = Op::ident();
    {
        int jj = B - R;
        for (int r = B - A; r < TR; ++r, ++jj) {
            d = op(d, cx[r]);
            int m = jj % k; if (m < 0) m += k;
            if (m == k - 1) break;
        }
    }

    for (int y = B - 1; y >= A; --y) {
        const int j  = y - R;
        const int rj = y - A;                 // = j - r0
        int m = j % k; if (m < 0) m += k;
        const uint8_t dj = (m == k - 1) ? cx[rj] : op(cx[rj], d);
        const uint8_t v  = op(dj, cu[rj + k - 1]);
        dst[(size_t)y * dstep + x] = v;
        d = dj;
    }
}

static void fillPat(std::vector<uint8_t>& h, int W, int H, unsigned seed)
{
    unsigned st = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            st = st * 1664525u + 1013904223u;
            h[(size_t)y * W + x] = (uint8_t)((st >> 16) ^ (x * 37 + y * 91));
        }
}

template <typename F>
static double timeMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CUDA_CHECK(cudaDeviceSynchronize());
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
    CUDA_CHECK(cudaDeviceSynchronize());
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

struct Img {
    std::vector<uint8_t> h;
    uint8_t* d = nullptr;
    size_t step = 0;
    int W = 0, H = 0;
    void make(int w, int hh, unsigned seed) {
        W = w; H = hh;
        h.resize((size_t)w * hh);
        fillPat(h, w, hh, seed);
        CUDA_CHECK(cudaMallocPitch(&d, &step, (size_t)w, hh));
        CUDA_CHECK(cudaMemcpy2D(d, step, h.data(), (size_t)w, (size_t)w, hh, cudaMemcpyHostToDevice));
    }
    ~Img() { if (d) cudaFree(d); }
};

// 分块列遍的启动器（RMAX 按 k 分档）
template <typename Op>
static void launchTile(const uint8_t* s, size_t sstep, uint8_t* d, size_t dstep,
                       int W, int H, int k, cudaStream_t stream)
{
    constexpr int TB = 64, TC = 128;
    if (k <= 17) {
        dim3 g((W + TC - 1) / TC, (H + TB - 1) / TB);
        cols_tile_kernel<Op, TB, TC, 8><<<g, TC, 0, stream>>>(s, sstep, d, dstep, W, H, k);
    } else {
        dim3 g((W + TC - 1) / TC, (H + TB - 1) / TB);
        cols_tile_kernel<Op, TB, TC, 16><<<g, TC, 0, stream>>>(s, sstep, d, dstep, W, H, k);
    }
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "asym";
    const int reps = (argc > 2) ? atoi(argv[2]) : 30;
    const char* acc = getenv("CK_ACC");
    const bool accOn = acc && acc[0] == '1';

    struct Case { const char* name; int W, H; };
    const Case perf[] = {{"640x480", 640, 480}, {"1280x720", 1280, 720},
                         {"1920x1080", 1920, 1080}, {"3840x2160", 3840, 2160}};

    if (mode == "asym") {
        // 行遍/列遍拆分：(1,1) 基线、(k,1) 行遍 k、(1,k) 列遍 k
        for (const Case& c : perf) {
            Img src, dst;
            src.make(c.W, c.H, 999u);
            dst.make(c.W, c.H, 1u);
            printf("=== %s ===\n", c.name);
            for (int k : {1, 15, 31}) {
                double t11 = timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, 1, 1); }, 5, reps);
                double tR  = (k == 1) ? t11 : timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, 1); }, 5, reps);
                double tC  = (k == 1) ? t11 : timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, 1, k); }, 5, reps);
                double tKK = (k == 1) ? t11 : timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                printf("  k=%-2d  (1,1) %7.4f  (%d,1) %7.4f  (1,%d) %7.4f  (%d,%d) %7.4f\n",
                       k, t11, k, tR, k, tC, k, k, tKK);
            }
        }
        return 0;
    }

    if (mode == "check") {
        struct S { int W, H; };
        const S shapes[] = {{16, 1}, {16, 2}, {17, 3}, {31, 17}, {64, 64}, {100, 100},
                            {127, 129}, {160, 17}, {640, 480}, {720, 479}, {1280, 720},
                            {1920, 1080}, {3840, 2160}, {333, 257}, {4096, 64}};
        int fails = 0;
        for (int k : {9, 15, 17, 21, 31, 63}) {
            for (const S& s : shapes) {
                const int W = s.W, H = s.H;
                Img src, a, b;
                src.make(W, H, 4242u + (unsigned)(W * 3 + H + k));
                a.make(W, H, 1u);
                b.make(W, H, 1u);
                gpu_erode(src.d, src.step, a.d, a.step, W, H, 1, k);      // 参照（VHG 列遍）
                gpu_dilate(src.d, src.step, b.d, b.step, W, H, 1, k);
                std::vector<uint8_t> ra((size_t)W * H), rb((size_t)W * H);
                CK(cudaMemcpy2D(ra.data(), (size_t)W, a.d, a.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy2D(rb.data(), (size_t)W, b.d, b.step, (size_t)W, H, cudaMemcpyDeviceToHost));

                Img ea, eb;
                ea.make(W, H, 1u); eb.make(W, H, 1u);
                launchTile<MinOp>(src.d, src.step, ea.d, ea.step, W, H, k, 0);
                launchTile<MaxOp>(src.d, src.step, eb.d, eb.step, W, H, k, 0);
                CK(cudaDeviceSynchronize());
                std::vector<uint8_t> oa((size_t)W * H), ob((size_t)W * H);
                CK(cudaMemcpy2D(oa.data(), (size_t)W, ea.d, ea.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy2D(ob.data(), (size_t)W, eb.d, eb.step, (size_t)W, H, cudaMemcpyDeviceToHost));

                int badE = 0, mdE = 0, badD = 0, mdD = 0;
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    int de = std::abs((int)oa[i] - (int)ra[i]);
                    int dd = std::abs((int)ob[i] - (int)rb[i]);
                    if (de) { ++badE; mdE = std::max(mdE, de); }
                    if (dd) { ++badD; mdD = std::max(mdD, dd); }
                }
                const bool ok = !badE && !badD;
                printf("  k=%-2d %4dx%-4d %s", k, W, H, ok ? "[OK]" : "[FAIL]");
                if (!ok) { printf("  erode bad=%d md=%d | dilate bad=%d md=%d", badE, mdE, badD, mdD); ++fails; }
                printf("\n");
            }
        }
        printf("\n%s (fail=%d)\n", fails ? "PROBE_MCOL_FAIL" : "PROBE_MCOL_ALL_OK", fails);
        return fails ? 1 : 0;
    }

    if (mode == "check2d") {
        // 新（走查）vs 旧（VHG）逐位一致：二维大核全组合
        struct S { int W, H; };
        const S shapes[] = {{16, 1}, {16, 2}, {17, 3}, {31, 17}, {33, 33}, {64, 64}, {100, 100},
                            {127, 129}, {160, 17}, {640, 480}, {720, 479}, {1280, 720},
                            {1920, 1080}, {3840, 2160}, {333, 257}, {4096, 64}, {7, 400}};
        int fails = 0;
        for (int k : {9, 13, 15, 21, 31, 33}) {
            for (const S& s : shapes) {
                const int W = s.W, H = s.H;
                Img src, a, b;
                src.make(W, H, 4242u + (unsigned)(W * 3 + H + k));
                a.make(W, H, 1u); b.make(W, H, 1u);
                std::vector<uint8_t> ra((size_t)W * H), rb((size_t)W * H);
                _putenv_s("CK_MORPH_LEGACY", "1");
                gpu_erode(src.d, src.step, a.d, a.step, W, H, k, k);
                gpu_dilate(src.d, src.step, b.d, b.step, W, H, k, k);
                _putenv_s("CK_MORPH_LEGACY", "");
                CK(cudaMemcpy2D(ra.data(), (size_t)W, a.d, a.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy2D(rb.data(), (size_t)W, b.d, b.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                gpu_erode(src.d, src.step, a.d, a.step, W, H, k, k);
                gpu_dilate(src.d, src.step, b.d, b.step, W, H, k, k);
                std::vector<uint8_t> oa((size_t)W * H), ob((size_t)W * H);
                CK(cudaMemcpy2D(oa.data(), (size_t)W, a.d, a.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy2D(ob.data(), (size_t)W, b.d, b.step, (size_t)W, H, cudaMemcpyDeviceToHost));
                int badE = 0, mdE = 0, badD = 0, mdD = 0;
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    int de = std::abs((int)oa[i] - (int)ra[i]);
                    int dd = std::abs((int)ob[i] - (int)rb[i]);
                    if (de) { ++badE; mdE = std::max(mdE, de); }
                    if (dd) { ++badD; mdD = std::max(mdD, dd); }
                }
                const bool ok = !badE && !badD;
                printf("  k=%-2d %4dx%-4d %s", k, W, H, ok ? "[OK]" : "[FAIL]");
                if (!ok) { printf("  erode bad=%d md=%d | dilate bad=%d md=%d", badE, mdE, badD, mdD); ++fails; }
                printf("\n");
            }
        }
        printf("\n%s (fail=%d)\n", fails ? "PROBE_MCOL_2D_FAIL" : "PROBE_MCOL_2D_OK", fails);
        return fails ? 1 : 0;
    }

    if (mode == "ab") {
        // 新旧对比：CK_MORPH_LEGACY=1 走 VHG，默认走 r53 走查内核（交错 A/B，取 min）
        const bool legacyFirst = (argc > 3) ? (atoi(argv[3]) != 0) : true;
        for (const Case& c : perf) {
            printf("=== %s ===\n", c.name);
            for (int k : {15, 31}) {
                Img src, dst;
                src.make(c.W, c.H, 999u);
                dst.make(c.W, c.H, 1u);
                double tOld = 0, tNew = 0;
                _putenv_s("CK_MORPH_LEGACY", "1");
                double a1 = timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                _putenv_s("CK_MORPH_LEGACY", "");
                double b1 = timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                _putenv_s("CK_MORPH_LEGACY", "1");
                double a2 = timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                _putenv_s("CK_MORPH_LEGACY", "");
                double b2 = timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps);
                tOld = std::min(a1, a2);
                tNew = std::min(b1, b2);
                printf("  erode k=%-2d VHG %8.4f  line %8.4f  (%.2fx)  %s\n", k, tOld, tNew,
                       tOld / tNew, legacyFirst ? "" : "");
            }
        }
        return 0;
    }

    if (mode == "time") {
        for (const Case& c : perf) {
            printf("=== %s ===\n", c.name);
            for (int k : {15, 31}) {
                Img src, dst;
                src.make(c.W, c.H, 999u);
                dst.make(c.W, c.H, 1u);
                double tLegacy = 0, tTile = 0;
                abMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, 1, k); },
                      [&] { launchTile<MinOp>(src.d, src.step, dst.d, dst.step, c.W, c.H, k, 0); },
                      5, reps, tLegacy, tTile);
                printf("  k=%-2d cols: legacy %8.4f  tile %8.4f  (%.2fx)   | full: legacy-rows+cols %8.4f\n",
                       k, tLegacy, tTile, tLegacy / tTile,
                       timeMin([&] { gpu_erode(src.d, src.step, dst.d, dst.step, c.W, c.H, k, k); }, 5, reps));
            }
        }
        return 0;
    }

    printf("usage: probe_mcol [asym|check|time] [reps]\n");
    return 0;
}