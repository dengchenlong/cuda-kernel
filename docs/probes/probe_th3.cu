// probe_th3：mean k=3 / morph k=3 原地 stash 路径的 chunk 高度（TH）扫描（r49）
// 动机：1080p 现在 TH=16 → 68 个 chunk → stash 填充 139 行；TH=32 只要 71 行（fill 砍半），
//       但块数腰斩会掉并行度 —— 只能实测。
// 用法：probe_th3 sweep        交错 A/B，min 口径，3 轮，TH ∈ {8,16,32,64}
//       probe_th3 time         实际发布口径（不设 CK_TH3）四档尺寸 in-place vs out-of-place
// 口径：benchmark 同款 —— 原地反复跑、计时内不重传数据。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static void envSet(const char* k, const char* v) { _putenv_s(k, v); }

static void fillPat(std::vector<uint8_t>& h, int W, int H, unsigned seed)
{
    unsigned st = seed;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            st = st * 1664525u + 1013904223u;
            h[(size_t)y * W + x] = (uint8_t)((st >> 16) ^ (x * 37 + y * 91));
        }
}

static void runOp(int op, uint8_t* s, size_t ss, uint8_t* d, size_t ds, int W, int H)
{
    if (op == 0)      gpu_mean(s, ss, d, ds, W, H, 3, 3);
    else if (op == 1) gpu_erode(s, ss, d, ds, W, H, 3, 3);
    else              gpu_dilate(s, ss, d, ds, W, H, 3, 3);
}
static const char* opName(int op) { return op == 0 ? "mean" : (op == 1 ? "erode" : "dilate"); }

template <typename F>
static double timeitMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "sweep";

    if (mode == "time") {
        const int sizes[][2] = {{640,480}, {1280,720}, {1920,1080}, {3840,2160}};
        for (auto& sz : sizes) {
            const int W = sz[0], H = sz[1];
            std::vector<uint8_t> img((size_t)W * H);
            fillPat(img, W, H, 999u + W);
            uint8_t* dA; size_t astep; uint8_t* dB; size_t bstep;
            CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
            CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            const int reps = (H >= 1080) ? 60 : 150;
            for (int op = 0; op < 3; ++op) {
                const double oop = timeitMin([&] { runOp(op, dA, astep, dB, bstep, W, H); }, 5, reps);
                const double ip  = timeitMin([&] { runOp(op, dA, astep, dA, astep, W, H); }, 5, reps);
                printf("  %4dx%-4d %-6s oop=%.4f ip=%.4f ratio=%.2f\n", W, H, opName(op), oop, ip, ip / oop);
            }
            CK(cudaFree(dA)); CK(cudaFree(dB));
        }
        return 0;
    }

    // sweep：TH ∈ {8,16,32,64} 的原地计时（同一进程内交错，各取 min）
    const int sizes[][2] = {{640,480}, {1280,720}, {1920,1080}, {3840,2160}};
    const int ths[] = {8, 16, 32, 64};
    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        std::vector<uint8_t> img((size_t)W * H);
        fillPat(img, W, H, 4242u + W);
        uint8_t* dA; size_t astep;
        CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
        CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
        const int reps = (H >= 2000) ? 70 : (H >= 1000 ? 150 : 250);
        for (int op = 0; op < 3; ++op) {
            double best[4];
            for (int i = 0; i < 4; ++i) best[i] = 1e9;
            for (int r = 0; r < 3; ++r) {
                for (int i = 0; i < 4; ++i) {
                    char buf[8]; snprintf(buf, sizeof(buf), "%d", ths[i]);
                    envSet("CK_TH3", buf);
                    const double t = timeitMin([&] { runOp(op, dA, astep, dA, astep, W, H); }, 3, reps);
                    if (t < best[i]) best[i] = t;
                }
            }
            envSet("CK_TH3", "");
            const int bi = (int)(std::min_element(best, best + 4) - best);
            const int di = (H >= 1600) ? 3 : 1;      // 当前发布口径：4K TH=64、1080p TH=16
            printf("  %4dx%-4d %-6s", W, H, opName(op));
            for (int i = 0; i < 4; ++i) printf("  TH%-2d %.4f", ths[i], best[i]);
            printf("   best=TH%d vs TH%d(默认) %.3f\n", ths[bi], ths[di], best[di] / best[bi]);
            fflush(stdout);
        }
        CK(cudaFree(dA));
    }
    return 0;
}