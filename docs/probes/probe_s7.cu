// probe_s7：gaussian k=7 原地 stash 路径（r48）—— 正确性 + A/B 计时
// 1) check：原地（stash）输出必须与「同内核 out-of-place」逐位一致（out-of-place 读原始 src，是原子 truth），
//    每种形状重复 N 轮（抓非确定性竞态）。setenv 切换 CK_FORCE_STASH / CK_NO_STASH 覆盖两条原地路径。
// 2) time：四档尺寸 in-place(stash) vs in-place(copy) vs out-of-place 中位计时。
// 3) chunk：CK_TH7 遍历 TH=8/16/32/64 的正确性（覆盖 chunk 数与末块组合）。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <string>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

// Windows/MSVC 没有 setenv/unsetenv
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

template <typename F>
static double timeit(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    CK(cudaDeviceSynchronize());
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    std::vector<float> t;
    t.reserve(reps);
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        t.push_back(ms);
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

template <typename F>
static double timeitMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    cudaDeviceSynchronize();
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

// 交错 A/B：每对里先后各跑一次，各取 min（外部负载只加不减 → min 是最干净的估计）
template <typename FA, typename FB>
static void abMin(FA fa, FB fb, int warm, int reps, double& outA, double& outB)
{
    for (int i = 0; i < warm; ++i) { fa(); fb(); }
    cudaDeviceSynchronize();
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

// 一轮：原地跑一次，与 ref（out-of-place 结果）逐位比对；返回 bad 数，并打印前若干差异
static int roundTrip(int W, int H, const std::vector<uint8_t>& img, const std::vector<uint8_t>& ref,
                     uint8_t* dA, size_t astep, std::vector<uint8_t>& out, const char* tag, int showN)
{
    CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
    gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f);
    CK(cudaMemcpy2D(out.data(), (size_t)W, dA, astep, (size_t)W, H, cudaMemcpyDeviceToHost));
    int bad = 0, md = 0, shown = 0;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const int d = std::abs((int)out[(size_t)y * W + x] - (int)ref[(size_t)y * W + x]);
            if (d) {
                ++bad; md = std::max(md, d);
                if (shown < showN) {
                    printf("      %s diff (y=%d x=%d) ref=%d mine=%d  y%%64=%d y%%16=%d\n", tag, y, x,
                           ref[(size_t)y * W + x], out[(size_t)y * W + x], y % 64, y % 16);
                    ++shown;
                }
            }
        }
    if (bad) printf("      %s bad=%d maxdiff=%d\n", tag, bad, md);
    return bad;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "check";
    const int rounds = (argc > 2) ? atoi(argv[2]) : 4;

    struct Case { int W, H; };
    const Case shapes[] = {
        {16, 4}, {24, 5}, {64, 4}, {100, 100}, {127, 129}, {160, 17},
        {640, 480}, {720, 479}, {1280, 720}, {1920, 1080}, {3840, 2160}, {4096, 64}
    };

    if (mode == "check") {
        int fails = 0;
        for (int forced = 0; forced < 2; ++forced) {
            if (forced) { envSet("CK_FORCE_STASH", "1"); }
            else        { envSet("CK_FORCE_STASH", ""); envSet("CK_NO_STASH", ""); }
            printf("=== check %s ===\n", forced ? "CK_FORCE_STASH=1 (所有形状都走 stash)" : "默认阈值 (H>=1000 才走 stash)");
            for (const Case& c : shapes) {
                const int W = c.W, H = c.H;
                const size_t N = (size_t)W * H;
                std::vector<uint8_t> img(N), ref(N), out(N);
                fillPat(img, W, H, 12345u + (unsigned)(W * 7 + H));
                uint8_t* dA; size_t astep;
                uint8_t* dB; size_t bstep;
                CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
                CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
                CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
                gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f);       // 原子 truth
                CK(cudaMemcpy2D(ref.data(), (size_t)W, dB, bstep, (size_t)W, H, cudaMemcpyDeviceToHost));
                int caseBad = 0;
                for (int r = 0; r < rounds; ++r) {
                    char tag[64];
                    snprintf(tag, sizeof(tag), "r%d", r);
                    caseBad += roundTrip(W, H, img, ref, dA, astep, out, tag, 6);
                }
                printf("  %4dx%-4d rounds=%d %s\n", W, H, rounds, caseBad ? "[FAIL]" : "[OK]");
                if (caseBad) { ++fails; }
                CK(cudaFree(dA)); CK(cudaFree(dB));
            }
        }
        printf("\n%s (fail=%d)\n", fails ? "PROBE_S7_FAIL" : "PROBE_S7_ALL_OK", fails);
        return fails ? 1 : 0;
    }

    if (mode == "chunk") {
        // TH 覆盖：CK_TH7 由外部设置；这里只验证原地 vs 原地外逐位一致
        const char* ths = getenv("CK_TH7");
        printf("=== chunk check TH=%s ===\n", ths ? ths : "(default)");
        envSet("CK_FORCE_STASH", "1");
        int fails = 0;
        for (const Case& c : shapes) {
            const int W = c.W, H = c.H;
            const size_t N = (size_t)W * H;
            std::vector<uint8_t> img(N), ref(N), out(N);
            fillPat(img, W, H, 777u + (unsigned)(W * 13 + H));
            uint8_t* dA; size_t astep; uint8_t* dB; size_t bstep;
            CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
            CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f);
            CK(cudaMemcpy2D(ref.data(), (size_t)W, dB, bstep, (size_t)W, H, cudaMemcpyDeviceToHost));
            int bad = 0;
            for (int r = 0; r < rounds; ++r) bad += roundTrip(W, H, img, ref, dA, astep, out, "th", 4);
            printf("  %4dx%-4d %s\n", W, H, bad ? "[FAIL]" : "[OK]");
            if (bad) ++fails;
            CK(cudaFree(dA)); CK(cudaFree(dB));
        }
        printf("\n%s (fail=%d)\n", fails ? "PROBE_S7_CHUNK_FAIL" : "PROBE_S7_CHUNK_OK", fails);
        return fails ? 1 : 0;
    }

    if (mode == "time") {
        const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
        for (const Case& c : perf) {
            const int W = c.W, H = c.H;
            const size_t N = (size_t)W * H;
            std::vector<uint8_t> img(N);
            fillPat(img, W, H, 999u);
            uint8_t* dA; size_t astep; uint8_t* dB; size_t bstep;
            CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
            CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            const int reps = (N > 4000000) ? 60 : 250;
            // benchmark 口径：计时内不重传数据（原地反复跑，数据渐趋平台化）—— 与 bench 一致
            auto runIp = [&] { gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f); };
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            envSet("CK_FORCE_STASH", "1");
            const double ipStash = timeit(runIp, 5, reps);
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            envSet("CK_FORCE_STASH", "");
            envSet("CK_NO_STASH", "1");
            const double ipCopy = timeit(runIp, 5, reps);
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            envSet("CK_NO_STASH", "");
            const double ipDefault = timeit(runIp, 5, reps);
            const double op = timeit([&] { gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f); }, 5, reps);
            printf("%4dx%-4d stash %.4f | copy %.4f | default %.4f | out-of-place %.4f | stash/copy %.2f\n",
                   W, H, ipStash, ipCopy, ipDefault, op, ipStash / ipCopy);
            CK(cudaFree(dA)); CK(cudaFree(dB));
        }
        return 0;
    }
    if (mode == "sweep") {
        // 交错 A/B 取 min：外部负载（同机 WSL 训练任务）只加不减，min 是最干净的估计
        const Case perf[] = {{1920, 1080}, {3840, 2160}};
        for (const Case& c : perf) {
            const int W = c.W, H = c.H;
            const size_t N = (size_t)W * H;
            std::vector<uint8_t> img(N);
            fillPat(img, W, H, 4242u);
            uint8_t* dA; size_t astep; uint8_t* dB; size_t bstep;
            CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));
            CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            const int reps = (N > 4000000) ? 40 : 120;
            printf("%4dx%-4d\n", W, H);
            envSet("CK_TH7", "");
            for (int r = 0; r < 3; ++r) {
                envSet("CK_FORCE_STASH", "1");
                envSet("CK_NO_STASH", "");
                double tStash = 0, tCopy = 0;
                abMin([&] { gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f); },
                      [&] { envSet("CK_NO_STASH", "1"); envSet("CK_FORCE_STASH", "");
                            gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f);
                            envSet("CK_NO_STASH", ""); envSet("CK_FORCE_STASH", "1"); },
                      3, reps, tStash, tCopy);
                const double op = timeitMin([&] { gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f); }, 5, reps);
                printf("   round%d stash %.4f | copy %.4f | stash/copy %.3f | out-of-place %.4f\n",
                       r, tStash, tCopy, tStash / tCopy, op);
            }
            envSet("CK_FORCE_STASH", "");
            envSet("CK_NO_STASH", "");
            CK(cudaFree(dA)); CK(cudaFree(dB));
        }
        return 0;
    }

    printf("usage: probe_s7 [check|chunk|time|sweep] [rounds]\n");
    return 2;
}
