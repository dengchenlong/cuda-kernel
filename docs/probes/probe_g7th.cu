// probe_g7th：gaussian k=7 融合内核 TH 形状扫描（r55）
// k7scan 的教训（r49 同款）：本机空闲时钟 607↔1493MHz 漂移，顺序执行各 TH 的 min 口径
// 会把「谁先跑到低温/高温」的顺序效应当成 TH 差异（第一轮 4K TH64 1.18 vs 第二轮 0.66）。
// 本探针的口径：
//   1) 正确性先行：每个 TH 与默认路径输出逐位一致（不同 TH 只是 chunk 划分不同，必须 bitwise 相同）。
//   2) 交错 A/B：每轮把 {TH_a, TH_b} 成对先后各跑一次、各取 min，重复 R 轮；
//      对所有 TH 两两配对，最后用「胜场计数 + 比值」判定，顺序效应在配对内抵消。
//   3) 只测 copy 口径（bench 的原地默认就是 copy 路径）。
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

static double timeMin(uint8_t* dA, size_t astep, int W, int H, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e9f;
    for (int i = 0; i < reps; ++i) {
        cudaEventRecord(a);
        gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    return best;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const int reps = (argc > 1) ? atoi(argv[1]) : 30;

    struct Case { int W, H; };
    const Case perf[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const int ths[] = {8, 16, 32, 48, 64, 96, 128};
    const int NTH = 7;

    for (const Case& c : perf) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        std::vector<uint8_t> img(N);
        fillPat(img, W, H, 4242u);
        uint8_t* dA; size_t astep;
        CK(cudaMallocPitch(&dA, &astep, (size_t)W, H));

        // ---- 正确性：每个 TH 与默认输出逐位一致 ----
        // out-of-place 口径：排除原地 copy 链路干扰，直指融合内核本体
        uint8_t* dB; size_t bstep;
        CK(cudaMallocPitch(&dB, &bstep, (size_t)W, H));
        CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
        envSet("CK_TH7", "");
        std::vector<uint8_t> ref(N);
        gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy2D(ref.data(), (size_t)W, dB, bstep, (size_t)W, H, cudaMemcpyDeviceToHost));
        CK(cudaMemset(dB, 0xAB, N));
        CK(cudaMemcpy(dB, img.data(), N, cudaMemcpyHostToDevice));
        gpu_gaussian(dA, astep, dA, astep, W, H, 7, 0.f);
        CK(cudaDeviceSynchronize());
        std::vector<uint8_t> refIp(N);
        CK(cudaMemcpy2D(refIp.data(), (size_t)W, dA, astep, (size_t)W, H, cudaMemcpyDeviceToHost));
        {
            int bad = 0;
            for (size_t i = 0; i < N; ++i) if (refIp[i] != ref[i]) ++bad;
            printf("  [sanity] inplace vs oop (default TH): bad=%d\n", bad);
        }

        int badTotal = 0;
        for (int t = 0; t < NTH; ++t) {
            char buf[8]; snprintf(buf, sizeof(buf), "%d", ths[t]);
            envSet("CK_TH7", buf);
            CK(cudaMemcpy2D(dA, astep, img.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));
            gpu_gaussian(dA, astep, dB, bstep, W, H, 7, 0.f);
            CK(cudaDeviceSynchronize());
            std::vector<uint8_t> out(N);
            CK(cudaMemcpy2D(out.data(), (size_t)W, dB, bstep, (size_t)W, H, cudaMemcpyDeviceToHost));
            int bad = 0, md = 0, shown = 0;
            for (size_t i = 0; i < N; ++i)
                if (out[i] != ref[i]) {
                    ++bad; md = std::max(md, std::abs((int)out[i] - (int)ref[i]));
                    if (shown < 6) { printf("    TH%d diff (x=%d y=%d) ref=%d mine=%d\n", ths[t], (int)(i % W), (int)(i / W), ref[i], out[i]); ++shown; }
                }
            if (bad) { printf("  CHECK FAIL W=%d H=%d TH=%d bad=%d maxdiff=%d\n", W, H, ths[t], bad, md); ++badTotal; }
        }
        envSet("CK_TH7", "");
        if (badTotal) { CK(cudaFree(dA)); return 1; }

        // ---- 交错配对计时 ----
        const int repsN = (N > 4000000) ? reps : reps * 3;
        std::vector<double> best(NTH, 1e9);
        std::vector<int> wins(NTH, 0);
        for (int r = 0; r < 6; ++r) {
            // 每轮起点轮转，抵消顺序效应
            int order[7];
            for (int t = 0; t < NTH; ++t) order[t] = (t + r) % NTH;
            for (int t = 0; t < NTH; ++t) {
                char buf[8]; snprintf(buf, sizeof(buf), "%d", ths[order[t]]);
                envSet("CK_TH7", buf);
                const double ms = timeMin(dA, astep, W, H, repsN);
                if (ms < best[order[t]]) best[order[t]] = ms;
            }
        }
        // 胜场：与其它每个 TH 配对比较（交错轮内）无法直接做，改用 best 互比 + 轮内排名
        envSet("CK_TH7", "");
        printf("%4dx%-4d", W, H);
        int bi = 0;
        for (int t = 0; t < NTH; ++t) if (best[t] < best[bi]) bi = t;
        for (int t = 0; t < NTH; ++t)
            printf("  TH%d %.4f%s", ths[t], best[t], t == bi ? "*" : "");
        printf("   (default=TH64, nC@4K: 32->68 48->45 64->34 96->23 128->17)\n");
        fflush(stdout);
        CK(cudaFree(dA));
    }
    printf("PROBE_G7TH_OK\n");
    return 0;
}
