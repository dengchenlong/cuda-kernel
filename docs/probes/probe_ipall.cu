// probe_ipall.cu — 「原地口径」体检：对每个算子比较 in-place 与 out-of-place 的结果。
// 动机：mean k=3 的旧路径原地有跨 block halo 竞态（smoke 只测 out-of-place 所以没抓到）。
// 本探针不依赖 CPU 参照，只比两条路径是否一致 —— 一致不代表对，但不一致就一定错。
#include <opencv2/core.hpp>
#include "ops.h"
#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main()
{
    CK(cudaFree(0));
    const int sizes[][2] = {{640, 480}, {1920, 1080}, {3840, 2160}};
    int bad = 0;

    for (auto& sz : sizes) {
        const int W = sz[0], H = sz[1];
        const size_t N = (size_t)W * H;
        cv::Mat rnd(H, W, CV_8UC1);
        unsigned st = 31337 + W;
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) { st = st * 1664525u + 1013904223u; rnd.at<uchar>(y, x) = (uchar)(st >> 16); }

        uint8_t *a = nullptr, *b = nullptr; size_t sa = 0, sb = 0;
        CK(cudaMallocPitch(&a, &sa, W, H));
        CK(cudaMallocPitch(&b, &sb, W, H));
        std::vector<uint8_t> h1(N), h2(N);

        auto check = [&](const char* name, const std::function<void(uint8_t*, size_t, uint8_t*, size_t)>& fn) {
            CK(cudaMemcpy2D(a, sa, rnd.data, rnd.step, W, H, cudaMemcpyHostToDevice));
            CK(cudaMemcpy2D(b, sb, rnd.data, rnd.step, W, H, cudaMemcpyHostToDevice));
            fn(a, sa, b, sb);                 // out-of-place
            CK(cudaDeviceSynchronize());
            fn(a, sa, a, sa);                 // in-place
            CK(cudaDeviceSynchronize());
            CK(cudaMemcpy2D(h1.data(), W, b, sb, W, H, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy2D(h2.data(), W, a, sa, W, H, cudaMemcpyDeviceToHost));
            int nbad = 0, mx = 0;
            for (size_t i = 0; i < N; ++i) { int d = abs((int)h1[i] - (int)h2[i]); if (d) ++nbad; mx = std::max(mx, d); }
            printf("  %-16s %4dx%-4d  in-place vs out-of-place: diff=%d max=%d %s\n",
                   name, W, H, nbad, mx, nbad ? "<== MISMATCH" : "");
            bad += nbad;
        };

        for (int k : {5, 7, 15}) {
            char nm[32]; snprintf(nm, 32, "gaussian k=%d", k);
            check(nm, [&](uint8_t* s, size_t ss, uint8_t* d, size_t ds) { gpu_gaussian(s, ss, d, ds, W, H, k, 0.f); });
        }
        for (int k : {15, 31}) {
            char nm[32]; snprintf(nm, 32, "mean k=%d", k);
            check(nm, [&](uint8_t* s, size_t ss, uint8_t* d, size_t ds) { gpu_mean(s, ss, d, ds, W, H, k, k); });
        }
        for (int k : {15, 31}) {
            char nm[32]; snprintf(nm, 32, "erode k=%d", k);
            check(nm, [&](uint8_t* s, size_t ss, uint8_t* d, size_t ds) { gpu_erode(s, ss, d, ds, W, H, k, k); });
        }
        CK(cudaFree(a)); CK(cudaFree(b));
    }
    printf("IP_TOTAL_DIFF %d  %s\n", bad, bad ? "IP_MISMATCH" : "IP_ALL_SAME");
    return 0;
}
