// probe_int4：r51 融合路径的逐列/逐行错点分布（定位是行内扫描还是列累加器）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    int W = 64, H = 128;
    if (argc > 2) { W = atoi(argv[1]); H = atoi(argv[2]); }
    std::vector<uint8_t> img((size_t)W * H);
    for (size_t i = 0; i < img.size(); ++i) img[i] = (uint8_t)((i * 37 + 11) & 0xff);
    std::vector<uint32_t> ref((size_t)W * H);
    {
        std::vector<int64_t> colacc(W, 0);
        for (int y = 0; y < H; ++y) {
            int64_t s = 0;
            for (int x = 0; x < W; ++x) { s += img[(size_t)y * W + x]; colacc[x] += s; ref[(size_t)y * W + x] = (uint32_t)colacc[x]; }
        }
    }
    uint8_t* d_src = nullptr; size_t sstep = 0;
    uint32_t* d_dst = nullptr; size_t dstep = 0;
    CK(cudaMallocPitch(&d_src, &sstep, (size_t)W, H));
    CK(cudaMallocPitch(&d_dst, &dstep, (size_t)W * 4, H));
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy((uint8_t*)d_src + (size_t)y * sstep, &img[(size_t)y * W], W, cudaMemcpyHostToDevice));
    gpu_integral(d_src, sstep, d_dst, dstep, W, H);
    std::vector<uint32_t> got((size_t)W * H);
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy(&got[(size_t)y * W], (uint8_t*)d_dst + (size_t)y * dstep, (size_t)W * 4, cudaMemcpyDeviceToHost));

    printf("== %dx%d (CK_TH_INT=%s) ==\n", W, H, getenv("CK_TH_INT") ? getenv("CK_TH_INT") : "default");
    printf("bad per row (row:count) first 12 rows with errors:\n");
    int shown = 0;
    for (int y = 0; y < H && shown < 12; ++y) {
        int n = 0, x0 = -1, x1 = -1;
        for (int x = 0; x < W; ++x) if (got[(size_t)y * W + x] != ref[(size_t)y * W + x]) { if (x0 < 0) x0 = x; x1 = x; ++n; }
        if (n) { printf("  y=%d n=%d span=[%d,%d]\n", y, n, x0, x1); ++shown; }
    }
    printf("bad per column-group of 16 (group:count):\n");
    for (int g = 0; g * 16 < W; ++g) {
        int n = 0;
        for (int y = 0; y < H; ++y)
            for (int x = g * 16; x < std::min(W, g * 16 + 16); ++x)
                if (got[(size_t)y * W + x] != ref[(size_t)y * W + x]) ++n;
        if (n) printf("  cols %d..%d : %d\n", g * 16, std::min(W, g * 16 + 15), n);
    }
    printf("sample row 0 and 1:\n");
    for (int y = 0; y < 2; ++y) {
        printf("  y=%d got:", y);
        for (int x = 0; x < std::min(W, 56); ++x) printf(" %u", got[(size_t)y * W + x]);
        printf("\n  y=%d ref:", y);
        for (int x = 0; x < std::min(W, 56); ++x) printf(" %u", ref[(size_t)y * W + x]);
        printf("\n");
    }
    return 0;
}
