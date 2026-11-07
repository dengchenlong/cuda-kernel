// probe_int3：r51 融合单遍路径的调试/扫参探针
// 用途：
//   1) 小尺寸逐像素比对（打印错点位置，定位是 chunk 前缀还是一行内扫描错）
//   2) CK_TH_INT 扫 chunk 高度、CK_NOFUSE 做 A/B（融合 vs r50 两遍）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

static int dumpCase(int W, int H, bool verbose)
{
    const size_t sw = W;
    std::vector<uint8_t> img((size_t)H * sw);
    for (size_t i = 0; i < img.size(); ++i) img[i] = (uint8_t)((i * 37 + 11) & 0xff);

    std::vector<uint32_t> ref((size_t)W * H);
    {
        std::vector<int64_t> colacc(W, 0);
        for (int y = 0; y < H; ++y) {
            int64_t s = 0;
            for (int x = 0; x < W; ++x) {
                s += img[(size_t)y * sw + x];
                colacc[x] += s;
                ref[(size_t)y * W + x] = (uint32_t)colacc[x];
            }
        }
    }

    uint8_t* d_src = nullptr; size_t sstep = 0;
    uint32_t* d_dst = nullptr; size_t dstep = 0;
    CK(cudaMallocPitch(&d_src, &sstep, (size_t)W, H));
    CK(cudaMallocPitch(&d_dst, &dstep, (size_t)W * 4, H));
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy((uint8_t*)d_src + (size_t)y * sstep, &img[(size_t)y * sw], W, cudaMemcpyHostToDevice));
    gpu_integral(d_src, sstep, d_dst, dstep, W, H);
    std::vector<uint32_t> got((size_t)W * H);
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy(&got[(size_t)y * W], (uint8_t*)d_dst + (size_t)y * dstep, (size_t)W * 4, cudaMemcpyDeviceToHost));

    int bad = 0; uint32_t md = 0; int fy = -1, fx = -1;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
            const uint32_t a = got[(size_t)y * W + x], b = ref[(size_t)y * W + x];
            const uint32_t d = a > b ? a - b : b - a;
            if (d) { if (!bad) { fy = y; fx = x; } ++bad; md = std::max(md, d); }
        }
    printf("  %4dx%-4d -> %s bad=%d maxdiff=%u first=(%d,%d)\n", W, H, bad ? "[FAIL]" : "[OK]", bad, md, fy, fx);
    if (bad && verbose) {
        printf("    first 8 mismatches:\n");
        int shown = 0;
        for (int y = 0; y < H && shown < 8; ++y)
            for (int x = 0; x < W && shown < 8; ++x) {
                const uint32_t a = got[(size_t)y * W + x], b = ref[(size_t)y * W + x];
                if (a != b) { printf("      (%d,%d) got=%u ref=%u\n", x, y, a, b); ++shown; }
            }
        // 打印错点分布：每行有几个错
        int rows = 0, lastrow = -1, colsAt0 = 0;
        for (int y = 0; y < H; ++y) {
            int n = 0;
            for (int x = 0; x < W; ++x) if (got[(size_t)y * W + x] != ref[(size_t)y * W + x]) ++n;
            if (n) { if (lastrow < 0) lastrow = y; if (y - lastrow > 1) colsAt0 = -999; ++rows; if (n) lastrow = y; }
        }
        printf("    bad rows=%d\n", rows);
    }
    CK(cudaFree(d_src)); CK(cudaFree(d_dst));
    return bad;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const bool verbose = (argc > 1 && atoi(argv[1]) != 0);
    const char* th = getenv("CK_TH_INT");
    printf("CK_TH_INT=%s\n", th ? th : "(default)");

    struct Case { int W, H; };
    const Case cases[] = {{64, 128}, {64, 96}, {128, 128}, {256, 256}, {640, 480}, {1280, 720}, {1920, 1080}};
    int fails = 0;
    for (const Case& c : cases) if (dumpCase(c.W, c.H, verbose)) ++fails;
    printf("\n%s (fail=%d)\n", fails ? "PROBE_INT3_FAIL" : "PROBE_INT3_ALL_OK", fails);
    return fails ? 1 : 0;
}
