// probe_int2：积分图 dispatch 覆盖测试（r50）
// r50 改了 gpu_integral 的两处 dispatch（pass1 用 uint4 store、pass2 宽图走单遍 1col），
// 需要覆盖此前没被测过的边界组合：W%4!=0、W 非 16 倍数、奇数 step、4B 对齐但非 16B 对齐、
// 极扁/极高、W 恰好跨 512 阈值两端。参照用 CPU int64 精确前缀和（u8 输入最大 4K×255 万级
// 累加也能装下，不依赖 OpenCV）。
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

static int runCase(int W, int H, int padBytes, bool tight, unsigned seed, bool verbose)
{
    const size_t sw = (size_t)W + padBytes;
    std::vector<uint8_t> img((size_t)H * sw);
    unsigned st = seed;
    for (size_t i = 0; i < img.size(); ++i) {
        st = st * 1664525u + 1013904223u;
        img[i] = (uint8_t)((st >> 16) ^ (i * 37));
    }
    // CPU 参照
    std::vector<uint32_t> ref((size_t)W * H);
    {
        std::vector<int64_t> rowsum(W);
        std::vector<int64_t> colacc(W, 0);
        for (int y = 0; y < H; ++y) {
            int64_t s = 0;
            for (int x = 0; x < W; ++x) {
                s += img[(size_t)y * sw + x];
                rowsum[x] = s;
            }
            for (int x = 0; x < W; ++x) {
                colacc[x] += rowsum[x];
                ref[(size_t)y * W + x] = (uint32_t)colacc[x];
            }
        }
    }

    uint8_t* d_src = nullptr; size_t sstep = 0;
    uint32_t* d_dst = nullptr; size_t dstep = 0;
    if (tight) {
        // 紧排：src step = W（可能不是 16 倍数），dst step = W*4
        CK(cudaMalloc(&d_src, (size_t)W * H));
        CK(cudaMalloc(&d_dst, (size_t)W * 4 * H));
        sstep = W; dstep = (size_t)W * 4;
    } else {
        CK(cudaMallocPitch(&d_src, &sstep, (size_t)W, H));
        CK(cudaMallocPitch(&d_dst, &dstep, (size_t)W * 4, H));
    }
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy((uint8_t*)d_src + (size_t)y * sstep, &img[(size_t)y * sw], W, cudaMemcpyHostToDevice));

    gpu_integral(d_src, sstep, d_dst, dstep, W, H);

    std::vector<uint32_t> got((size_t)W * H);
    for (int y = 0; y < H; ++y)
        CK(cudaMemcpy(&got[(size_t)y * W], (uint8_t*)d_dst + (size_t)y * dstep, (size_t)W * 4, cudaMemcpyDeviceToHost));

    int bad = 0; uint32_t md = 0;
    for (size_t i = 0; i < got.size(); ++i) {
        const uint32_t d = got[i] > ref[i] ? got[i] - ref[i] : ref[i] - got[i];
        if (d) { ++bad; md = std::max(md, d); }
    }
    if (bad || verbose)
        printf("  %4dx%-4d %s srcStep=%zu dstStep=%zu -> %s bad=%d maxdiff=%u\n",
               W, H, tight ? "tight" : "pitch", sstep, dstep, bad ? "[FAIL]" : "[OK]", bad, md);
    CK(cudaFree(d_src)); CK(cudaFree(d_dst));
    return bad;
}

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const bool verbose = (argc > 1 && atoi(argv[1]) != 0);
    struct Case { int W, H; };
    const Case cases[] = {
        {16, 1}, {17, 1}, {31, 3}, {33, 5}, {63, 7}, {64, 8},
        {500, 10}, {511, 9}, {512, 9}, {513, 9}, {514, 5},
        {640, 480}, {720, 479}, {723, 41}, {1023, 33}, {1024, 33}, {1025, 33},
        {1279, 720}, {1280, 720}, {1919, 100}, {1920, 1080}, {1921, 7},
        {2047, 5}, {2048, 5}, {2049, 5}, {3839, 3}, {3840, 2160}, {3841, 3},
        {4096, 64}, {4095, 64},
        {100, 1000}, {8, 4000},
    };
    int fails = 0, total = 0;
    for (const Case& c : cases) {
        for (int tight = 0; tight < 2; ++tight) {
            for (int pad = 0; pad <= 4; pad += 4) {
                ++total;
                if (runCase(c.W, c.H, pad, tight != 0, 1234u + (unsigned)(c.W * 31 + c.H), verbose)) ++fails;
            }
        }
    }
    printf("\n%s (cases=%d fail=%d)\n", fails ? "PROBE_INT2_FAIL" : "PROBE_INT2_ALL_OK", total, fails);
    return fails ? 1 : 0;
}