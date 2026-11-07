// probe_int5：integral 单口径计时器（一次进程一个配置，配合 shell 交错 A/B、3 轮取 min）
// 用法：probe_int5 [reps]
//   CK_NOFUSE=1 → 走 r50 两遍路径；否则走 r51 融合单遍（CK_TH_INT 可覆盖 chunk 高度）
// 为什么拆成单进程：dispatch 里的 env 读取做了 static 缓存（getenv 在 0.03ms 级小图上不可忽略），
// 进程内无法切换配置 —— 交错 A/B 改由 shell 反复起进程完成。
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return -1; } } while (0)

template <typename F>
static double timeitMin(F f, int warm, int reps)
{
    for (int i = 0; i < warm; ++i) f();
    cudaDeviceSynchronize();
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

int main(int argc, char** argv)
{
    CK(cudaFree(0));
    const int reps = (argc > 1) ? atoi(argv[1]) : 30;
    struct Case { int W, H; };
    const Case cases[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    const bool nofuse = getenv("CK_NOFUSE") != nullptr;
    const char* th = getenv("CK_TH_INT");

    for (const Case& c : cases) {
        const int W = c.W, H = c.H;
        const size_t N = (size_t)W * H;
        uint8_t* d_src = nullptr; size_t sstep = 0;
        uint32_t* d_dst = nullptr; size_t dstep = 0;
        CK(cudaMallocPitch(&d_src, &sstep, (size_t)W, H));
        CK(cudaMallocPitch(&d_dst, &dstep, (size_t)W * 4, H));
        std::vector<uint8_t> h(N);
        for (size_t i = 0; i < N; ++i) h[i] = (uint8_t)(i * 37 + 11);
        CK(cudaMemcpy2D(d_src, sstep, h.data(), (size_t)W, (size_t)W, H, cudaMemcpyHostToDevice));

        const double t = timeitMin([&] { gpu_integral(d_src, sstep, d_dst, dstep, W, H); }, 5, reps);
        const double bytes = (double)N * (nofuse ? 13.0 : 6.0) / 1e6;
        printf("SIZE %d %d %.4f %.0f %s th=%s\n", W, H, t, bytes / t,
               nofuse ? "twopass" : "fused", th ? th : "auto");
        CK(cudaFree(d_src)); CK(cudaFree(d_dst));
    }
    return 0;
}
