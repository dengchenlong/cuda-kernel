// probe_mf32k3：morph-f32 k=3 单遍融合内核（对标 mean-f32 k=3 的 r45 路线）
// 现状：gpu_erode_f32 k=3 走「rows 直加 smem 窗口 + cols ring」两遍 = 16B/px，
//       4K 1.80ms / 73 GB/s，而纯拷贝天花板 93 GB/s -> 只能砍流量到 8B/px。
// 定案：4 px/线程 float4，水平 3-tap 用 __shfl 取邻（min/max 的 clamp 边界 == 恒等元，
//       不需要 REFLECT_101，这点与 mean 不同），垂直 3 行寄存器滚动。
// 模式：check（新 vs 旧逐位一致） / ab（交错 min 计时）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include "ops.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

struct MnF { __device__ __forceinline__ float operator()(float a, float b) const { return fminf(a,b); } static __device__ __forceinline__ float ident() { return __int_as_float(0x7f800000); } };
struct MxF { __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a,b); } static __device__ __forceinline__ float ident() { return __int_as_float(0xff800000); } };

template <typename Op, int TH>
__global__ void mf32_fuse3_kernel(const float* __restrict__ src, size_t sstep,
                                  float* __restrict__ dst, size_t dstep,
                                  int W, int H)
{
    const int tid  = threadIdx.x;
    const int x0   = (blockIdx.x * blockDim.x + tid) * 4;
    const int y0   = blockIdx.y * TH;
    const int y1   = min(H, y0 + TH);
    const int lane = tid & 31;
    const bool act = (x0 < W);
    const int xa   = act ? x0 : ((W - 4) & ~3);
    const int rows = (int)(sstep >> 2), orows = (int)(dstep >> 2);
    const Op op;

    // 一行内水平 3 邻域（clamp 到图内 = 恒等元语义）
    auto hrow = [&](int ry, float o[4]) {
        const float* srow = src + (size_t)ry * rows;
        const float4 v = *(const float4*)(srow + xa);
        float l1 = __shfl_up_sync(0xffffffffu, v.w, 1);
        float r1 = __shfl_down_sync(0xffffffffu, v.x, 1);
        if (lane == 0 || xa == 0)      l1 = (xa == 0) ? v.x : srow[xa - 1];
        if (lane == 31 || xa + 4 >= W) r1 = (xa + 4 >= W) ? v.w : srow[xa + 4];
        o[0] = op(v.x, op(l1, v.y));
        o[1] = op(v.y, op(v.x, v.z));
        o[2] = op(v.z, op(v.y, v.w));
        o[3] = op(v.w, op(v.z, r1));
    };

    float hp[4], hc[4], hn[4];
    hrow(max(0, y0 - 1), hp);
    hrow(y0, hc);
    for (int y = y0; y < y1; ++y) {
        hrow(min(H - 1, y + 1), hn);
        if (act) {
            float4 o;
            o.x = op(hp[0], op(hc[0], hn[0]));
            o.y = op(hp[1], op(hc[1], hn[1]));
            o.z = op(hp[2], op(hc[2], hn[2]));
            o.w = op(hp[3], op(hc[3], hn[3]));
            *(float4*)(dst + (size_t)y * orows + x0) = o;
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) { hp[i] = hc[i]; hc[i] = hn[i]; }
    }
}

template <typename Op>
static bool fused(int k, const float* s, size_t sstep, float* d, size_t dstep, int W, int H)
{
    if (k != 3 || W < 8 || W > 4096 || (W % 4) || (sstep % 16) || (dstep % 16)) return false;
    const int TH = (H >= 1600) ? 64 : (H > 800 ? 32 : (H > 400 ? 16 : 8));
    int th = ((W + 3) / 4 + 31) / 32 * 32;
    if (th > 256) th = 256;
    if (th < 32)  th = 32;
    dim3 g((W + 4 * th - 1) / (4 * th), (H + TH - 1) / TH);
    switch (TH) {
    case 8:  mf32_fuse3_kernel<Op, 8> <<<g, th>>>(s, sstep, d, dstep, W, H); break;
    case 16: mf32_fuse3_kernel<Op, 16><<<g, th>>>(s, sstep, d, dstep, W, H); break;
    case 32: mf32_fuse3_kernel<Op, 32><<<g, th>>>(s, sstep, d, dstep, W, H); break;
    default: mf32_fuse3_kernel<Op, 64><<<g, th>>>(s, sstep, d, dstep, W, H); break;
    }
    return true;
}

struct FImg {
    std::vector<float> h;
    float* d = nullptr; size_t step = 0; int W = 0, H = 0;
    void make(int w, int hh, unsigned seed) {
        W = w; H = hh; h.resize((size_t)w * hh);
        unsigned st = seed;
        for (auto& v : h) { st = st * 1664525u + 1013904223u; v = (float)((st >> 8) & 0xffff) / 7.0f; }
        CKV(cudaMallocPitch(&d, &step, (size_t)(w + 8) * 4, hh));   // 多 8 列余量，check2 的 +4B 偏移读才不越界
        CKV(cudaMemcpy2D(d, step, h.data(), (size_t)w * 4, (size_t)w * 4, hh, cudaMemcpyHostToDevice));
    }
    ~FImg() { if (d) cudaFree(d); }
};

template <typename F>
static double timeMin(F f, int warm, int reps) {
    for (int i = 0; i < warm; ++i) f();
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
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
static void abMin(FA fa, FB fb, int warm, int reps, double& oa, double& ob) {
    for (int i = 0; i < warm; ++i) { fa(); fb(); }
    CKV(cudaDeviceSynchronize());
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float ba = 1e9f, bb = 1e9f;
    for (int i = 0; i < reps; ++i) {
        float ms = 0.f;
        cudaEventRecord(a); fa(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < ba) ba = ms;
        cudaEventRecord(a); fb(); cudaEventRecord(b); cudaEventSynchronize(b);
        cudaEventElapsedTime(&ms, a, b); if (ms < bb) bb = ms;
    }
    cudaEventDestroy(a); cudaEventDestroy(b);
    oa = ba; ob = bb;
}

int main(int argc, char** argv) {
    CK(cudaFree(0));
    const std::string mode = (argc > 1) ? argv[1] : "check";
    const int reps = (argc > 2) ? atoi(argv[2]) : 20;

    if (mode == "check2") {
        // 独立比对：把旧两遍路径"强制"出来（dst 指针 +4 字节 => %16 != 0 => 回退），
        // 与库内融合路径逐位比对。参照侧用偏移指针写、偏移指针读，语义完全一致。
        struct S { int W, H; };
        const S shapes[] = {{8,1},{12,1},{16,1},{16,2},{16,3},{16,4},{64,1},{64,2},{64,3},{64,4},
                            {100,100},{640,480},{720,479},{1280,720},{1920,1080},{3840,2160},
                            {4096,64},{320,240},{256,256},{1024,768},{3840,64},{512,512}};
        int fails = 0, total = 0;
        for (const S& s : shapes) {
            const int W = s.W, H = s.H;
            for (int opn = 0; opn < 2; ++opn) {
                FImg src, a, b;
                src.make(W, H, 7777u + (unsigned)(W * 7 + H));
                a.make(W, H, 1u); b.make(W, H, 1u);
                // b 偏移 4 字节：强制旧两遍路径
                float* bd = b.d + 1;
                size_t bst = b.step;
                if (opn == 0) {
                    gpu_erode_f32(src.d, src.step, a.d, a.step, W, H, 3, 3);   // 融合（新默认）
                    gpu_erode_f32(src.d, src.step, bd,  bst,   W, H, 3, 3);     // 两遍（强制回退）
                } else {
                    gpu_dilate_f32(src.d, src.step, a.d, a.step, W, H, 3, 3);
                    gpu_dilate_f32(src.d, src.step, bd,  bst,   W, H, 3, 3);
                }
                CK(cudaDeviceSynchronize());
                std::vector<float> ra((size_t)W*H), rb((size_t)W*H);
                CK(cudaMemcpy2D(ra.data(), (size_t)W*4, a.d,  a.step, (size_t)W*4, H, cudaMemcpyDeviceToHost));
                for (int y = 0; y < H; ++y)
                    CK(cudaMemcpy(rb.data() + (size_t)y*W, bd + (size_t)y*(bst>>2), (size_t)W*4, cudaMemcpyDeviceToHost));
                size_t firstbad = (size_t)-1;
                for (size_t i = 0; i < (size_t)W*H; ++i)
                    if (memcmp(&ra[i], &rb[i], 4)) { firstbad = i; break; }
                ++total;
                const bool ok = (firstbad == (size_t)-1);
                printf("  %4dx%-4d %s %s", W, H, opn ? "dilate" : "erode", ok ? "[OK]" : "[FAIL]");
                if (!ok) { printf("  firstbad %zu (%d,%d) fused %.7g vs twopass %.7g", firstbad,
                                  (int)(firstbad % W), (int)(firstbad / W), ra[firstbad], rb[firstbad]); ++fails; }
                printf("\n");
            }
        }
        printf("\n%s (total=%d fail=%d)\n", fails ? "PROBE_MF32K3_CHECK2_FAIL" : "PROBE_MF32K3_CHECK2_ALL_OK", total, fails);
        return fails ? 1 : 0;
    }

    if (mode == "check") {
        struct S { int W, H; };
        const S shapes[] = {{8,1},{12,1},{16,1},{16,2},{16,3},{17,3},{31,17},{33,33},{64,64},{100,100},
                            {127,129},{160,17},{640,480},{720,479},{1280,720},{1920,1080},{3840,2160},
                            {333,257},{4096,64},{7,400},{320,240},{1281,721}};
        int fails = 0, total = 0, skipped = 0;
        for (const S& s : shapes) {
            const int W = s.W, H = s.H;
            for (int opn = 0; opn < 2; ++opn) {
                FImg src, a, b;
                src.make(W, H, 4242u + (unsigned)(W * 3 + H));
                a.make(W, H, 1u); b.make(W, H, 1u);
                if (opn == 0) gpu_erode_f32(src.d, src.step, a.d, a.step, W, H, 3, 3);
                else          gpu_dilate_f32(src.d, src.step, a.d, a.step, W, H, 3, 3);
                CK(cudaDeviceSynchronize());
                bool used = false;
                if (opn == 0) used = fused<MnF>(3, src.d, src.step, b.d, b.step, W, H);
                else          used = fused<MxF>(3, src.d, src.step, b.d, b.step, W, H);
                if (!used) { ++skipped; printf("  %4dx%-4d %s SKIP(fallback)\n", W, H, opn ? "dilate" : "erode"); continue; }
                CK(cudaDeviceSynchronize());
                std::vector<float> ra((size_t)W*H), rb((size_t)W*H);
                CK(cudaMemcpy2D(ra.data(), (size_t)W*4, a.d, a.step, (size_t)W*4, H, cudaMemcpyDeviceToHost));
                CK(cudaMemcpy2D(rb.data(), (size_t)W*4, b.d, b.step, (size_t)W*4, H, cudaMemcpyDeviceToHost));
                size_t firstbad = (size_t)-1;
                for (size_t i = 0; i < (size_t)W*H; ++i)
                    if (memcmp(&ra[i], &rb[i], 4)) { firstbad = i; break; }
                ++total;
                const bool ok = (firstbad == (size_t)-1);
                printf("  %4dx%-4d %s %s", W, H, opn ? "dilate" : "erode", ok ? "[OK]" : "[FAIL]");
                if (!ok) { printf("  firstbad %zu (%d,%d) %.6g vs %.6g", firstbad,
                                  (int)(firstbad % W), (int)(firstbad / W), ra[firstbad], rb[firstbad]); ++fails; }
                printf("\n");
            }
        }
        printf("\n%s (total=%d fail=%d skip=%d)\n", fails ? "PROBE_MF32K3_FAIL" : "PROBE_MF32K3_ALL_OK", total, fails, skipped);
        return fails ? 1 : 0;
    }

    if (mode == "ab") {
        struct C { const char* n; int W, H; };
        const C cs[] = {{"640x480",640,480},{"1280x720",1280,720},{"1920x1080",1920,1080},{"3840x2160",3840,2160}};
        for (const C& c : cs) {
            FImg src, dst; src.make(c.W, c.H, 999u); dst.make(c.W, c.H, 1u);
            double oe=0, ne=0, od=0, nd=0;
            abMin([&]{ gpu_erode_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, 3, 3); },
                  [&]{ fused<MnF>(3, src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, oe, ne);
            abMin([&]{ gpu_dilate_f32(src.d, src.step, dst.d, dst.step, c.W, c.H, 3, 3); },
                  [&]{ fused<MxF>(3, src.d, src.step, dst.d, dst.step, c.W, c.H); }, 5, reps, od, nd);
            printf("%-10s erode old %8.4f fused %8.4f (%.2fx) | dilate old %8.4f fused %8.4f (%.2fx)\n",
                   c.n, oe, ne, oe/ne, od, nd, od/nd);
        }
        return 0;
    }
    printf("usage: probe_mf32k3 [check|ab] [reps]\n");
    return 0;
}