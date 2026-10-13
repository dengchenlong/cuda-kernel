// probe6 (spike 003c 续): 修正 probe5 两个缺陷后复测
//   - probe5 的 cv 槽复用了 integral 的 8MB dHist → cv 状态被毒化（半数窗口 0.13ms）,
//     全部 cv 计时改用专用 dHistOut
//   - probe5 的 cvk4@170 在 4K 无 grid-stride, 只处理了 1360/2160 行 → 计时无效, 本版剔除,
//     有效网格恒为 divUp(H,8)（cv 原版）或带行 grid-stride 的变体
// 候选（gray）:
//   v0_170        现版: uint4 + last-block, 无预清零
//   cvk4_h8       cv 复刻: 4B 读 + per-block global atomicAdd, 预清零在计时内
//   cvk16_h8      同上但 uint4 读
//   lb4_h8        4B 读 + last-block 归约（无预清零, 保持覆写契约）
//   v0gs_170      uint4 + last-block + 行 grid-stride(170) —— 与现版等价(4K 时现版也是 170)
// 候选（binary）:
//   bino_256      现版 popcount
//   b4_256        4B 读 popcount + last-block
//   bcvk_h8       cv 复刻（binary 数据走 histogram256 路径: 4B 读 + smem atomic + global atomic 归约 + 预清零）
// 口径: 主口径前奏(integral+box+gaussian) + 本方/cv 同窗交错取中位数
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <functional>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)

__global__ void hist_v0(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                        int* __restrict__ d_hist, int* __restrict__ partials,
                        unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const unsigned cols4 = (unsigned)W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    int* myp = partials + blockIdx.x * 256;
    myp[tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridBlocks - 1);
        amLast = (ticket == (unsigned)(gridBlocks - 1));
    }
    __syncthreads();
    if (amLast) {
        int s = 0;
        for (int b = 0; b < gridBlocks; ++b) s += partials[b * 256 + tid];
        d_hist[tid] = s;
    }
}

// 4B 读 + per-block global atomicAdd（预清零由 host 负责）
__global__ void hist_ga4(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                         int* __restrict__ d_hist, int gridBlocks)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int cols_4 = W >> 2;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const unsigned* row = (const unsigned*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < cols_4; x += 32) {
            const unsigned data = row[x];
            atomicAdd(&shist[(data >> 0) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 8) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
    }
    __syncthreads();
    const int histVal = shist[tid];
    if (histVal > 0) atomicAdd(d_hist + tid, histVal);
}

// uint4 读 + per-block global atomicAdd（预清零由 host 负责）
__global__ void hist_ga16(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                          int* __restrict__ d_hist, int gridBlocks)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const unsigned cols4 = (unsigned)W >> 4;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        for (unsigned x = threadIdx.x; x < cols4; x += 32) {
            const uint4 v = row4[x];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u)
#pragma unroll
                for (int b = 0; b < 32; b += 8)
                    atomicAdd(&shist[(w[u] >> b) & 0xff], 1);
        }
    }
    __syncthreads();
    const int histVal = shist[tid];
    if (histVal > 0) atomicAdd(d_hist + tid, histVal);
}

// 4B 读 + last-block 归约（无预清零, 完全覆写）
__global__ void hist_lb4(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                         int* __restrict__ d_hist, int* __restrict__ partials,
                         unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ int shist[256];
    __shared__ bool amLast;
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int cols_4 = W >> 2;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const unsigned* row = (const unsigned*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < cols_4; x += 32) {
            const unsigned data = row[x];
            atomicAdd(&shist[(data >> 0) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 8) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
    }
    __syncthreads();
    int* myp = partials + blockIdx.x * 256;
    myp[tid] = shist[tid];
    __threadfence();
    if (tid == 0) {
        const unsigned ticket = atomicInc(counter, gridBlocks - 1);
        amLast = (ticket == (unsigned)(gridBlocks - 1));
    }
    __syncthreads();
    if (amLast) {
        int s = 0;
        for (int b = 0; b < gridBlocks; ++b) s += partials[b * 256 + tid];
        d_hist[tid] = s;
    }
}

// ---------- binary ----------
__global__ void hist_binary_v0(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                               int* __restrict__ d_hist, int* __restrict__ partials,
                               unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ bool amLast;
    unsigned zeros = 0, ones = 0;
    const bool aligned = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0) && ((W & 15) == 0);
    const size_t N = (size_t)W * H;
    if (aligned) {
        const size_t bpr = (size_t)W >> 4;
        const bool tight = (sstep == (size_t)W);
        const size_t total4 = tight ? (N >> 4) : (size_t)H * bpr;
        const size_t stride = (size_t)gridDim.x * blockDim.x;
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < total4; i += stride) {
            const uint4 v = tight ? ((const uint4*)src)[i]
                                  : ((const uint4*)(src + (i / bpr) * sstep))[i - (i / bpr) * bpr];
            const unsigned w[4] = { v.x, v.y, v.z, v.w };
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned b255 = ((w[u] & 0xff) == 0xff) | (((w[u] >> 8) & 0xff) == 0xff) << 1 |
                                      (((w[u] >> 16) & 0xff) == 0xff) << 2 | (((w[u] >> 24) & 0xff) == 0xff) << 3;
                const unsigned b0 = ((w[u] & 0xff) == 0x00) | (((w[u] >> 8) & 0xff) == 0x00) << 1 |
                                    (((w[u] >> 16) & 0xff) == 0x00) << 2 | (((w[u] >> 24) & 0xff) == 0x00) << 3;
                ones += __popc(b255);
                zeros += __popc(b0);
            }
        }
    }
    __shared__ unsigned s0[32], s255[32];
    for (int d = 16; d > 0; d >>= 1) {
        zeros += __shfl_down_sync(0xffffffffu, zeros, d);
        ones += __shfl_down_sync(0xffffffffu, ones, d);
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { s0[wid] = zeros; s255[wid] = ones; }
    __syncthreads();
    if (wid == 0) {
        zeros = (lane < 8) ? s0[lane] : 0;
        ones = (lane < 8) ? s255[lane] : 0;
        for (int d = 16; d > 0; d >>= 1) {
            zeros += __shfl_down_sync(0xffffffffu, zeros, d);
            ones += __shfl_down_sync(0xffffffffu, ones, d);
        }
        if (lane == 0) {
            int* myp = partials + blockIdx.x * 2;
            myp[0] = (int)zeros; myp[1] = (int)ones;
            __threadfence();
            const unsigned ticket = atomicInc(counter, gridBlocks - 1);
            amLast = (ticket == (unsigned)(gridBlocks - 1));
        }
    }
    __syncthreads();
    if (amLast) {
        int s = 0;
        if (threadIdx.x == 0 || threadIdx.x == 255)
            for (int b = 0; b < gridBlocks; ++b) s += partials[b * 2 + (threadIdx.x ? 1 : 0)];
        d_hist[threadIdx.x] = (threadIdx.x == 0) ? s : (threadIdx.x == 255) ? s : 0;
    }
}

// binary 4B 读 popcount + last-block
__global__ void hist_binary4(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                             int* __restrict__ d_hist, int* __restrict__ partials,
                             unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ bool amLast;
    unsigned zeros = 0, ones = 0;
    const bool aligned = ((sstep & 3) == 0) && ((reinterpret_cast<uintptr_t>(src) & 3) == 0) && ((W & 3) == 0);
    if (aligned) {
        const size_t bpr = (size_t)W >> 2;
        const size_t stride = (size_t)gridDim.x * blockDim.x;
        for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < (size_t)H * bpr; i += stride) {
            const size_t y = i / bpr, x = i - y * bpr;
            const unsigned v = ((const unsigned*)(src + y * sstep))[x];
            const unsigned b255 = ((v & 0xff) == 0xff) | (((v >> 8) & 0xff) == 0xff) << 1 |
                                  (((v >> 16) & 0xff) == 0xff) << 2 | (((v >> 24) & 0xff) == 0xff) << 3;
            const unsigned b0 = ((v & 0xff) == 0x00) | (((v >> 8) & 0xff) == 0x00) << 1 |
                                (((v >> 16) & 0xff) == 0x00) << 2 | (((v >> 24) & 0xff) == 0x00) << 3;
            ones += __popc(b255);
            zeros += __popc(b0);
        }
    } else {
        for (int y = blockIdx.x; y < H; y += gridDim.x)
            for (int x = threadIdx.x; x < W; x += blockDim.x) {
                const uint8_t p = src[y * sstep + x];
                zeros += (p == 0); ones += (p == 255);
            }
    }
    __shared__ unsigned s0[32], s255[32];
    for (int d = 16; d > 0; d >>= 1) {
        zeros += __shfl_down_sync(0xffffffffu, zeros, d);
        ones += __shfl_down_sync(0xffffffffu, ones, d);
    }
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) { s0[wid] = zeros; s255[wid] = ones; }
    __syncthreads();
    if (wid == 0) {
        zeros = (lane < 8) ? s0[lane] : 0;
        ones = (lane < 8) ? s255[lane] : 0;
        for (int d = 16; d > 0; d >>= 1) {
            zeros += __shfl_down_sync(0xffffffffu, zeros, d);
            ones += __shfl_down_sync(0xffffffffu, ones, d);
        }
        if (lane == 0) {
            int* myp = partials + blockIdx.x * 2;
            myp[0] = (int)zeros; myp[1] = (int)ones;
            __threadfence();
            const unsigned ticket = atomicInc(counter, gridBlocks - 1);
            amLast = (ticket == (unsigned)(gridBlocks - 1));
        }
    }
    __syncthreads();
    if (amLast) {
        int s = 0;
        if (threadIdx.x == 0 || threadIdx.x == 255)
            for (int b = 0; b < gridBlocks; ++b) s += partials[b * 2 + (threadIdx.x ? 1 : 0)];
        d_hist[threadIdx.x] = (threadIdx.x == 0) ? s : (threadIdx.x == 255) ? s : 0;
    }
}

// binary cv 复刻: 4B 读, smem atomic(每字节一次), per-block global atomicAdd, 预清零由 host 负责
__global__ void hist_binary_cvk(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                                int* __restrict__ d_hist, int gridBlocks)
{
    __shared__ int shist[256];
    const int tid = threadIdx.y * 32 + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    const int cols_4 = W >> 2;
    for (int y = blockIdx.x * blockDim.y + threadIdx.y; y < H; y += gridBlocks * blockDim.y) {
        const unsigned* row = (const unsigned*)(src + (size_t)y * sstep);
        for (int x = threadIdx.x; x < cols_4; x += 32) {
            const unsigned data = row[x];
            atomicAdd(&shist[(data >> 0) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 8) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
    }
    __syncthreads();
    const int histVal = shist[tid];
    if (histVal > 0) atomicAdd(d_hist + tid, histVal);
}

struct Timer {
    cudaEvent_t e0, e1;
    std::vector<float> a, b;
    void init() { cudaEventCreate(&e0); cudaEventCreate(&e1); }
    void rep(auto fnMine, auto fnCv) {
        cudaEventRecord(e0); fnMine(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float t; cudaEventElapsedTime(&t, e0, e1); a.push_back(t);
        cudaEventRecord(e0); fnCv(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t, e0, e1); b.push_back(t);
    }
    void report(const char* name) {
        std::vector<float> ma = a, mb = b;
        std::nth_element(ma.begin(), ma.begin() + ma.size() / 2, ma.end());
        std::nth_element(mb.begin(), mb.begin() + mb.size() / 2, mb.end());
        printf("  %-24s med=%.4f  (cv同窗med=%.4f, ratio=%.3f)\n", name, ma[ma.size() / 2], mb[mb.size() / 2],
               mb[mb.size() / 2] / ma[ma.size() / 2]);
        fflush(stdout);
    }
};

int main()
{
    CUDA_CHECK(cudaFree(0));
    int* partials; unsigned* counter; int* dHistRaw;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));
    CUDA_CHECK(cudaMalloc(&dHistRaw, 256 * 4));

    const dim3 t(32, 8);
    for (auto& sz : {std::pair<int,int>{640,480}, std::pair<int,int>{1280,720}, std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        const int gh8 = (H + 7) / 8;          // cv 原版网格
        printf("== %dx%d gray ==\n", W, H); fflush(stdout);
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
        cvo::GpuMat dG(m), dHistOut;          // cv 专用输出, 绝不复用 integral 大 buffer
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

        cvo::GpuMat dBig;                     // 前奏大 buffer 单独放
        cvo::integral(dG, dBig);
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dBig);
        }
        for (int k : {5, 7, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 20; ++i) f->apply(dG, dBig);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        Timer T; T.init();
        auto warm = [&](auto fn) { for (int i = 0; i < 8; ++i) fn(); CUDA_CHECK(cudaDeviceSynchronize()); };
        auto cwarm = [&] { for (int i = 0; i < 20; ++i) cvo::calcHist(dG, dHistOut); CUDA_CHECK(cudaDeviceSynchronize()); };
        const int REPS = 40;
        auto cvf = [&] { cvo::calcHist(dG, dHistOut); };

        struct V { const char* name; std::function<void()> f; };
        std::vector<V> vs;
        vs.push_back({"v0_170(现版)", [=] { hist_v0<<<170, t>>>(s8, ss8, W, H, dHistRaw, partials, counter, 170); }});
        vs.push_back({"ga4_h8(cv复刻4B)", [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 1024));
                                                hist_ga4<<<gh8, t>>>(s8, ss8, W, H, dHistRaw, gh8); }});
        vs.push_back({"ga4_170gs(4B+stride)", [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 1024));
                                                    hist_ga4<<<170, t>>>(s8, ss8, W, H, dHistRaw, 170); }});
        vs.push_back({"ga16_h8(uint4+gatomic)", [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 1024));
                                                      hist_ga16<<<gh8, t>>>(s8, ss8, W, H, dHistRaw, gh8); }});
        vs.push_back({"lb4_h8(4B+lastblock)", [=] { hist_lb4<<<gh8, t>>>(s8, ss8, W, H, dHistRaw, partials, counter, gh8); }});

        // 正确性对照: 每个 kernel 的输出 vs cv
        {
            cvo::calcHist(dG, dHistOut);
            std::vector<int> ref(256);
            CUDA_CHECK(cudaMemcpy(ref.data(), dHistOut.data, 1024, cudaMemcpyDeviceToHost));
            for (auto& v : vs) {
                CUDA_CHECK(cudaMemset(dHistRaw, 0, 1024));
                v.f();
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<int> got(256);
                CUDA_CHECK(cudaMemcpy(got.data(), dHistRaw, 1024, cudaMemcpyDeviceToHost));
                // 注意: 非预清零 kernel 会累加, 先清零再跑一次
                CUDA_CHECK(cudaMemset(dHistRaw, 0, 1024));
                v.f();
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(got.data(), dHistRaw, 1024, cudaMemcpyDeviceToHost));
                int diff = 0;
                for (int i = 0; i < 256; ++i) diff += (got[i] != ref[i]);
                printf("  correctness %-24s diff=%d (total=%d)\n", v.name, diff, got[0] + got[1] + got[2] + got[255]);
                fflush(stdout);
            }
        }

        for (auto& v : vs) {
            warm(v.f);
            cwarm();
            T.a.clear(); T.b.clear();
            for (int i = 0; i < REPS; ++i) T.rep(v.f, cvf);
            T.report(v.name);
        }
        fflush(stdout);
        cudaFree(s8);
    }

    // ---------- binary ----------
    for (auto& sz : {std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        const int gh8 = (H + 7) / 8;
        printf("== %dx%d binary ==\n", W, H); fflush(stdout);
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(9);
        std::uniform_int_distribution<int> dd(0, 1);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)(dd(rng) * 255);
        cvo::GpuMat dG(m), dHistOut;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        cvo::GpuMat dBig;
        cvo::integral(dG, dBig);
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dBig);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        Timer T; T.init();
        auto warm = [&](auto fn) { for (int i = 0; i < 8; ++i) fn(); CUDA_CHECK(cudaDeviceSynchronize()); };
        auto cwarm = [&] { for (int i = 0; i < 20; ++i) cvo::calcHist(dG, dHistOut); CUDA_CHECK(cudaDeviceSynchronize()); };
        const int REPS = 40;
        auto cvf = [&] { cvo::calcHist(dG, dHistOut); };

        struct V { const char* name; std::function<void()> f; };
        std::vector<V> vs;
        vs.push_back({"bino_256(现版)", [=] { hist_binary_v0<<<256, 256>>>(s8, ss8, W, H, dHistRaw, partials, counter, 256); }});
        vs.push_back({"bin4_256(4B popcount)", [=] { hist_binary4<<<256, 256>>>(s8, ss8, W, H, dHistRaw, partials, counter, 256); }});
        vs.push_back({"bcvk_h8(cv复刻)", [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 1024));
                                               hist_binary_cvk<<<gh8, t>>>(s8, ss8, W, H, dHistRaw, gh8); }});

        // 正确性
        {
            cvo::calcHist(dG, dHistOut);
            std::vector<int> ref(256);
            CUDA_CHECK(cudaMemcpy(ref.data(), dHistOut.data, 1024, cudaMemcpyDeviceToHost));
            for (auto& v : vs) {
                CUDA_CHECK(cudaMemset(dHistRaw, 0, 1024));
                v.f();
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<int> got(256);
                CUDA_CHECK(cudaMemcpy(got.data(), dHistRaw, 1024, cudaMemcpyDeviceToHost));
                int diff = 0;
                for (int i = 0; i < 256; ++i) diff += (got[i] != ref[i]);
                printf("  correctness %-24s diff=%d (z=%d o=%d)\n", v.name, diff, got[0], got[255]);
                fflush(stdout);
            }
        }

        for (auto& v : vs) {
            warm(v.f);
            cwarm();
            T.a.clear(); T.b.clear();
            for (int i = 0; i < REPS; ++i) T.rep(v.f, cvf);
            T.report(v.name);
        }
        fflush(stdout);
        cudaFree(s8);
    }
    return 0;
}
