// probe5 (spike 003c): 4K hist gray/binary 攻坚
// Q1 网格均衡: H=2160=270*8, grid=270 无 grid-stride 尾巴 vs 170 vs 256 vs 336
// Q2 归约机制: cv 式 per-block global atomicAdd(+预清零) vs last-block
// Q3 带宽地板: ro16(uint4 只读校验和) / ro4(uint 只读) —— 8.29MB 物理上限
// Q4 binary: bino@256 vs cv(4B 读, 270blk) vs ro 地板
// 口径: 主口径前奏(integral+box+gaussian) + 交错计时(每轮内 cv 与本方 kernel 交替) 取中位数
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

// ---------- ro: 只读带宽地板（防优化: 校验和写回） ----------
__global__ void ro16(const uint4* __restrict__ p, size_t n4, unsigned* __restrict__ sink)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    unsigned acc = 0;
    const size_t stride = gridDim.x * (size_t)blockDim.x;
    for (; i < n4; i += stride) {
        const uint4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xdeadbeefu) *sink = acc;   // 永假, 防止死码消除
}

__global__ void ro4(const unsigned* __restrict__ p, size_t n, unsigned* __restrict__ sink)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    unsigned acc = 0;
    const size_t stride = gridDim.x * (size_t)blockDim.x;
    for (; i < n; i += stride) acc += p[i];
    if (acc == 0xdeadbeefu) *sink = acc;
}

// ---------- v0: 现版（last-block 归约, 完整覆写 d_hist） ----------
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

// ---------- cvk: cv 复刻（4B 读, per-block global atomicAdd, 需预清零; pre=true 时清零在计时内） ----------
template<bool fourByteAligned>
__global__ void hist_cvk(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                         int* __restrict__ d_hist)
{
    __shared__ int shist[256];
    const int y = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    if (y < H) {
        const unsigned* rowPtr = (const unsigned*)(fourByteAligned ? src + (size_t)y * sstep
                                                                   : src + (size_t)y * sstep + 4);
        const int cols_4 = fourByteAligned ? (W >> 2) : ((W - 4) >> 2);
        for (int x = threadIdx.x; x < cols_4; x += blockDim.x) {
            const unsigned data = rowPtr[x];
            atomicAdd(&shist[(data >> 0) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 8) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 16) & 0xFFU], 1);
            atomicAdd(&shist[(data >> 24) & 0xFFU], 1);
        }
        // 尾部（本探针 W 均为 4 倍数且对齐, fourByteAligned=true 时无尾巴）
    }
    __syncthreads();
    const int histVal = shist[tid];
    if (histVal > 0) atomicAdd(d_hist + tid, histVal);
}

// ---------- cvk16: cv 结构但 uint4 读 ----------
__global__ void hist_cvk16(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                           int* __restrict__ d_hist)
{
    __shared__ int shist[256];
    const int y = blockIdx.x * blockDim.y + threadIdx.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shist[tid] = 0;
    __syncthreads();
    if (y < H) {
        const uint4* row4 = (const uint4*)(src + (size_t)y * sstep);
        const int cols4 = W >> 4;
        for (int x = threadIdx.x; x < cols4; x += blockDim.x) {
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

// ---------- bino: 现版 binary（256 blk） ----------
__global__ void hist_binary_v0(const uint8_t* __restrict__ src, size_t sstep, int W, int H,
                               int* __restrict__ d_hist, int* __restrict__ partials,
                               unsigned* __restrict__ counter, int gridBlocks)
{
    __shared__ bool amLast;
    const size_t N = (size_t)W * H;
    unsigned zeros = 0, ones = 0;
    const bool aligned = ((sstep & 15) == 0) && ((reinterpret_cast<uintptr_t>(src) & 15) == 0) && ((W & 15) == 0);
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

// ---------- 交错计时器 ----------
struct Timer {
    cudaEvent_t e0, e1;
    std::vector<float> a, b;   // a=本方, b=cv
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
        printf("  %-26s med=%.4f  (cv同窗med=%.4f, ratio=%.3f)\n", name, ma[ma.size() / 2], mb[mb.size() / 2],
               mb[mb.size() / 2] / ma[ma.size() / 2]);
        fflush(stdout);
    }
};

int main()
{
    CUDA_CHECK(cudaFree(0));
    int* partials; unsigned* counter; unsigned* sink; int* dHistRaw;
    CUDA_CHECK(cudaMalloc(&partials, 512 * 256 * 4));
    CUDA_CHECK(cudaMalloc(&counter, 4));
    CUDA_CHECK(cudaMemset(counter, 0, 4));
    CUDA_CHECK(cudaMalloc(&sink, 4));
    CUDA_CHECK(cudaMalloc(&dHistRaw, 256 * 4));

    const dim3 t(32, 8);
    for (auto& sz : {std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        printf("== %dx%d (gray, random) ==\n", W, H); fflush(stdout);
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
        cvo::GpuMat dG(m), dHist;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));

        // 主口径前奏
        cvo::integral(dG, dHist);
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dHist);
        }
        for (int k : {5, 7, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 20; ++i) f->apply(dG, dHist);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        const size_t N = (size_t)W * H;
        const int blocks170 = std::min(170, std::max(48, (H + 7) / 8));

        Timer T; T.init();
        auto warm = [&](auto fn) { for (int i = 0; i < 8; ++i) fn(); CUDA_CHECK(cudaDeviceSynchronize()); };
        auto cwarm = [&] { for (int i = 0; i < 20; ++i) cvo::calcHist(dG, dHist); CUDA_CHECK(cudaDeviceSynchronize()); };
        const int REPS = 40;

        // Q3 带宽地板（cv 同窗无意义, 但保持结构一致 —— cv 槽位放真实 cv calcHist）
        auto ro16f = [&] { ro16<<<170, 256>>>((const uint4*)s8, ss8 == (size_t)W ? N >> 4 : (size_t)H * (ss8 >> 4), sink); auto e_=cudaGetLastError(); if(e_!=cudaSuccess){printf("ro16 launch err %s\n",cudaGetErrorString(e_)); exit(1);} };
        auto ro4f  = [&] { ro4<<<170, 256>>>((const unsigned*)s8, ss8 / 4, sink); auto e_=cudaGetLastError(); if(e_!=cudaSuccess){printf("ro4 launch err %s\n",cudaGetErrorString(e_)); exit(1);} };
        warm(ro16f); warm(ro4f);
        for (int i = 0; i < REPS; ++i) T.rep(ro16f, [&] { cvo::calcHist(dG, dHist); });
        T.report("ro16 只读地板(uint4)");
        T.a.clear(); T.b.clear();
        for (int i = 0; i < REPS; ++i) T.rep(ro4f, [&] { cvo::calcHist(dG, dHist); });
        T.report("ro4 只读地板(uint)");

        // Q1/Q2: 网格 × 归约, 全部与 cv 同窗交错
        auto mk_v0 = [&](int gb) { return [=] { hist_v0<<<gb, t>>>(s8, ss8, W, H, dHistRaw, partials, counter, gb); auto e_=cudaGetLastError(); if(e_!=cudaSuccess){printf("v0@%d launch err %s\n",gb,cudaGetErrorString(e_)); exit(1);} }; };
        auto mk_cvk4 = [&](int gb) {
            return [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 256 * 4));
                         hist_cvk<true><<<gb, t>>>(s8, ss8, W, H, dHistRaw); auto e_=cudaGetLastError(); if(e_!=cudaSuccess){printf("cvk4@%d launch err %s\n",gb,cudaGetErrorString(e_)); exit(1);} };
        };
        auto mk_cvk16 = [&](int gb) {
            return [=] { CUDA_CHECK(cudaMemsetAsync(dHistRaw, 0, 256 * 4));
                         hist_cvk16<<<gb, t>>>(s8, ss8, W, H, dHistRaw); auto e_=cudaGetLastError(); if(e_!=cudaSuccess){printf("cvk16@%d launch err %s\n",gb,cudaGetErrorString(e_)); exit(1);} };
        };
        struct V { const char* name; int gb; int kind; };   // kind 0=v0 1=cvk4 2=cvk16
        std::vector<V> vs = {
            {"v0@170(现版)", blocks170, 0},
            {"v0@256", 256, 0}, {"v0@270", 270, 0}, {"v0@336", 336, 0},
            {"cvk4@270(cv复刻)", 270, 1}, {"cvk4@170", blocks170, 1},
            {"cvk16@270", 270, 2}, {"cvk16@336", 336, 2},
        };
        for (auto& v : vs) {
            std::function<void()> f;
            if (v.kind == 0) f = mk_v0(v.gb); else if (v.kind == 1) f = mk_cvk4(v.gb); else f = mk_cvk16(v.gb);
            warm(f);
            cwarm();
            T.a.clear(); T.b.clear();
            for (int i = 0; i < REPS; ++i) T.rep(f, [&] { cvo::calcHist(dG, dHist); });
            T.report(v.name);
        }
        fflush(stdout);
        cudaFree(s8);
    }

    // ---------- Q4 binary ----------
    for (auto& sz : {std::pair<int,int>{1920,1080}, std::pair<int,int>{3840,2160}}) {
        const int W = sz.first, H = sz.second;
        printf("== %dx%d (binary 0/255) ==\n", W, H); fflush(stdout);
        cv::Mat m(H, W, CV_8UC1);
        std::mt19937 rng(9);
        std::uniform_int_distribution<int> dd(0, 1);
        for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)(dd(rng) * 255);
        cvo::GpuMat dG(m), dHist;
        uint8_t* s8; size_t ss8;
        CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        cvo::integral(dG, dHist);
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dHist);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        Timer T; T.init();
        const size_t N = (size_t)W * H;
        auto ro16f = [&] { ro16<<<170, 256>>>((const uint4*)s8, ss8 == (size_t)W ? N >> 4 : (size_t)H * (ss8 >> 4), sink); };
        auto bino = [&] { hist_binary_v0<<<256, 256>>>(s8, ss8, W, H, dHistRaw, partials, counter, 256); };
        auto bin270 = [&] { hist_binary_v0<<<270, 256>>>(s8, ss8, W, H, dHistRaw, partials, counter, 270); };
        for (int i = 0; i < 8; ++i) { bino(); bin270(); ro16f(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        const int REPS = 40;
        for (int i = 0; i < REPS; ++i) T.rep(bino, [&] { cvo::calcHist(dG, dHist); });
        T.report("bino@256(现版)");
        T.a.clear(); T.b.clear();
        for (int i = 0; i < REPS; ++i) T.rep(bin270, [&] { cvo::calcHist(dG, dHist); });
        T.report("bino@270");
        T.a.clear(); T.b.clear();
        for (int i = 0; i < REPS; ++i) T.rep(ro16f, [&] { cvo::calcHist(dG, dHist); });
        T.report("ro16 只读地板(binary)");
        fflush(stdout);
        cudaFree(s8);
    }
    return 0;
}
