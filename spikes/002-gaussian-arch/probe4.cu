// probe4: 逐字复刻主 benchmark 的计时结构与 op 序列（integral → mean → morph → gaussian），
// 目标：解释 spike probe2 里 cv gaussian 0.28ms vs 主 benchmark 0.030ms 的矛盾。
// 变量隔离：A) Timer 结构（warmup 内嵌 sync + 每次 event sync） B) 前序 op 序列
// C) filter 对象生命周期 D) GpuMat 复用历史。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err\n"); exit(1); } } while (0)

struct Timer {
    cudaEvent_t s, e;
    Timer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    float run(const std::function<void()>& f, int warmup, int reps) const {
        for (int i = 0; i < warmup; ++i) f();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<float> ms(reps);
        for (int i = 0; i < reps; ++i) {
            cudaEventRecord(s);
            f();
            cudaEventRecord(e);
            cudaEventSynchronize(e);
            cudaEventElapsedTime(&ms[i], s, e);
        }
        std::nth_element(ms.begin(), ms.begin() + reps / 2, ms.end());
        return ms[reps / 2];
    }
};

static cv::Mat randU8(int w, int h, unsigned seed)
{
    cv::Mat m(h, w, CV_8UC1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)d(rng);
    return m;
}

int main()
{
    CUDA_CHECK(cudaFree(0));
    Timer tm;
    const int W = 1920, H = 1080;
    const int reps = 30, warmup = 6;

    cv::Mat gray = randU8(W, H, 7);
    cv::Mat bin = randU8(W, H, 8);
    cvo::GpuMat dGray(gray), dBin(bin), dOutCV;

    // ---- 完整复刻主 benchmark 前序序列 ----
    cvo::integral(dGray, dOutCV);
    tm.run([&] { cvo::integral(dGray, dOutCV); }, warmup, reps);

    for (int k : {3, 15, 31}) {
        auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
        tm.run([&] { f->apply(dGray, dOutCV); }, warmup, reps);
    }
    for (int k : {3, 15, 31}) {
        cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
        auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker);
        auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, ker);
        tm.run([&] { fe->apply(dBin, dOutCV); }, warmup, reps);
        tm.run([&] { fd->apply(dBin, dOutCV); }, warmup, reps);
    }

    // ---- gaussian：主口径 ----
    for (int k : {5, 7, 15}) {
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
        double cvms = tm.run([&] { f->apply(dGray, dOutCV); }, warmup, reps);
        printf("主口径 cv k=%-3d %8.4f ms\n", k, cvms);
    }

    // ---- 变量 A：Timer 换 probe2 风格（每次 event 前无 sync，交错）----
    for (int k : {5, 15}) {
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
        for (int i = 0; i < 8; ++i) f->apply(dGray, dOutCV);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms;
        for (int i = 0; i < 20; ++i) {
            cudaEventRecord(e0); f->apply(dGray, dOutCV); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
        }
        std::nth_element(ms.begin(), ms.begin() + 10, ms.end());
        printf("probe2风格 cv k=%-3d %8.4f ms\n", k, ms[10]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // ---- 变量 C：每次计时前重新 createGaussianFilter ----
    for (int k : {5, 15}) {
        double cvms = tm.run([&] {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            f->apply(dGray, dOutCV);
        }, warmup, reps);
        printf("每次重建filter cv k=%-3d %8.4f ms\n", k, cvms);
    }

    // ---- 变量 D：dOutCV 换成独立新 GpuMat ----
    for (int k : {5, 15}) {
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
        cvo::GpuMat fresh;
        double cvms = tm.run([&] { f->apply(dGray, fresh); }, warmup, reps);
        printf("独立输出GpuMat cv k=%-3d %8.4f ms\n", k, cvms);
    }
    return 0;
}
