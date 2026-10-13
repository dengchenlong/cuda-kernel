// probe8 (spike 003c 验收): 生产代码 head-to-head —— 链接 libgpuops.a, 直接调
// gpu_histogram / gpu_histogram_binary（含 API 内 memsetAsync）, 与 cv calcHist 同窗交错,
// cv 用专用独立输出（绝不复用 integral 大 buffer, 规避分配器毒化）。
// 报中位数与最小值（cv 快路径口径 = min）。4K gray + 4K binary。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include "ops.h"
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#ifndef CUDA_CHECK
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); exit(1); } } while (0)
#endif

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int W = 3840, H = 2160;
    cv::Mat gray(H, W, CV_8UC1), bin(H, W, CV_8UC1);
    {
        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dd(0, 255);
        std::uniform_int_distribution<int> db(0, 1);
        for (auto it = gray.begin<uchar>(); it != gray.end<uchar>(); ++it) *it = (uchar)dd(rng);
        for (auto it = bin.begin<uchar>(); it != bin.end<uchar>(); ++it) *it = (uchar)(db(rng) * 255);
    }
    uint8_t* s8; size_t ss8;
    CUDA_CHECK(cudaMallocPitch(&s8, &ss8, W, H));
    int* dHist;
    CUDA_CHECK(cudaMalloc(&dHist, 1024));
    std::vector<int> got(256), ref(256);

    for (int mode = 0; mode < 2; ++mode) {
        const cv::Mat& m = mode ? bin : gray;
        printf("== 3840x2160 %s ==\n", mode ? "binary" : "gray");
        CUDA_CHECK(cudaMemcpy2D(s8, ss8, m.data, m.step, W, H, cudaMemcpyHostToDevice));
        cvo::GpuMat dG(m), dHistOut, dBig;
        cvo::integral(dG, dBig);   // 主口径前奏（用独立大 buffer, 不碰 dHistOut）
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            for (int i = 0; i < 20; ++i) f->apply(dG, dBig);
        }
        for (int k : {5, 7, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            for (int i = 0; i < 20; ++i) f->apply(dG, dBig);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // 正确性（生产 API vs cv）
        cvo::calcHist(dG, dHistOut);
        CUDA_CHECK(cudaMemcpy(ref.data(), dHistOut.data, 1024, cudaMemcpyDeviceToHost));
        if (mode) gpu_histogram_binary(s8, ss8, W, H, dHist);
        else      gpu_histogram(s8, ss8, W, H, dHist);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got.data(), dHist, 1024, cudaMemcpyDeviceToHost));
        int diff = 0;
        for (int i = 0; i < 256; ++i) diff += (got[i] != ref[i]);
        printf("  correctness: max diff = %d (sum=%lld, ref sum=%lld)\n", diff,
               std::accumulate(got.begin(), got.end(), 0LL), std::accumulate(ref.begin(), ref.end(), 0LL));
        fflush(stdout);

        // 交错计时: 60 轮, 每轮本方→cv→cv→本方（平衡顺序效应）
        auto mineF = [&] { if (mode) gpu_histogram_binary(s8, ss8, W, H, dHist);
                           else         gpu_histogram(s8, ss8, W, H, dHist); };
        auto cvF = [&] { cvo::calcHist(dG, dHistOut); };
        for (int i = 0; i < 10; ++i) { mineF(); cvF(); }
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> tm_, tc;
        for (int i = 0; i < 60; ++i) {
            cudaEventRecord(e0); mineF(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); tm_.push_back(t);
            cudaEventRecord(e0); cvF(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            cudaEventElapsedTime(&t, e0, e1); tc.push_back(t);
        }
        std::vector<float> a = tm_, b = tc;
        std::nth_element(a.begin(), a.begin() + 30, a.end());
        std::nth_element(b.begin(), b.begin() + 30, b.end());
        float amin = *std::min_element(tm_.begin(), tm_.end());
        float bmin = *std::min_element(tc.begin(), tc.end());
        printf("  mine  med=%.4f min=%.4f\n", a[30], amin);
        printf("  cv    med=%.4f min=%.4f\n", b[30], bmin);
        printf("  speedup(med)=%.3f  speedup(cv快路径min)=%.3f\n", b[30] / a[30], bmin / amin);
        fflush(stdout);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}
