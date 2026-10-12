// probe5: cv gaussian k=15 @1080p 连续 200 次 apply 计时，打印时间序列 —— 找相变。
// 同时对照：先跑 50 次 mean k=31（焐热分离 buf 内存池）再测。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err\n"); exit(1); } } while (0)

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int W = 1920, H = 1080;
    cv::Mat m(H, W, CV_8UC1);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dd(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
    cvo::GpuMat dG(m), dOut;

    // 场景 1：直接测 cv gaussian k=15，200 次逐次打印（每 20 次汇总）
    {
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(15, 15), 0, 0);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        printf("场景1（无预热，直接 k=15 x200）:\n");
        for (int b = 0; b < 10; ++b) {
            std::vector<float> ms;
            for (int i = 0; i < 20; ++i) {
                cudaEventRecord(e0); f->apply(dG, dOut); cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
            }
            std::sort(ms.begin(), ms.end());
            printf("  [%2d] min=%8.4f med=%8.4f max=%8.4f\n", b, ms[0], ms[10], ms[19]);
        }
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // 场景 2：先跑 mean k=31 x100 焐热，再测 k=15
    {
        auto fm = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(31, 31), cv::Point(-1, -1));
        for (int i = 0; i < 100; ++i) fm->apply(dG, dOut);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(15, 15), 0, 0);
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        printf("场景2（mean k=31 x100 预热后）:\n");
        std::vector<float> all;
        for (int i = 0; i < 60; ++i) {
            cudaEventRecord(e0); f->apply(dG, dOut); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); all.push_back(t);
        }
        std::sort(all.begin(), all.end());
        printf("  min=%8.4f med=%8.4f max=%8.4f\n", all[0], all[30], all[59]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }
    return 0;
}
