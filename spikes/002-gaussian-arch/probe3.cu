// probe3: 复现项目 benchmark 的 cv 口径（0.030ms@1080p k=15 vs 本 spike 的 0.28ms）
// 与项目 benchmark 完全一致的调用序列：先跑别的 op（GpuMat 池焐热）再测 gaussian。
// 若 cv 仍 0.28 → OpenCV-CUDA 4.14 高斯在我们这台机有两种路径（NPP / 自研 kernel），
// 由 OpenCV 编译选项或运行时决定 —— 需要查 cv::cuda::createGaussianFilter 的实现走哪条。
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err\n"); exit(1); } } while (0)

int main()
{
    CUDA_CHECK(cudaFree(0));
    cv::cuda::GpuMat poolWarm(4000, 4000, CV_8UC1);   // 先把 OpenCV 的 GpuMemoryPool 焐热
    poolWarm.setTo(0);
    CUDA_CHECK(cudaDeviceSynchronize());

    const int W = 1920, H = 1080;
    cv::Mat m(H, W, CV_8UC1);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> dd(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it) *it = (uchar)dd(rng);
    cv::cuda::GpuMat dG(m), dOut;

    // 与项目 benchmark 相同：先做 integral 预热 GpuMat 分配
    cv::cuda::integral(dG, dOut);

    for (int k : {5, 15}) {
        auto f = cv::cuda::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
        for (int i = 0; i < 8; ++i) f->apply(dG, dOut);
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
        std::vector<float> ms;
        for (int i = 0; i < 30; ++i) {
            cudaEventRecord(e0); f->apply(dG, dOut); cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float t; cudaEventElapsedTime(&t, e0, e1); ms.push_back(t);
        }
        std::nth_element(ms.begin(), ms.begin() + 15, ms.end());
        printf("cv k=%-3d %8.4f ms (项目 benchmark 口径应是 ~0.020/0.030)\n", k, ms[15]);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    }

    // 输出 dOut 的 step/size 是否每次 apply 变化 —— 判断是否在重新分配
    printf("dOut size=%dx%d step=%zu\n", dOut.cols, dOut.rows, dOut.step);
    return 0;
}
