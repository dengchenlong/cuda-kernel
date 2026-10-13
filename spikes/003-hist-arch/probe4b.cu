// probe4b: 定位 p4run 卡死 —— 无前奏，cv calcHist 短循环 + 分步打印
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <cstdio>

namespace cvo = cv::cuda;
#define CUDA_CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { printf("CUDA err %s @%d\n", cudaGetErrorString(e_), __LINE__); return 1; } } while (0)

int main()
{
    CUDA_CHECK(cudaFree(0));
    const int W = 1920, H = 1080;
    cv::Mat m(H, W, CV_8UC1, cv::Scalar(128));
    cvo::GpuMat dG(m), dHist;

    printf("step1: calcHist first\n"); fflush(stdout);
    cvo::calcHist(dG, dHist);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("step2: first done\n"); fflush(stdout);

    for (int i = 0; i < 100; ++i) cvo::calcHist(dG, dHist);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("step3: 100 calcHist done\n"); fflush(stdout);

    // 前奏：integral 到 dHist（8MB），随后 calcHist
    cvo::integral(dG, dHist);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("step4: integral into dHist done (size %dx%d)\n", dHist.cols, dHist.rows); fflush(stdout);

    cvo::calcHist(dG, dHist);   // 输出尺寸变化：8MB → 256x1
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("step5: calcHist after integral done (size %dx%d)\n", dHist.cols, dHist.rows); fflush(stdout);

    for (int i = 0; i < 100; ++i) cvo::calcHist(dG, dHist);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("step6: 100 calcHist after integral done\n"); fflush(stdout);
    return 0;
}
