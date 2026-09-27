// 冒烟测试：小尺寸 + 边角尺寸，逐算子与 OpenCV-CUDA 结果比对
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include "ops.h"
#include <cstdio>
#include <random>

namespace cvo = cv::cuda;

static int g_fail = 0;

// 最大绝对差；ncomp=1
static int maxDiff(const cv::Mat& a, const cv::Mat& b, const char* what)
{
    if (a.size() != b.size() || a.type() != b.type()) {
        printf("  [FAIL] %s: size/type mismatch (mine %dx%d t%d vs cv %dx%d t%d)\n",
               what, a.cols, a.rows, a.type(), b.cols, b.rows, b.type());
        ++g_fail;
        return -1;
    }
    int md = 0;
    for (int y = 0; y < a.rows; ++y) {
        const uchar* pa = a.ptr(y);
        const uchar* pb = b.ptr(y);
        for (size_t x = 0; x < (size_t)a.cols * a.elemSize(); ++x)
            md = std::max(md, std::abs((int)pa[x] - (int)pb[x]));
    }
    if (md > 0) { printf("  [WARN] %s: max diff = %d\n", what, md); }
    else        { printf("  [OK]   %s: identical\n", what); }
    if (md > 1) ++g_fail;
    return md;
}

static cv::Mat randU8(int w, int h, bool binary, unsigned seed)
{
    cv::Mat m(h, w, CV_8UC1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto it = m.begin<uchar>(); it != m.end<uchar>(); ++it)
        *it = binary ? (d(rng) < 77 ? 255 : 0) : (uchar)d(rng);
    return m;
}

int main()
{
    setvbuf(stdout, nullptr, _IONBF, 0);
    CUDA_CHECK(cudaFree(0));
    struct Case { int w, h; };
    const Case cases[] = {{65, 49}, {128, 128}, {640, 480}, {321, 200}};

    for (const Case& c : cases) {
        const int W = c.w, H = c.h;
        printf("== case %dx%d ==\n", W, H);

        cv::Mat gray = randU8(W, H, false, 1);
        cv::Mat bin  = randU8(W, H, true, 2);

        cvo::GpuMat dGray(gray), dBin(bin);
        cvo::GpuMat dOutCV, dTmp;

        // ---------- 积分图 ----------
        {
            cvo::integral(dGray, dOutCV);
            cv::Mat ref; dOutCV.download(ref);

            uint32_t* dMine; size_t mstep = (size_t)W * 4;
            CUDA_CHECK(cudaMalloc(&dMine, mstep * H));
            gpu_integral(dGray.data, dGray.step, dMine, mstep, W, H);
            cv::Mat mine(H, W, CV_32SC1);
            CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dMine, mstep, (size_t)W * 4, H, cudaMemcpyDeviceToHost));
            // OpenCV 的 integral 输出是 (W+1)x(H+1)，左上多一行一列 0；取子矩形比对
            maxDiff(mine, ref(cv::Rect(1, 1, W, H)), "integral");
            cudaFree(dMine);
        }

        // ---------- 均值滤波 ----------
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            f->apply(dGray, dOutCV);
            cv::Mat ref; dOutCV.download(ref);

            uint8_t* dMine; size_t mstep = (size_t)W;
            CUDA_CHECK(cudaMallocPitch(&dMine, &mstep, W, H));
            gpu_mean(dGray.data, dGray.step, dMine, mstep, W, H, k, k);
            cv::Mat mine(H, W, CV_8UC1);
            CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dMine, mstep, W, H, cudaMemcpyDeviceToHost));
            char name[32]; snprintf(name, 32, "mean k=%d", k);
            maxDiff(mine, ref, name);
            cudaFree(dMine);
        }

        // ---------- 腐蚀/膨胀 ----------
        for (int k : {3, 15, 31}) {
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
            auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker);
            auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, ker);
            for (int opn = 0; opn < 2; ++opn) {
                if (opn == 0) fe->apply(dBin, dOutCV);
                else          fd->apply(dBin, dOutCV);
                cv::Mat ref; dOutCV.download(ref);

                uint8_t* dMine; size_t mstep = (size_t)W;
                CUDA_CHECK(cudaMallocPitch(&dMine, &mstep, W, H));
                if (opn == 0) gpu_erode(dBin.data, dBin.step, dMine, mstep, W, H, k, k);
                else          gpu_dilate(dBin.data, dBin.step, dMine, mstep, W, H, k, k);
                cv::Mat mine(H, W, CV_8UC1);
                CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dMine, mstep, W, H, cudaMemcpyDeviceToHost));
                char name[32]; snprintf(name, 32, "%s k=%d (bin)", opn ? "dilate" : "erode", k);
                maxDiff(mine, ref, name);
                // 灰度图也测
                if (opn == 0) gpu_erode(dGray.data, dGray.step, dMine, mstep, W, H, k, k);
                else          gpu_dilate(dGray.data, dGray.step, dMine, mstep, W, H, k, k);
                CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dMine, mstep, W, H, cudaMemcpyDeviceToHost));
                if (opn == 0) fe->apply(dGray, dOutCV);
                else          fd->apply(dGray, dOutCV);
                dOutCV.download(ref);
                snprintf(name, 32, "%s k=%d (gray)", opn ? "dilate" : "erode", k);
                maxDiff(mine, ref, name);
                cudaFree(dMine);
            }
        }

        // ---------- 高斯 ----------
        for (int k : {5, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            f->apply(dGray, dOutCV);
            cv::Mat ref; dOutCV.download(ref);

            uint8_t* dMine; size_t mstep = (size_t)W;
            CUDA_CHECK(cudaMallocPitch(&dMine, &mstep, W, H));
            gpu_gaussian(dGray.data, dGray.step, dMine, mstep, W, H, k, 0.f);
            cv::Mat mine(H, W, CV_8UC1);
            CUDA_CHECK(cudaMemcpy2D(mine.data, mine.step, dMine, mstep, W, H, cudaMemcpyDeviceToHost));
            char name[32]; snprintf(name, 32, "gauss k=%d", k);
            maxDiff(mine, ref, name);
            cudaFree(dMine);
        }

        // ---------- 直方图 ----------
        {
            // 通用（灰度）
            cvo::calcHist(dGray, dOutCV);
            cv::Mat ref; dOutCV.download(ref);
            int* dMine; CUDA_CHECK(cudaMalloc(&dMine, 256 * sizeof(int)));
            // 新契约：hist kernel 完全覆写全 256 bin，不做任何预清零直接验证
            gpu_histogram(dGray.data, dGray.step, W, H, dMine);
            int mine[256]; CUDA_CHECK(cudaMemcpy(mine, dMine, sizeof(mine), cudaMemcpyDeviceToHost));
            const int* rp = ref.ptr<int>();
            int md = 0;
            for (int i = 0; i < 256; ++i) md = std::max(md, std::abs(rp[i] - mine[i]));
            printf("  [%s] hist gray: max diff = %d\n", md ? "WARN" : "OK", md);
            if (md) ++g_fail;

            // 二值
            gpu_histogram_binary(dBin.data, dBin.step, W, H, dMine);
            CUDA_CHECK(cudaMemcpy(mine, dMine, sizeof(mine), cudaMemcpyDeviceToHost));
            cvo::calcHist(dBin, dOutCV);
            dOutCV.download(ref);
            rp = ref.ptr<int>();
            md = 0;
            for (int i = 0; i < 256; ++i) {
                const int expect = (i == 0 || i == 255) ? rp[i] : 0;
                md = std::max(md, std::abs(expect - mine[i]));
            }
            printf("  [%s] hist binary: max diff = %d\n", md ? "WARN" : "OK", md);
            if (md) ++g_fail;
            cudaFree(dMine);
        }
    }

    printf(g_fail ? "\nSMOKE: %d problem(s)\n" : "\nSMOKE: all pass (%d flagged)\n", g_fail);
    return g_fail ? 1 : 0;
}
