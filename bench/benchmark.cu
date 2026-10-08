// benchmark：我的算子 vs OpenCV-CUDA（同机同卡），cudaEvents 计时取中位数
// 输出：stdout 表格 + bench_results.csv
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/imgproc.hpp>
#include "ops.h"
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>

namespace cvo = cv::cuda;

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

static FILE* g_csv = nullptr;

static void report(const std::string& op, int W, int H, const std::string& param,
                   double mine, double cvms)
{
    const double mp = (double)W * H / 1e6;
    printf("%-14s %5dx%-5d %-12s mine %8.3f ms (%7.1f MP/s) | opencv %8.3f ms (%7.1f MP/s) | speedup %.2fx %s\n",
           op.c_str(), W, H, param.c_str(),
           mine, mp / (mine * 1e-3), cvms, mp / (cvms * 1e-3),
           cvms / mine, cvms / mine >= 1.0 ? "WIN" : "lose");
    if (g_csv)
        fprintf(g_csv, "%s,%d,%d,%s,%.4f,%.4f,%.3f\n", op.c_str(), W, H, param.c_str(), mine, cvms, cvms / mine);
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

struct Bufs {
    uint8_t* u8 = nullptr;   size_t u8step = 0;
    uint32_t* u32 = nullptr; size_t u32step = 0;
    int* hist = nullptr;
    int W = 0, H = 0;
    void alloc(int w, int h) {
        W = w; H = h;
        CUDA_CHECK(cudaMallocPitch(&u8, &u8step, w, h));
        CUDA_CHECK(cudaMallocPitch(&u32, &u32step, (size_t)w * 4, h));
        CUDA_CHECK(cudaMalloc(&hist, 256 * sizeof(int)));
    }
};

int main(int argc, char** argv)
{
    CUDA_CHECK(cudaFree(0));
    g_csv = fopen("bench_results.csv", "w");
    if (g_csv) fprintf(g_csv, "op,width,height,param,mine_ms,opencv_ms,speedup\n");

    Timer tm;
    const struct { int w, h; } sizes[] = {{640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};

    for (const auto& sz : sizes) {
        const int W = sz.w, H = sz.h;
        const int reps = std::max(5, std::min(50, (int)(6e8 / ((double)W * H))));
        const int warmup = std::max(3, reps / 5);

        cv::Mat gray = randU8(W, H, false, 7);
        cv::Mat bin = randU8(W, H, true, 8);
        cvo::GpuMat dGray(gray), dBin(bin), dOutCV;

        Bufs b; b.alloc(W, H);
        CUDA_CHECK(cudaMemcpy2D(b.u8, b.u8step, gray.data, gray.step, W, H, cudaMemcpyHostToDevice));

        // ---------- 积分图 ----------
        {
            cvo::integral(dGray, dOutCV);
            double cvms = tm.run([&] { cvo::integral(dGray, dOutCV); }, warmup, reps);
            double mine = tm.run([&] { gpu_integral(b.u8, b.u8step, b.u32, b.u32step, W, H); }, warmup, reps);
            report("integral", W, H, "", mine, cvms);
        }

        // ---------- 均值滤波 ----------
        for (int k : {3, 15, 31}) {
            auto f = cvo::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), cv::Point(-1, -1));
            double cvms = tm.run([&] { f->apply(dGray, dOutCV); }, warmup, reps);
            double mine = tm.run([&] { gpu_mean(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k); }, warmup, reps);
            report("mean", W, H, "k=" + std::to_string(k), mine, cvms);
        }

        // ---------- 腐蚀/膨胀（二值 + 灰度） ----------
        for (int k : {3, 15, 31}) {
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
            auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker);
            auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, ker);
            double cve = tm.run([&] { fe->apply(dBin, dOutCV); }, warmup, reps);
            double cvd = tm.run([&] { fd->apply(dBin, dOutCV); }, warmup, reps);
            double mine = tm.run([&] {
                gpu_erode(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k);
                gpu_dilate(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k);
            }, warmup, reps);
            // 拆开报告：erode/dilate 各自时间 = 合计时一半近似；分别计时更准
            double me = tm.run([&] { gpu_erode(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k); }, warmup, reps);
            double md = tm.run([&] { gpu_dilate(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k); }, warmup, reps);
            char p[16]; snprintf(p, 16, "k=%d bin", k);
            report("erode", W, H, p, me, cve);
            report("dilate", W, H, p, md, cvd);
        }
        for (int k : {15, 31}) {
            cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
            auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_8UC1, ker);
            auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_8UC1, ker);
            double cve = tm.run([&] { fe->apply(dGray, dOutCV); }, warmup, reps);
            double cvd = tm.run([&] { fd->apply(dGray, dOutCV); }, warmup, reps);
            double me = tm.run([&] { gpu_erode(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k); }, warmup, reps);
            double md = tm.run([&] { gpu_dilate(b.u8, b.u8step, b.u8, b.u8step, W, H, k, k); }, warmup, reps);
            char p[16]; snprintf(p, 16, "k=%d gray", k);
            report("erode", W, H, p, me, cve);
            report("dilate", W, H, p, md, cvd);
        }

        // ---------- 高斯 ----------
        for (int k : {5, 7, 15}) {
            auto f = cvo::createGaussianFilter(CV_8UC1, CV_8UC1, cv::Size(k, k), 0, 0);
            double cvms = tm.run([&] { f->apply(dGray, dOutCV); }, warmup, reps);
            double mine = tm.run([&] { gpu_gaussian(b.u8, b.u8step, b.u8, b.u8step, W, H, k, 0.f); }, warmup, reps);
            report("gaussian", W, H, "k=" + std::to_string(k), mine, cvms);
        }

        // ---------- CV_32F 输入 ----------
        {
            cv::Mat grayF(H, W, CV_32FC1);
            for (int y = 0; y < H; ++y)
                for (int x = 0; x < W; ++x)
                    grayF.at<float>(y, x) = (float)gray.at<uchar>(y, x);
            cvo::GpuMat dGrayF(grayF), dOutF;
            float* dMineF; size_t mfstep;
            CUDA_CHECK(cudaMallocPitch(&dMineF, &mfstep, (size_t)W * 4, H));

            // 注：OpenCV-CUDA 没有 f32 积分图（cuda::integral 仅支持 CV_8UC1），无法同卡对照，
            // integral-f32 不进对照表；均值/形态学/高斯的 f32 均可对照。
            double cvms = 0.0, mine = 0.0;

            for (int k : {3, 15, 31}) {
                auto f = cvo::createBoxFilter(CV_32FC1, CV_32FC1, cv::Size(k, k), cv::Point(-1, -1));
                cvms = tm.run([&] { f->apply(dGrayF, dOutF); }, warmup, reps);
                mine = tm.run([&] {
                    gpu_mean_f32((const float*)dGrayF.data, dGrayF.step, dMineF, mfstep, W, H, k, k);
                }, warmup, reps);
                report("mean-f32", W, H, "k=" + std::to_string(k), mine, cvms);
            }
            for (int k : {3, 15, 31}) {
                cv::Mat ker = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(k, k));
                auto fe = cvo::createMorphologyFilter(cv::MORPH_ERODE, CV_32FC1, ker);
                auto fd = cvo::createMorphologyFilter(cv::MORPH_DILATE, CV_32FC1, ker);
                double cve = tm.run([&] { fe->apply(dGrayF, dOutF); }, warmup, reps);
                double cvd = tm.run([&] { fd->apply(dGrayF, dOutF); }, warmup, reps);
                double me = tm.run([&] {
                    gpu_erode_f32((const float*)dGrayF.data, dGrayF.step, dMineF, mfstep, W, H, k, k);
                }, warmup, reps);
                double md = tm.run([&] {
                    gpu_dilate_f32((const float*)dGrayF.data, dGrayF.step, dMineF, mfstep, W, H, k, k);
                }, warmup, reps);
                char p[16]; snprintf(p, 16, "k=%d", k);
                report("erode-f32", W, H, p, me, cve);
                report("dilate-f32", W, H, p, md, cvd);
            }
            for (int k : {5, 15}) {
                auto f = cvo::createGaussianFilter(CV_32FC1, CV_32FC1, cv::Size(k, k), 0, 0);
                cvms = tm.run([&] { f->apply(dGrayF, dOutF); }, warmup, reps);
                mine = tm.run([&] {
                    gpu_gaussian_f32((const float*)dGrayF.data, dGrayF.step, dMineF, mfstep, W, H, k, 0.f);
                }, warmup, reps);
                report("gaussian-f32", W, H, "k=" + std::to_string(k), mine, cvms);
            }
            cudaFree(dMineF);
        }

        // ---------- 直方图 ----------
        {
            // 关键：前面形态学/均值测试把 b.u8 原地写成了大片等值平台区（opening 效应），
            // 平台数据会让 warp 内大量 lane 落同一 bin，共享内存原子 32 路串行化 → 直方图被测出假慢 4~5 倍。
            // 必须在计时前重传干净的随机灰度图，保证与 OpenCV（用未污染的 dGray）同口径。
            CUDA_CHECK(cudaMemcpy2D(b.u8, b.u8step, gray.data, gray.step, W, H, cudaMemcpyHostToDevice));
            double cvb = tm.run([&] { cvo::calcHist(dBin, dOutCV); }, warmup, reps);
            double cvg = tm.run([&] { cvo::calcHist(dGray, dOutCV); }, warmup, reps);
            CUDA_CHECK(cudaMemcpy2D(b.u8, b.u8step, bin.data, bin.step, W, H, cudaMemcpyHostToDevice));
            // 新契约：hist kernel 完全覆写全部 256 bin，无需（也不再）预清零 —— 旧版计时内的
            // cudaMemsetAsync 实测固定多花 ~20us，比直方图本体还贵（见 probe_h4）
            double mb = tm.run([&] {
                gpu_histogram_binary(b.u8, b.u8step, W, H, b.hist);
            }, warmup, reps);
            // 灰度版换回干净随机数据（binary 数据只有 0/255 两种取值，会人为放大原子冲突）
            CUDA_CHECK(cudaMemcpy2D(b.u8, b.u8step, gray.data, gray.step, W, H, cudaMemcpyHostToDevice));
            double mg = tm.run([&] {
                gpu_histogram(b.u8, b.u8step, W, H, b.hist);
            }, warmup, reps);
            report("hist-binary", W, H, "", mb, cvb);
            report("hist-gray", W, H, "", mg, cvg);
        }

        cudaFree(b.u8); cudaFree(b.u32); cudaFree(b.hist);
        printf("\n");
    }

    if (g_csv) fclose(g_csv);
    printf("done. results -> bench_results.csv\n");
    return 0;
}
