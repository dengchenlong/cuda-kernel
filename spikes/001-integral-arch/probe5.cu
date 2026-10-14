// probe5: 复现 probe1 的 bad rows=5 —— 怀疑点：probe1 用 memset 后跑 v2，但 v0 基准先跑、
// v2 随后跑，p2 是 in-place（in==out），carry buffer 复用同一 ws，v0/v1/v2 交错后 carry 槽里
// 残留了上一次 launch 的数据？不对——carry 每次 launch 全量覆写（nChunks>1 时每 chunk 都写）。
// 真正可疑点：probe1 的对照是 cvr.convertTo(cv32) —— OpenCV cuda::integral 输出 CV_32SC1，
// 但 dOut GpuMat 内部 step 有 4 字节对齐 pad（W=3840×4=15360 本来就 16 对齐）；
// download 后 cvr 是 CV_32SC1 连续，convertTo 恒等。等等——OpenCV 的 integral 语义是
// (W+1)x(H+1) 包含第 0 行/列的 inclusive 前缀？cuda::integral 输出是 W×H 还是 (W+1)×(H+1)？
// OpenCV cv::cuda::integral 输出尺寸 = (W+1)x(H+1)！probe1 的对照用了 (y,x) 索引同一尺寸，
// 必然错位 → bad rows 就是“首行/首列偏移”，跟实现无关！
// 本 probe 验证：打印 OpenCV 输出尺寸。
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cstdio>

int main()
{
    cv::cuda::GpuMat d(16, 16, CV_8UC1, cv::Scalar(1));
    cv::cuda::GpuMat out;
    cv::cuda::integral(d, out);
    printf("cuda::integral output size: %dx%d (input 16x16)\n", out.cols, out.rows);
    cv::Mat cpu(16, 16, CV_8UC1, cv::Scalar(1));
    cv::Mat cpuOut, sumSq;
    cv::integral(cpu, cpuOut, sumSq, CV_32SC1);
    printf("cpu::integral output size: %dx%d\n", cpuOut.cols, cpuOut.rows);
    return 0;
}
