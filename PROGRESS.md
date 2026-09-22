# 高性能 CUDA 算子库 — 进度与计划

> 本文件是 cron 自动续作的依据。每次会话开工前先读本文件，做完一步更新一步。
> 目标：在 GTX 1050 Ti (sm_61) 上实现以下算子，性能全面超过 OpenCV-CUDA（G:\opencv 源码编译，同机同卡基准）。

## 算子清单（全部完成才算完）
- [ ] 积分图 (integral image, u8 in → u32 out)
- [ ] 均值滤波 (box mean, 任意核尺寸, u8/u32f)
- [ ] 二值腐蚀膨胀 (u8 0/255, 任意核)
- [ ] 灰度腐蚀膨胀 (u8, 任意核)
- [ ] 高斯滤波 (u8→u8, 常用 sigma 1~4)
- [ ] 灰度直方图 (u8, 256 bins)

## 环境事实
- GPU: GTX 1050 Ti 4GB, sm_61 (Pascal), 驱动 582.66
- CUDA Toolkit 12.5: D:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5
- OpenCV 4.x 源码: G:\opencv（勿动其源码；旧 build/ 是 WSL 孤儿构建，留着别删）
- 编译器: 需 VS2022 BuildTools v143（正在装到 D:\VS2022BuildTools），VS2026 cl19.50 与 CUDA12.5 不兼容
- CMake: D:\Program Files\CMake\bin\cmake.exe；ninja 在 VS 目录里
- 磁盘: C 盘仅 ~7G 禁用；D 盘 ~141G（放 OpenCV 构建与下载）；G 盘 ~10G（源码/临时小文件）
- OpenCV-CUDA 构建目录（新）: D:\opencv-cuda-build，BUILD_LIST=core,imgproc,cudaarithm,cudawarping,cudaimgproc,cudafilters, arch=6.1

## 仓库结构（G:\cuda-kernel）
```
G:\cuda-kernel
├── PROGRESS.md            本文件
├── CMakeLists.txt         (重写)
├── deps\                  下载的依赖
├── src\
│   ├── common.cuh         共享工具 (时序、图像生成、校验)
│   ├── integral.cu        积分图
│   ├── mean_filter.cu     均值滤波（积分图法 O(1)/像素）
│   ├── morphology.cu      二值/灰度 腐蚀膨胀（van Herk/Gil-Werman 分离算法）
│   ├── gaussian.cu        可分离高斯
│   └── histogram.cu       直方图（uint4 向量化+共享内存私有化）
├── bench\
│   └── benchmark.cu       对照 benchmark：vs OpenCV-CUDA + CPU 基准
└── build\                 构建输出
```

## 阶段
1. [进行中] 环境搭建：装 VS2022BT → 编 OpenCV-CUDA (D:\opencv-cuda-build)
2. [ ] 逐算子实现+正确性验证（vs OpenCV 结果逐像素比对）
3. [ ] benchmark 框架：cudaEvents 计时，多分辨率 {640x480, 1280x720, 1920x1080, 3840x2160}，二值图/灰度图/自然图，预热+多次重复取中位
4. [ ] 逐算子疯狂优化直到稳定 ≥ OpenCV（记录每次优化前后的数字进 BENCHMARK.md）
5. [ ] 收尾：README + BENCHMARK.md 数据齐全，全部算子达标

## 优化思路备忘
- sm_61 无张量核；优化抓手：向量化访存 (uint4/float4)、warp shuffle 扫描、共享内存 tiling+无 bank conflict、分离算法降复杂度、占满 SM (grid 足够大)、-O3 --use_fast_math、常量内存存核参数
- 均值滤波杀手锏：先算积分图再 O(1) 出结果（OpenCV 用分离线性滤波，核越大我们优势越大）
- 形态学杀手锏：van Herk/Gil-Werman O(1)/像素（OpenCV CUDA 对大核是暴力窗）
- 直方图杀手锏：每线程 uint4 读 4 像素、共享 256-bin 私有直方图、跨块原子合并
- 对比口径：都含 H2D 就绪后的纯 GPU 时间（cudaEvent 包住 kernel 或含拷贝分两栏报告，明确标注）

## 当前下一步（cron 续作从这里接）
- 若 VS2022BT 未装完：等装完/重试安装
- 然后配置+编译 OpenCV-CUDA（ninja，后台跑，约 30-60 分钟）
- 编译期间开始写 src\ 下的算子代码
- 注意：git 仓库不能破坏，只许 git add/commit，禁止 checkout/reset 丢工作
