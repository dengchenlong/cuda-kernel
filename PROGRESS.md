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
2. [x] 逐算子实现+正确性验证（vs OpenCV 结果逐像素比对，smoke 全绿，mean ±1 舍入容差）
3. [x] benchmark 框架：cudaEvents 计时，多分辨率 {640x480, 1280x720, 1920x1080, 3840x2160}，二值图/灰度图/自然图，预热+多次重复取中位
4. [进行中] 逐算子疯狂优化直到稳定 ≥ OpenCV（记录每次优化前后的数字进 BENCHMARK.md）
5. [ ] 收尾：README + BENCHMARK.md 数据齐全，全部算子达标

## 优化思路备忘
- sm_61 无张量核；优化抓手：向量化访存 (uint4/float4)、warp shuffle 扫描、共享内存 tiling+无 bank conflict、分离算法降复杂度、占满 SM (grid 足够大)、-O3 --use_fast_math、常量内存存核参数
- 均值滤波杀手锏：先算积分图再 O(1) 出结果（OpenCV 用分离线性滤波，核越大我们优势越大）
- 形态学杀手锏：van Herk/Gil-Werman O(1)/像素（OpenCV CUDA 对大核是暴力窗）
- 直方图杀手锏：每线程 uint4 读 4 像素、共享 256-bin 私有直方图、跨块原子合并
- 对比口径：都含 H2D 就绪后的纯 GPU 时间（cudaEvent 包住 kernel 或含拷贝分两栏报告，明确标注）

## 当前下一步（cron 续作从这里接）
- [x] 环境搭建完成：VS2022BT v143 (cl 19.44) 装在 D:\VS2022BuildTools；ninja 用 VS 内置
- [x] 算子代码初版全部写完并 nvcc 编译通过（utf-8 BOM 必须，nvcc 加 -Xcompiler "/utf-8 /W3"）
- [ ] OpenCV-CUDA 基准库编译中：后台脚本 D:\deps\build_opencv_cuda.bat → D:\opencv-cuda-build
       （前两次失败原因：① 未加 WITH_CUDA=ON ② contrib cudev 缺失，已修正，含 contrib 路径 G:\opencv_contrib）
       编完检查 D:\opencv-cuda-build\install\x64\vc17\bin\ 应有 opencv_cuda*.dll
- [ ] 编译 smoke + benchmark（需要 OpenCV lib 就绪）：nvcc 命令参考 D:\deps\compile_gpuops.bat，
       链接加 -L"D:/opencv-cuda-build/install/x64/vc17/lib" -lopencv_core4130 -lopencv_imgproc4130 -lopencv_cudaarithm4130 -lopencv_cudafilters4130 -lopencv_cudaimgproc4130 -lopencv_cudawarping4130，运行时 PATH 加 install\x64\vc17\bin
- [x] 跑 smoke 修正确性：全部通过（integral/morph 全核全图 identical；gauss k=5/15 identical；hist identical；mean 残差 ≤1 舍入差异，smoke 容忍 md<=1）
- [x] 跑 benchmark 记录首轮数字进 BENCHMARK.md（2026-09-26 全尺寸全算子，见 BENCHMARK.md 表格）
- [x] 优化第一轮（2026-09-27 全部落地，smoke 全绿，BENCHMARK.md 已更新第二轮表格）：
       1. benchmark 修复：hist 计时前重传干净随机数据（此前被 morph/mean 原地写成平台区，等值数据
          令 warp 原子同 bin 串行化，hist 假慢 4~5 倍 —— 数据分布是直方图基准的第一坑）
       2. histogram：去掉破坏 warp 合并的 ILP-2 取数（16B load stride 32B 慢 7 倍），回归 uint4 步进 32
       3. mean：小核 k≤15 行遍直加窗口 + 列遍行分块（rpt 64@4K/16@小图）
       4. morphology：k≤7 全走 rows 直加 + cols 行分块（k=3 三寄存器滚动），VHG 只管 k≥9；静态 workspace
       5. gaussian：列遍行分块 rpt 上调（4K 用 64）
- [ ] 逐算子优化（第二轮后剩余 LOSE，差距全部 <1.4x，按性价比排序）：
       1. mean k=3 (0.65~0.78x) —— 疑似 u16 tmp 往返带宽；试 rows 融合输出（行和+列和两遍合一或 u16→u8 单遍）
       2. hist-gray 1080p/4K (0.93~0.95x) —— 与 OpenCV 同 layout 下差 5%；试每 warp 双行、u32 载入
       3. gaussian k=5 (0.94~0.99x) —— 全尺寸贴 1.0；试 u16 定点中间缓冲降带宽
- [ ] 全部算子稳定 ≥1 后：README + 收尾（阶段 5）
- 注意：git 仓库不能破坏，只许 add/commit，禁止 reset/checkout 丢工作；每完成一阶段 commit 一次
- 经验：cmd /c 长命令在 git-bash 下要用 MSYS_NO_PATHCONV=1 cmd /c；.cu 文件写中文注释必须带 BOM
