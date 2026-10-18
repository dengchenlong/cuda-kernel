# 高性能 CUDA 算子库 — 进度与计划（内部开发记录，含本机路径，发布时可省略）

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
4. [x] 逐算子疯狂优化直到稳定 ≥ OpenCV（三轮优化后全算子×全尺寸×全参数 WIN，见 BENCHMARK.md 第三轮）
5. [x] 收尾：README.md + BENCHMARK.md 数据齐全，全部算子达标（2026-09-28 项目完成）

## 优化思路备忘
- sm_61 无张量核；优化抓手：向量化访存 (uint4/float4)、warp shuffle 扫描、共享内存 tiling+无 bank conflict、分离算法降复杂度、占满 SM (grid 足够大)、-O3 --use_fast_math、常量内存存核参数
- 均值滤波杀手锏：先算积分图再 O(1) 出结果（OpenCV 用分离线性滤波，核越大我们优势越大）
- 形态学杀手锏：van Herk/Gil-Werman O(1)/像素（OpenCV CUDA 对大核是暴力窗）
- 直方图杀手锏：每线程 uint4 读 4 像素、共享 256-bin 私有直方图、跨块原子合并
- 对比口径：都含 H2D 就绪后的纯 GPU 时间（cudaEvent 包住 kernel 或含拷贝分两栏报告，明确标注）

## 当前下一步（cron 续作从这里接）
- [x] 环境搭建完成：VS2022BT v143 (cl 19.44) 装在 D:\VS2022BuildTools；ninja 用 VS 内置
- [x] 算子代码初版全部写完并 nvcc 编译通过（utf-8 BOM 必须，nvcc 加 -Xcompiler "/utf-8 /W3"）
- [x] OpenCV-CUDA 基准库编译完成（D:\opencv-cuda-build\install\x64\vc17\，含 opencv_cuda*.dll，2026-09-26 起 smoke/benchmark 全部依赖它运行）
- [x] 编译 smoke + benchmark 完成（CMake 全目标 build\smoke.exe / build\benchmark.exe，运行时 PATH 加 install\x64\vc17\bin）
- [x] 跑 smoke 修正确性：全部通过（integral/morph 全核全图 identical；gauss k=5/15 identical；hist identical；mean 残差 ≤1 舍入差异，smoke 容忍 md<=1）
- [x] 跑 benchmark 记录首轮数字进 BENCHMARK.md（2026-09-26 全尺寸全算子，见 BENCHMARK.md 表格）
- [x] 优化第一轮（2026-09-27 全部落地，smoke 全绿，BENCHMARK.md 已更新第二轮表格）：
       1. benchmark 修复：hist 计时前重传干净随机数据（此前被 morph/mean 原地写成平台区，等值数据
          令 warp 原子同 bin 串行化，hist 假慢 4~5 倍 —— 数据分布是直方图基准的第一坑）
       2. histogram：去掉破坏 warp 合并的 ILP-2 取数（16B load stride 32B 慢 7 倍），回归 uint4 步进 32
       3. mean：小核 k≤15 行遍直加窗口 + 列遍行分块（rpt 64@4K/16@小图）
       4. morphology：k≤7 全走 rows 直加 + cols 行分块（k=3 三寄存器滚动），VHG 只管 k≥9；静态 workspace
       5. gaussian：列遍行分块 rpt 上调（4K 用 64）
- [x] 逐算子优化（第二轮后剩余 LOSE，差距全部 <1.4x，按性价比排序）——三项全部完成：
       1. [x] mean k=3 (0.65~0.78x) → 1.17~1.71x：弃两遍分离，改单遍 3x3 直加（smem tile 32x8，
              有效流量 6B/px→~2B/px），整数窗口和与分离法一致，输出逐位相同
       2. [x] hist-gray 1080p/4K (0.93~0.95x) → 1.35~2.86x：真瓶颈不是 smem 原子（probe_h2/h3 证伪：
              warp 私有 hist、2/4 uint4 取数均在噪声内），而是计时内 cudaMemsetAsync 预清零固定
              ~20us；改 last-block 归约（block 写 partials → atomicInc 回绕计数 → 末块汇总覆写全
              256 bin），契约改为"调用前无需清零"，grid 固定 48（6SM×8 最优）
       3. [x] gaussian k=5 (0.94~0.99x) → 1.37~1.60x：固定核 [1,4,6,4,1]/16 走定点整数路径，
              u16 tmp 存未归一化行部分和（≤4080），列遍整数累加 ×1/256；OpenCV CV_32F 分离滤波
              在该核下中间值全为二进分数，浮点结果恰为 S_total/256，逐位一致（smoke identical）
       附带：hist-binary 同步改 last-block，2~4.9x → 2.5~6.5x
- [x] 增补（2026-09-29）：gaussian k=7 定点整数路径（sigma<=0），弃错误的首版（未缩放行和×
      缩放权重=输出×64 饱和），改整数 MAC + ×(1/4096)，smoke identical；bench 加 k=7 列，
      全尺寸 1.35~1.60x WIN，BENCHMARK.md 第四轮
- [x] CV_32F 输入格式支持（2026-10-09，gpu_integral_f32 / gpu_mean_f32 / gpu_erode_f32 /
      gpu_dilate_f32 / gpu_gaussian_f32）：
      - integral f32：OpenCV-CUDA 无此 API（cuda::integral 仅 8UC1），参照用 float 模拟
        （行前缀+列前缀，CPU cv::integral 是 double 累加器，语义不同见 probe_int_f32b），
        kernel 逐位一致
      - mean f32：分离两遍（行 tile 前缀和 + 列滑动），与 OpenCV boxFilter(CV_32FC1) 差
        ~1.5e-5（其内部累加序不同），WIN 1.25~19x（大核优势极大）
      - morph f32：两遍直加；第一版列遍原地读写 dst 有 block 间竞态（smoke diff≈100+ 抓出），
        修正为读 tmp；与 OpenCV f32 形态学逐位一致，WIN 2.2~12.9x
      - gauss f32：显式 fmaf 后与 OpenCV 逐位一致；RT 大图 128 占用率更高；
        720p k=5 偶发 0.97x（±3% 计时噪声），其余全 WIN（1.0~1.8x）
      - smoke 新增 f32 全算子组（all pass），bench 新增 f32 组 22 项（119 WIN / 1 噪声 lose）
- 注意：git 仓库不能破坏，只许 add/commit，禁止 reset/checkout 丢工作；每完成一阶段 commit 一次
- [x] 全部算子稳定 ≥1（2026-09-28 达成，全算子×全尺寸×全参数 WIN，最低 1.087x）
- [x] 收尾（阶段 5）：README.md 已写（构建方法、API、benchmark 复现步骤、结果摘要、踩坑总结）
- [x] 项目全部完成（2026-09-28）：6 算子全尺寸稳定超过 OpenCV-CUDA；后续如续作可做非 benchmark 路径
      的补充核（如 hist 非 16 倍数宽度的向量化、k=7 高斯定点路径、CV_32F 输入格式），非必需
- [x] 增补（2026-09-30）：histogram 任意宽度向量化路径（W 非 16 倍数/不对齐的 hist_vec_lastblock_kernel，
      头尾标量 + uint4 主体合并读 + per-warp 私有 hist + last-block 归约），替代旧纯标量回退。
      调试修两坑：lane 分工致 8 warp ×8 重复计数（probe_hfix11）；对齐偏移 a 未 clamp 到 W 越界读
      （probe_hfix12）。probe_hfix14 280 用例全过、smoke 全绿；任意宽度 vs OpenCV 1.4~3.2x，
      vs 旧标量 1.1~1.2x（见 BENCHMARK.md 第五轮）。commit 1167707。
- 注意：git 仓库不能破坏，只许 add/commit，禁止 reset/checkout 丢工作；每完成一阶段 commit 一次
- 状态（2026-10-15）：spike-hist4k（sm_120 服务器架构优化，spike 001~003c）已合并进 main（--no-ff，
  e9b37f1）并 push；服务器（RTX 5090 D）120 项全 WIN。合并后本机健康巡检 r14（1050 Ti）：
  smoke 全绿，bench 112 WIN / 8 lose（单轮口径），lose 集中在 erode/dilate k=3 bin @720p/1080p/4K
  （0.87~0.99x）与 mean k=3 @4K（0.96x），与合并前 12 轮聚合结论一致（确定性 lose 5 项，其余计时噪声）。
  本机基准结论：120 项中 115 项稳定 WIN，5 项确定性 lose（morph k=3 小核、mean k=3 大图）需开新 spike
  （先服务器 probe，本机复验）才可攻克；除非用户提出新需求，不再自行启动新开发。
  后续 cron 续作回归周期性健康巡检（smoke + benchmark）确认无回退。
- 巡检 r15/r16（2026-10-16，1050 Ti）：smoke 全绿（0 flagged）。r15 118 WIN / 2 lose（hist-gray 720p 0.91x、
  1080p 0.78x）；r16 119 WIN / 1 lose（hist-gray 4K 0.85x）。
- 十轮聚合判定（2026-10-17，r15~r24 各 120 项，数据 docs/bench/bench_sm61_r15~r24.csv）：
  1. **hist-gray 1080p/4K 判定为非确定性 lose，不开 hist 小图 spike**。1080p lose 2/10（0.78/0.85，
     中位 1.08）、4K lose 4/10（0.85~0.95，中位 1.08）；lose 在轮次间不定点复现，且整个 hist 家族
     （gray/binary × 四尺寸）都呈现同类临界摆动——每项 lose 2~5/10、中位全部 >1.0，其中 hist-gray 720p
     最差（5/10，r19 0.74、r24 0.69）。~0.1 ms 量级内核计时噪声主导，OpenCV 对应项也在 0.06~0.19 ms。
  2. **r14 结论修正：morph k=3 小核与 mean k=3 @4K 的「确定性 lose」在 r15~r24 十轮中零复现**
     （morph k=3 各口径 min ≥1.10，mean k=3 @4K min 1.13），合并 spike-hist4k 后实际已无该项 lose，
     「本机 5 项确定性 lose」的说法作废。十轮中偶发 lose 仅剩 mean-f32 640 k=3（2/10，0.98~0.99）
     与 gaussian-f32 640 k=5（1/10，0.84），均为临界噪声。
  3. 结论：服务器口径（RTX 5090 D）120 项全 WIN；本机口径（1050 Ti）无确定性 lose，hist 家族为
     中位 WIN + 高方差临界项。benchmark 计时对 <0.2 ms 内核的分辨率有限，hist 若要「逐轮全 WIN」
     需换更稳的计时法（如固定 clock/更多重复），属可选优化，不自行启动。
- 巡检 r17~r24（2026-10-17，1050 Ti）：smoke 全绿（0 flagged）；8 轮 lose 1~5 项/轮，全部落在 hist 家族
  与 f32 临界项，无新回退模式。
- 巡检 r25/r26（2026-10-18，1050 Ti）：smoke 全绿（0 flagged，120 [OK]）；r25 lose 2（hist-binary/gray
  640 0.72/0.75）、r26 lose 3（hist-binary 1080p 0.83、hist-gray 1080p 0.88、hist-gray 4K 0.81），
  仍全部落在 hist 家族。
- 巡检 r27/r28（2026-10-19，1050 Ti）：smoke 全绿（0 flagged）；bench 两轮均 120/120 全 WIN
  （十四轮以来首次单轮零 lose），CSV 落库 docs/bench/bench_sm61_r27/r28.csv。hist 家族临界摆动
  本轮未复现，进一步支持「计时噪声、非确定性 lose」判定。
- 十二轮聚合确认（r15~r26，2026-10-18）：有 lose 的项共 10 个，全部为 hist 家族（8 项，各 lose 2~5/12、
  中位全部 >1.0）加 mean-f32 640 k=3（2/12，中位 1.25）与 gaussian-f32 640 k=5（1/12，中位 1.25）两个
  临界噪声项。无任何项构成确定性 lose，维持「不开 hist spike」判定。解析坑：bench CSV 的 speedup 是第 7 列
  （索引 6），误用 opencv_ms 列会得到 40 项假 lose。
- 经验：cmd /c 长命令在 git-bash 下要用 MSYS_NO_PATHCONV=1 cmd /c；.cu 文件写中文注释必须带 BOM；
  同一 .cu 内多个 extern __shared__ 不能重名异型（float ring[] 与 uint16_t ring[] 冲突，改名即可）；
  直方图性能排查先看"计时内预清零"这类隐形固定开销（~20us 比内核本体还贵），再看 smem 原子冲突；
  单轮 lose 判定不可靠——hist/f32 临界项需 ≥10 轮聚合，lose 若不定点复现即为计时噪声而非确定性 lose
