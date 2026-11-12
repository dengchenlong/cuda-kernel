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
- [x] 全部算子稳定 ≥1（2026-09-28 达成，全算子×全尺寸×全参数 WIN，最低 1.087x）
- [x] 下一步（2026-10-27 r41 后 → 2026-10-28 r42 完成）：**mean k=3 单遍融合**（BENCHMARK.md 第十一轮）
      - 旧 `mean3_direct_kernel`（32x8 smem tile + 每像素 9 次 smem 字节读）4K 1.10ms 只有 7.5 GB/s，
        且 halo 从 global 读 → **原地调用有跨 block 竞态**（probe_m3a：4K 69 万像素错、maxdiff 48；
        smoke 只测 out-of-place 所以多年没抓到）。
      - 新 `mean3_fuse_kernel`：16 px/线程、uint4、`__byte_perm` 展开 u16 lane + `__funnelshift_r`
        取左右邻 + `__vadd2`；垂直 3 行滚动；`(v*7282+32768)>>16` 除 9（0..2295 全量等价，probe 验证）；
        原地走 workspace + 整图 uint4 拷回。44 regs 无 spill。
      - **求和不能沿用 morph 的 clamp 边界**：min/max 的 min(a,a,b)==min(a,b) 让 clamp 蒙混过关，
        求和必须 REFLECT_101 镜像（左右取像素 1 / W-2，上下取 row 1 / H-2），否则 4K 11307 px 错、
        maxdiff 66。改镜像后 maxdiff=0，probe_m3b 80 用例（含极扁图/非 16 倍数宽回退/tight+4/in-place）全过。
      - smoke 新增 mean k=3/15/31 的 **in-place 三条用例**，防止这类「只在原地口径暴露」的 bug 再溜过去。
      - 结果：mean k=3 1.13~1.30x → **2.60~3.25x**；r42 全量 120/120 WIN 零 lose，min 1.235x，中位 4.49x。
- [x] 下一步（2026-10-29 r43 后 → 2026-10-31 r44 完成）：gaussian k=7 单遍融合（BENCHMARK.md 第十三轮）
      - 病灶：两遍定点路径 6B/px，4K 1.5718ms 只有 31.7 GB/s（拷贝天花板 93 GB/s / 0.183ms）。
      - 定案 `gauss7_fuse_kernel`：8 px/线程 + uint2 取数；**水平 7-tap 用 `__dp4a`**
        （sm_61 起支持 u8x4 无符号点积，每像素 2 条 = 7 tap 拆 4+3、末位权重 0；窗口字由
        12 条 `funnelshift_r` 从同一批 uint2 错位重排得到），**垂直必须 u32 lane**
        （Σ w_j h_j ≤ 64×16320 = 1,044,480 > 65535，k=5 的 `__vadd2` u16 双 lane 打包会溢出），
        `h[7][8]` 滚动窗口 + IMAD 立即数权重；输出 `__float2int_rn(S × 2^-12)` 与两遍路径逐位一致。
      - 原地：融合写 workspace + 整图 uint4 拷回（4.1B/px）；4K 原地 0.63 vs out-of-place 0.46。
      - 验证：probe_g7f **95 条断言全过**（19 形状 × {out/in-place, 紧排}，双参照：OpenCV-CUDA +
        本机 int64 精确参考 round-half-even(S/4096)，都要求 diff=0）；常量 0/1/127/128/255 极值 diff=0
        （全 255 恰达 S 上限）；smoke 120 [OK]。
      - **角落发现（既有行为）**：H<4 时旧两遍路径的「单次折叠 + clamp」与 REFLECT_101 周期镜像
        本就不等价（8x2 vs OpenCV diff=10、8x3 diff=3），融合路径对 H=3 也会差 3 → 融合加 **H>=4**
        门槛，H<4 保持旧输出不变（不为非 benchmark 角落改既有行为）。
      - 结果：k=7 1.27~1.61x → **3.20~3.73x**；r44 全量 120/120 WIN 零 lose，min 1.237，中位 4.509x，
        p10 2.156x，CSV 落 docs/bench/bench_sm61_r44.csv。
- [x] 下一步（2026-10-31 r44 后 → 2026-11-02 r45 完成）：② **gaussian-f32 k=5 / mean-f32 k=3 单遍 f32 融合**
      （BENCHMARK.md 第十四轮）—— 原估 ~1.4x，实测 2.6~3.6x，因为两遍 f32 路实际是 16B/px 不是 8B/px；
      顺带修掉 `cudaMemcpyToSymbolAsync` 上传权重的 11us 固定开销。详见文末 r45 记录。
- [x] 下一步（2026-11-02 r45 后 → 2026-11-03 r46 完成）：原地路径去整图拷回 —— 用「只暂存 halo 行」的
      stash 取代整图 workspace 拷回（原计划的 smem tile 只解决块内，跨 block halo 仍不安全，见 r46 记录）。
      结果 u8 高斯 k=5 1080p 0.170→0.140、4K 0.544→0.416；k=7 因 4K 偶发 diff 暂留原路径。
- [x] 下一步（2026-11-03 r46 后 → 2026-11-04 r47 完成）：**stash 推广到 mean k=3 / morph k=3**
      （BENCHMARK.md 第十五轮，两族共 12 项）。
      - 把 r46 的 halo 行 stash 抽成 `src/halo_stash.cuh`（按 R 模板：每 chunk 边界 2R 行窗口 + 末块
        R+2 行底段 + 16B 对齐行距 + 填充内核）；mean/morph 的 k=3 融合内核加 `bool STASH` 模板参数，
        原地且 H>=1000 走 stash，小图仍走整图 workspace 拷回（barrier 固定代价在小图上反超收益）。
      - 两族的边界语义不同但槽位映射相同：mean 严格 REFLECT_101（末行 y=H-1 要读 H-2，而 H-2 已被
        自己写掉 → 正是底段存在的理由），morph 是 clamp（等价 min/max 恒等元）。
      - 块内 warp 超车同样靠「每行一次 __syncthreads + 预载后补一次」压 skew（r46 的教训直接复用）。
      - 结果：mean k=3 1080p 0.1239→0.0850（2.74→3.94x）、4K 0.4444→0.2550（2.81→5.01x）；
        erode k=3 1080p 0.1034→0.0685（3.69→5.53x）、4K 0.3738→0.2027（3.71→6.83x）；
        dilate k=3 1080p 0.1034→0.0798（3.72→4.98x）、4K 0.3727→0.2048（3.71→6.81x）。
        4K 原地 4.1B/px → 2.1B/px（stash 71 行 = 273KB vs 整图 8.3MB）；原地 vs 原地外 ratio 0.96~1.07。
      - 验证：probe_s3 **75 用例**（25 形状 × 3 算子，含 16x1/16x2/16x3 极扁、H=999/1000 阈值两侧、
        3840x2160、W%16!=0 回退对照）原地 vs 原地外**逐位一致**且连跑两次相同；`CK_FORCE_STASH=1`
        强制口径再跑一遍同样全 0；smoke 120 [OK]；r47 全量 **120/120 WIN 零 lose**，min 1.094
        （hist-gray 1080p 噪声），p10 2.524，中位 4.522，CSV 落库 docs/bench/bench_sm61_r47.csv。
- [x] 下一步（2026-11-04 r47 后 → 2026-11-05 r48 完成）①：**k=7 原地偶发 diff 定位 = 水平 halo 跨 block**
      （commit 94baa99，`docs/probes/probe_s7.cu`）
      - **根因（不是末块边界、不是 y+R+1 折回）**：k=7 融合内核是 **8 px/线程**，`th` 上限 256 → 一个
        block 只覆盖 `8*256 = 2048` 列。W=3840 时 gridDim.x=2，块 1 的 lane0 走
        `rp[gxl-4]`（= x=2044..2047，属于块 0），而原地口径下块 0 可能已把这 4 个像素写掉 →
        错像素**永远落在 x=2048/2049 两列**（probe_s7 稳定复现，y 随机、maxdiff 12~13）。
        k=5 / mean / morph 是 16 px/线程 × 256 = 4096 列，W≤4096 天生单 block，所以从没暴露。
        r46 的「非确定 bad 6~50」正是这个两列条带在不同 y 上时隐时现。
      - 修法：`gauss7_fused` 里 `th` 上限按模板参数分档 —— `STASH ? 512 : 256`
        （STASH 时保证 `8*th ≥ W`，即整行一个 block；out-of-place 保持 256 线程多 block，
        parallelism 更好不要跟着放大）。**教训：原地卷积内核的前提不只是「读领先写 + 每行 barrier」，
        还要求「一行的水平 halo 不跨 block」**，否则块间无顺序保证这条会从列方向漏进来。
      - **修好后 stash 仍然不划算**：7-tap/8px 的块要 512 线程才保得住单 block，块数腰斩、并行度掉，
        probe_s7 sweep（交错 A/B、min 口径，3 轮）：1080p stash 0.1987 vs copy 0.1843（**慢 8%**）、
        4K 1.1889 vs 1.1940（打平）。barrier + stash 填充把 copy-back 的 ~0.17ms 全吃回去了。
        → **k=7 默认仍走「写 workspace + 整图拷回」**（`stash7_enabled()` 只认 `CK_FORCE_STASH=1`），
        r48 不改默认路径、不产生性能变化。这条「预期 0.639→0.50」的假设被实测否掉。
      - 验证：probe_s7 `check`（12 形状 × {默认阈值, CK_FORCE_STASH=1} × 6 轮重跑）全 0；
        `chunk` 在 CK_TH7=8/16/32/64 四种 chunk 划分下同样全 0；smoke 120 [OK]；probe_g7f 95 条仍全过。
- [x] 下一步（2026-11-05 r48 后 → 2026-11-06 r49 完成）：mean/morph k=3 的 TH 分档重调 —— **试了，判定为
      噪声拟合，已回退（净代码变化 0）**。新探针 `probe_th3`（sweep 扫 TH=8/16/32/64 交错 A/B、min 口径）
      给的信号很漂亮（640 TH8 比 TH16 快 26%），但同一二进制重编两次跑全量 bench 结论互相打脸；
      查到本机空闲时钟在 **607 ↔ 1493 MHz** 之间爬升，0.02~0.25ms 量级的原地微基准在这个漂移下没有分辨力。
      处置：TH 恢复 r42/r47/r48 原值，只在 `src/mean_filter.cu` / `src/morphology.cu` 留 `CK_TH3` 覆盖钩子
      （仅 probe 用）+ 给 morph 补 TH=32 模板分支。k=7 原地 stash 也做了同 TH 对照复核（`probe_s7 k7scan`，
      修掉上一版把 `_putenv_s` 放进计时区的错误）：四档尺寸 stash 只在 1080p/4K 赢 2~4%、640/720p 输 5~30%，
      全在噪声内 → 维持「默认走 copy」。
- [x] 下一步（2026-11-06 r49 后 → 2026-11-07 r50 完成）：**integral 两段都重写**（BENCHMARK.md 第十七轮）
      - 分段实测（`probe_int1`，之前从没做过）：4K 合计 2.43ms = **pass1 1.19 + pass2a 0.71 + pass2b 0.54**。
        pass2a/2b 已跑 93~98 GB/s（贴天花板），唯一大空间在 pass1（**31 GB/s**）和 pass2 的结构本身。
      - **病灶 1（pass1 写粒度）**：每 lane 负责 16 字节、前缀和逐条 u32 标量存 → 一条 store 的 32 个 lane
        落在 32 个相隔 64B 的地址上（32 个不同 sector，每个只写 4/32 字节），写放大把带宽压到 1/3。
        改成「16 个前缀先算进寄存器，再用 4 条 uint4 store 写回」：4K 1.316→0.468ms（**89 GB/s，2.8x**），
        数值路径零改动（同一批 run 值）。附带：`rows_seg`（每 lane 4B、32 lane 连续 128B）同速，
        说明按 128B 对齐就够了，不必最小化单指令字节数。
      - **病灶 2（pass2 白做一遍全图）**：旧策略「CHUNK_H=512 分块列扫描 + carry 回加」在 sm_61 上是净亏损 ——
        列扫描要读 4N 写 4N，回加再读 4N 写 4N = 16N 字节；而列扫描本来就不需要分块（每线程独占自己的列，
        依赖只在累加器里）。改成单遍「每线程 1 列走完 H 行」：4K 1.238→**0.706ms**（93 GB/s）。
        代价是并行度只剩 W 线程，所以保留分块路径作 W<512 的回退（实测 640 用 1col 也比旧 4col 快 33%，
        阈值取 512）。顺带让「W%4!=0 / 非 16B 对齐但 4B 对齐」的宽图也走单遍（原先掉进标量回退）。
      - 结果（`probe_int1` 纯内核 + bench 双口径一致）：640 0.127→0.0666、720p 0.337→0.145、
        1080p 0.697→0.299、4K 2.426→**1.159**（2.09x）；bench 口径 integral 全档 **4.24~7.17x**（原先 1.76~5.04x）。
      - 验证：`probe_int2` **128 用例**全过（32 形状 × {pitch 步长, 紧排紧 step} × pad，参照 CPU int64 精确
        前缀和；含 W=511/512/513/1023/1025/1919/1921/3839/3841、W%4!=0、H=1、8x4000 极扁）；smoke 120 [OK]。
      - 已知遗留：pass1 与 pass2 仍是两遍（合计读 4N + 写 4N + 读 4N + 写 4N = 16N 字节，理想单遍是 8N），
        且都只到 89~93 GB/s（拷贝天花板）。再往上要不就融合成一遍（列方向跨 block 依赖麻烦），
        要不就接受现状 —— 现在 4.24x 已经不再是最低项（见下一节）。
- [x] 下一步（2026-11-07 r50 后 → 2026-11-08 r51 完成）①：**integral 融成单遍（6N 字节）**（BENCHMARK.md 第十八轮）
      - 病灶：r50 之后两段都到 89~93 GB/s（拷贝天花板），但总流量 13N（pass1 读1N写4N + pass2 读4N写4N），
        理想 5N。带宽到顶 ⇒ 唯一空间是砍流量。
      - 定案：把「上方行在 x 列的累加和」提前独立算出来（它只依赖 src，不依赖任何块输出）——
        A) `integral_chunk_sum_kernel` 按 TH 行切段求每段列和（只读 1N，写 512KB 级小数组）；
        B) `integral_chunk_scan_kernel` 每线程一列做跨 chunk exclusive 前缀；
        C) `integral_fused_kernel` 每 block 一个 chunk，线程数 = ceil(W/16) 取整到 warp
        （640→64/1280→96/1920→128/3840→256），每线程 16 个寄存器列累加器 + 逐行块内行扫描，
        直接写 dst，无任何中间 4N 数组。总流量 6N，理论 2.17x。
      - 结果：640 0.0788→**0.0707**、720p 0.1546→**0.0952**、1080p 0.3102→**0.1720**、
        4K 1.1786→**0.6861**；bench 口径 integral **6.41~8.69x**（原 4.24~7.17x）。
        probe_int5 纯内核交错 A/B（3 轮 min）：4K 0.639 vs 1.160（1.82x）、1080p 1.83x、720p 1.62x。
      - **两个坑（都靠 probe 抓到）**：① 前缀扫描里**不能提前 return** —— W%16==0 但 W 非 512 倍数
        （128/640）时越界线程提前退出会让该 warp 不写 wsum[]，跨 warp 前缀读到未初始化值
        （smoke 128x128 maxdiff=253）；必须「保留线程 + valid 掩码」。② 部分 lane 参与的 shfl
        必须写对 mask（`if (tid<8)` 里要用 `0xff`，写 `0xffffffff` 是 UB）。③ 流程坑：
        `build_probe.bat` 链接 `build/obj/*.obj`，改完 src 必须先 `build_all.bat` 再 `build_probe.bat`，
        否则探针跑的还是旧代码（本轮第一次误判「改完仍错」就是它）。
      - 验证：probe_int2 128 用例全过；新 probe_int3（7 组尺寸逐像素）全 0；probe_int4 错点分布工具
        （修 bug 时定位到「第 32 列起整行错」= 跨 warp 前缀）；smoke 120 [OK]。
      - 已知剩余：4K 0.686ms / 6N = 72 GB/s，仍低于 93 天花板，气泡在「每行一次 syncthreads +
        跨 warp smem 广播」；TH 扫参（16~512）差异 0.599~0.651，落在时钟漂移噪声带内，未据此改默认。
        小图 33~61 GB/s（nChunks 少吃不满 6 SM）。
- [x] 下一步（2026-11-08 r51 后 → 2026-11-09 r52 完成）①：**hist-gray 分段实测 —— 结论是伪命题，不改**；
      ②：**integral 融合路径 TH 形状修正**（BENCHMARK.md 第十九轮）
      1) **hist-gray「全局最低项」判为伪命题**（probe_hgray2）：分段实测（4K，grid 48×256）full 0.1034ms /
         read-only 0.0932（= 89 GB/s 拷贝天花板）/ empty 0.0102（纯启动）⇒ 16 条 smem 原子只值 10%，
         且 4 条与 0 条原子同速（被访存延迟掩盖）。probe_hgray3 又试了 4 份私有 hist、stride 257 错 bank、
         按字节序号分副本、grid 24/12、去归约尾共 8 个变体，**全部在噪声带内（±3%）**。
         真正原因是**对手侧**：用同一张随机图逐尺寸对照，OpenCV 4K 0.1577（r51 记的 0.1649）、1080p
         0.0952（r51 0.0850）、720p 0.0567（r51 0.0748）——它的项本身在 0.16±0.01 抖动，我们三轮
         几乎不动（0.1116/0.1126）。1.48x↔1.40x 的差别几乎全在 OpenCV 的时间里。**不再开 hist spike。**
      2) **integral 融合内核 4K 形状修正**：r51 融合内核 4K 0.686ms / 5N = 71 GB/s，而 probe_int9 实测
         「读 1N(u8)+写 4N(u32)」的纯搬运地板 = **0.4792ms / 86.5 GB/s** ⇒ 缺口在形状不在带宽。
         - probe_int6/8：ROWS=2（两行共享一次屏障）4K -12.7%（复现两次）但 1080p **-19%**（84 regs 掉占用率）⇒ 弃。
         - probe_int8：预取下一行（PF-R1，62 regs）、`__stcs` 流式写、NOBAR 对照（去掉屏障故意算错）——
           NOBAR 只快 4% ⇒ **屏障本身不是瓶颈**，前两者零收益。
         - probe_int10（5 轮交错 A/B，四尺寸 × TH=32..256）：**甜点 = 每 SM ~4 个常驻 block，
           即 nChunks ≈ 24**。4K fth=256、60 regs ⇒ 15360 regs/块、65536/SM ⇒ 4 块/SM；旧规则
           ceil(H/64) 给 34 个 chunk（每 SM 5.7 块），块多但每块只 64 行，固定成本占比升高。
         - 定案（一行改动）：`fused_th` 改 `th = align32(ceil(H/24))`，clamp [32,256]。640/720p/1080p
           本来就都是 TH=32（ceil(H/24)=20/30/45 → 取整 32），**只有 4K 从 64 变 96**。
           4K kernel 0.5663 → **0.4626ms（-18.3%，89.7 GB/s）**。
      - 结果：640 0.0707→0.0614、720p 0.0952→0.0901、1080p 0.1720→0.1669、4K 0.6861→**0.5704**；
        bench 口径 integral **8.56~9.96x**（原 6.41~8.69x）。r52 全量 **120/120 WIN 零 lose**，
        min 1.400、p10 2.450、中位 5.003、max 21.866，CSV 落 docs/bench/bench_sm61_r52.csv。
      - 验证：probe_int2 128 用例全过、probe_int3 7 组尺寸逐像素全 0（新默认下重跑）、probe_int10
        每个 TH 与参照逐位 diff=0、smoke 120 [OK]。两轮 bench 均 120/120 WIN（冷启动+暖机）。
      - 新标定：4K 纯内核 89.7 GB/s 已**越过纯搬运地板 86.5 GB/s**（融合版每线程连续扫 96 行，
        比搬运版每线程 1 行的 ILP 更好）；1080p 80 GB/s = 地板的 93%。积分图基本到顶。
- [ ] 下一步（2026-11-09 r52 后 → 2026-11-10 r53 完成 ①）：**morph 大核（k>7）重写 —— VHG 换 smem 走查**
      （BENCHMARK.md 第二十轮）——∫ 部分仍是「已到物理上限」，本轮实际做的是 r52 遗留清单外的
      morph 大核（它不是最低项，但绝对耗时最丑：4K k=15 单项 13.4ms）。
      1) **病灶（probe_mcol asym，之前从没做过这种拆分）**：非对称核拆行遍/列遍，4K k=15 行遍 12.42ms、
         列遍 12.37ms、(1,1) 基线 0.98ms，而 **k=31 与 k=15 几乎同速** ⇒ 与窗口长度无关，是 VHG 的
         固定结构开销：每 block 只产 128 像素却 load 3 段 + 42 次 `__syncthreads` 双扫描
         （4K 行遍 = 2160×30 = 6.5 万 block × 42 屏障）。**行遍访存完全合并，一样慢** ⇒
         原先「列遍 32 倍 sector 放大是主因」的猜测不成立（那只是叠加项）。
      2) **定案**：`morph_rows_line_kernel` / `morph_cols_line_kernel` —— 一线程一条线、整行/整列搬进
         smem，正走填 `U[c]`、反走算 `D[c]`（块长 = k），输出 `op(D[i], U[i+k-1])`：
         **O(1)/像素、零屏障链**（只剩 load/输出两次同步）。覆盖 k≤33，更大仍走 VHG。
         边界沿用 clamp 载入（min/max 恒等元与 clamp 等价），与旧 VHG 逐位一致。
      3) **结果**：k=15 4K 13.37→**4.46ms**（4.09x→**12.3x**）、1080p 3.40→1.16（4.25x→12.5x）；
         k=31 4K 14.06→**4.59ms**（10.6x→**32.9x**）、1080p 3.61→1.19（10.9x→33.1x）。
         r53 全量 **120/120 WIN 零 lose**，min 1.500、p10 2.536、中位 9.001、max 44.102，
         CSV 落 docs/bench/bench_sm61_r53.csv。
      4) **踩坑**：① 模板里 `Op(u,v)` 是构造对象、必须写 `op(u,v)`（`Op` 是类型名，`op` 是实例）；
         ② 内核启动 `<<<g, th, smem>>>` 的 grid 里**不能放函数调用**（`idiv_up()` 触发 C2059，
         必须在 host 侧先算好）；③ **同机残留 bench 进程会污染计时**（第一次 r53 的 CSV 里 hist 项
         跑到 2.7x、mine 列整体偏低，作废重跑）。
      5) 验证：`probe_mcol check2d` **102 用例**（k=9/13/15/21/31/33 × 17 形状 × erode/dilate，
         含 16x1/16x2/17x3/7x400 极扁、4096x64、四档 bench 尺寸）新 vs 旧**逐位 diff=0**；
         `probe_mcol ab` 交错 A/B：4K k=15 VHG 12.76 → line 4.23ms（3.02x 纯内核）；
         smoke **120 [OK]** 全 identical。
      6) **未做**：morph-f32 的大核还是老路径（4K k=31 14.25ms / 11.7x），同一套路可套过去；
         以及 r52 清单里原定的三项（integral 1080p 80→86.5 GB/s、k=7 16px/线程、f32 小图 TH）都没动。
- [x] 下一步（2026-11-10 r53 后 → 2026-11-11 r54 完成）：**morph-f32 大核滑窗寄存器重写**
      （BENCHMARK.md 第二十一轮）。erode-f32 k=31 4K 14.25ms → **5.90ms**（11.7x→28.1x），
      k=15 4K 4.56→3.06（11.7x→17.4x）；r54 全量 **120/120 WIN 零 lose**，min 1.381、
      p10 5.276、中位 20.937，CSV 落 docs/bench/bench_sm61_r54.csv。
      1) **归因（probe_mf32 asym）**：4K (1,1) 2.26 / (31,1) 3.84 / **(1,31) 23.28ms** ⇒
         f32 的病灶与 u8 侧不同源——不是固定结构开销，是列遍自身的 O(k)：`float ringv[255]`
         动态下标落 **local memory** + 每像素一列一写（32 sector 放大）。
      2) **为什么不能照搬 u8 的 smem 整线走查**：两张 float 线表 128 列 × 96 行 ≈ 98KB
         超 48KB 上限；缩尺寸会把占用率压到 3~5%。改**滑窗寄存器**：窗口 K 值常驻寄存器、
         滑动=移位+补新值，O(1)/像素、零 local、零屏障。rows 一线程一行（RB=128，协作载入 +
         smem 输出暂存 SOUTP=TB+1 错 bank），cols 一线程一列、TB=16 行一段（输出攒寄存器、
         段尾合并写回）。覆盖奇数 k∈[9,33]，其余核回落旧路径。
      3) 验证：probe_mf32 **check 68 用例**逐位一致（17 形状 × k=15/31 × erode/dilate，含
         7x400/4096x64 极扁）；`ab` 纯内核 4K k=31 22.96→8.93（2.57x）、720p k=31 3.04x；
         smoke 120 [OK]（f32 形态学 vs OpenCV 逐位一致）。
      4) **踩坑**：① probe 的 g_tmp 按 3840 开、check 却有 4096x64 用例 → 越界假 FAIL，
         暂存宽度按最大用例开；② CK 宏在 void 函数里 `return 1` 编译错，分开 CK/CKV；
         ③ bench 口径 640 k=15 从 8.8x→30.2x 的跳跃里有一半是小图旧路径本就快、列遍
         sector 放大在小图上占比更高。
- [x] 下一步（2026-11-11 r54 后 → 2026-11-12 r55 完成）①：**gaussian k=7 融合内核 TH 形状修正**
      （BENCHMARK.md 第二十二轮）—— r52 给 integral 定的「每 SM ~4 常驻 block → nChunks≈24」
      甜点规则，k=7 的 TH 分档从来没扫过（84 regs → 3 块/SM 常驻，老规则 4K TH=64 → nChunks=34）。
      - 新探针 `probe_g7th`（TH∈{8,16,32,48,64,96,128} × 6 轮轮转起点交错 min + 逐 TH 正确性断言），
        两轮排序一致：640 TH16、720p **TH32**（-16%）、1080p **TH96**（-9%）、4K TH128（-3%）。
        4K 最优 TH128 与 r52 机制吻合（3 块/SM × 6 SM = 18 常驻 → TH≈ceil(2160/18)→128）。
      - 定案：TH 分档改 `(H>=1600)?128:(H>800?96:(H>600?32:16))`，内核本体零改动。
        bench 三轮（新二进制）120/120 WIN 零 lose；k=7 同轮配对 720p 0.112→0.096、
        1080p 0.203→0.182、4K 0.652→0.645、640 打平偏好。
      - **数据勘误**：r54 CSV 的 OpenCV gaussian 列被当时低时钟污染（720p k=7 记 1.47ms，
        同机实测 0.33ms），r54「中位 20.9」系对手侧膨胀，r55 起聚合以本轮为准（min 1.27~1.30、
        中位 9.1~9.3、p10 2.53~2.59）。
      - **流程坑重演（r51 #3）**：改 src 后只重建了 obj+probe，忘了 build_all——前两轮 bench
        跑的旧二进制。规则：**动 src/ 后必须 build_all.bat 再跑 smoke/bench**。
      - 探针踩坑：cudaMallocPitch + 紧排 cudaMemcpy 布局错位 → 第一次全 FAIL 假阳性；
        改 cudaMemcpy2D 后 sanity 全 0。虚惊反证「交叉 TH 比对」能抓 probe_s7 chunk 模式
        的盲区（同 TH 两边同错相消），probe_g7th 保留为固定验证面。
      - 验证：probe_g7th 7 档 × 4 尺寸 oop 逐位一致；probe_s7 check（默认+FORCE_STASH）+
        chunk（CK_TH7=96/128）全 OK（新 TH 下 stash 槽位映射复验）；smoke 120 [OK]；
        bench 三轮 CSV 落 docs/bench/bench_sm61_r55.csv（+ r55b 日志）。
- [x] 下一步（2026-11-12 r55 后 → 2026-11-13 r56 完成）①：**gaussian k=7 融合内核垂直寄存器打包**
      （BENCHMARK.md 第二十三轮）—— r52 遗留候选①的原始形态（16px/线程 + uint4）被探针实测
      否掉，换成 8px 不变、垂直行和对打包 + dp2a 的路线。
      - 新探针 `probe_g7px`（`docs/probes/probe_g7px.cu`）：V1 = 8px + 行和对打包
        (h0,h1)(h2,h3)(h4,h5)+h6，垂直和 3 条 `__dp2a_lo` + 1 IMAD；V2 = 16px/线程 + uint4
        （r52 原方案）。两轮 6 轮轮转交错 A/B：**V1 4K 1.19x（0.445→0.374ms）、640 1.17x、
        720p 1.04x、1080p 0.96x**；V2 全尺寸 0.91~0.97x 被否——16px 的寄存器爆炸
        （h[7][16]=112 regs → ~150 → 1 块/SM）压过了 uint4 的访存收益。
      - **机制**：行和 ≤16320 装 u16，相邻行对打包成 3 个 u32 + h6 单份 → 垂直寄存器
        56→32，84 regs → ~60 → 4 块/SM 常驻（原 3 块）。垂直和 u32 精确，与旧 8 条
        IMAD 版逐位一致。
      - **V2 的 3 个探针踩坑**（调试耗时很长，记录防重蹈）：① `__funnelshift_r(lo,hi,s)`
        参数序是 (低位,高位)，首版当 (hi,lo) 写导致 CPU 仿真与 GPU 全错位；② 窗循环三元链
        `(a==1)?W3:(a==2)?W4:W5` 漏 a==0 分支（a=0 落到 W5，rw[0..3] 全错）——「穷举反推窗字节」
        定位；③ shfl 放进 if/else 分支是 UB（边界 lane 不参与）→ 必须**无条件执行 + 分支覆盖**
        （旧内核的同构写法是对的）。V2 仍有 tid≥1 窗循环异常（输出错、dump 显示 lw/rw 恒定），
        未定案前不集成，探针注释保留现场。
      - **集成**：`gauss7_fuse_kernel` 垂直存储 h[7][8] → P01/P23/P45/h6，滚动改 2 条
        funnelshift + 1 打包/像素。smoke 120 [OK]；probe_g7th 7 档 TH sanity 全 0；
        probe_s7 check+chunk 全 OK（stash 路径同内核模板兼容）。
      - 结果（probe_g7px V0=新 src 口径）：4K 0.445→**0.375**、640 0.035→0.029、
        720p 打平、1080p 0.96x；bench 两轮 CSV 落 docs/bench/bench_sm61_r56*.csv。
- [ ] 下一步（r56 后候选，按性价比排序）：① V2 的 tid≥1 窗循环异常见 probe_g7px 注释，
      若定案可量化 16px 路线（当前证据 0.91~0.97x，即使修好也难赢）；② integral 1080p
      80→86.5 GB/s（收益 ≤8%，地板附近）③ f32 小图 TH 微调（收益小）。
- 同步状态（2026-11-10 r53 现场核实）：本轮 3 个 commit —— `d5d8271`（morph 走查内核 + probe_mcol
  归档）+ `b95c8fc`（BENCHMARK 第二十轮 + PROGRESS + r53 CSV + 原始日志）+ 本条状态行补丁。
  每次 `bash scripts/push.sh` 都亲眼看到 `SYNC_OK <sha>`（依次 b95c8fc → 状态行后的 sha），
  并用 `git ls-remote --heads origin main` 与 `git rev-parse HEAD` 核对一致、`git status` 干净
  （工作区只剩 .gitignore 的 build/ 产物）。本地分支仍只有 `main` + 两个已合入 main 的
  `spike-hist4k` / `spike-clock-lock`（非遗留问题）。**r52 的 `ca7a9a4`/`ddfb959` 已不再是最新，
  勿再引用。**（以上均为本轮现场核实，非照抄。）
- 上一轮同步状态（2026-11-09 r52，历史）：`c79584e` + `e6395a9` + `ddfb959`，SYNC_OK ddfb959。
- 注意：git 仓库不能破坏，只许 add/commit，禁止 reset/checkout 丢工作；每完成一阶段 commit 一次
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
- 巡检 r29（2026-10-20，1050 Ti，新 reps 口径）：smoke 全绿（0 flagged）；bench 118 WIN / 2 lose
  （hist-binary 640 0.94x、hist-gray 1080p 0.92x，仍为 hist 家族临界摆动，非确定性 lose）。CSV 落库
  docs/bench/bench_sm61_r29.csv。
- 巡检 r27/r28（2026-10-19，1050 Ti）：smoke 全绿（0 flagged）；bench 两轮均 120/120 全 WIN
  （十四轮以来首次单轮零 lose），CSV 落库 docs/bench/bench_sm61_r27/r28.csv。hist 家族临界摆动
  本轮未复现，进一步支持「计时噪声、非确定性 lose」判定。
- 巡检 r30（2026-10-22，1050 Ti，reps≤250 新口径）：smoke 全绿（0 flagged）；bench 119 WIN / 1 lose
  （hist-binary 1080p 0.96x，hist 家族临界摆动，非确定性 lose）。CSV 落库 docs/bench/bench_sm61_r30.csv。
- [x] spike-clock-lock 完成并合并（2026-10-19，--no-ff）：本卡不支持锁频（nvidia-smi -lgc/-lmc
  与 nvcc -lgc/-ac 均被拒），改为 bench 计时重复次数上限 50→250。三轮验证 120 项中位 speedup
  全部 ≥1.0（旧口径 hist 家族 lose 2~5/12），单轮偶发 lose 仅 3 项各 1/3（min 0.91~0.95），
  median CV 1.1%。结论：增 reps 有效收敛中位数；hist 家族本机中位口径全 WIN，单轮逐 WIN 仍受
  ~0.1ms 内核噪声限制。合并后本机 r27/r28 巡检（新口径）两轮均 120/120 全 WIN。
- 十二轮聚合确认（r15~r26，2026-10-18）：有 lose 的项共 10 个，全部为 hist 家族（8 项，各 lose 2~5/12、
  中位全部 >1.0）加 mean-f32 640 k=3（2/12，中位 1.25）与 gaussian-f32 640 k=5（1/12，中位 1.25）两个
  临界噪声项。无任何项构成确定性 lose，维持「不开 hist spike」判定。解析坑：bench CSV 的 speedup 是第 7 列
  （索引 6），误用 opencv_ms 列会得到 40 项假 lose。
- 巡检 r31（2026-10-23，1050 Ti，reps≤250 新口径）：smoke 全绿（0 flagged）；bench 120/120 全 WIN
  （零 lose）。CSV 落库 docs/bench/bench_sm61_r31.csv。r30 曾因网络未 push 的 commit 已于本轮补 push。
- 巡检 r32（2026-10-24，1050 Ti，reps≤250 新口径）：smoke 全绿（0 flagged）；bench 118 WIN / 2 lose
  （hist-gray 640 0.88x、hist-gray 4K 0.94x，hist 家族临界摆动，非确定性 lose）。CSV 落库
  docs/bench/bench_sm61_r32.csv。
- 巡检 r33（2026-10-25，1050 Ti，reps≤250 新口径）：smoke 全绿（0 flagged）；bench 119 WIN / 1 lose
  （hist-gray 4K 0.96x，hist 家族临界摆动，非确定性 lose）。CSV 落库 docs/bench/bench_sm61_r33.csv
  （commit 65368ed）。
- 巡检 r34（2026-10-26，1050 Ti，reps≤250 新口径）：smoke 全绿（0 flagged）；bench 120/120 全 WIN
  （零 lose，min 1.038 hist-binary 4K，中位 4.26x）。CSV 被同日 r35 覆盖，本轮权威数据见 r35 行。
- 优化（2026-10-26，r35）：hist-binary 二值专用 popc 聚合路径，4K 6.0x、全尺寸 5.1~7.3x（原 1.04~1.13x）。
  - 病灶定位（probe_binh3，4K 只算内核）：二值数据每 warp 32 lane 全落 bin0/bin255，smem 原子 32 路
    串行 → 0.5919ms，而同字节量的灰度数据只要 0.1413ms；纯读天花板 0.0922ms。
  - 走错的第一版：`__vcmpeq4` 计数反而更慢（0.6656ms）—— ptxas 把 vcmp4 展开成 ~12 条
    PRMT/LOP/IADD32I/POPC，比原子还贵。**sm_61 上别用 vcmp 系列做按字节判定**。
  - 定案：`popc(w)>>3` 直接得 255 字节数（二值数据每 0xFF 占 8 位），0 的个数 = 4 - 该值；
    合法性用 `m=(w>>7)&0x01010101`、要求 `m*0x7F == (w & 0x7F7F7F7F)`（4 条 ALU，无 PRMT），
    块内发现非二值字节则整块回退通用 smem 原子路径（任意数据都正确）→ 0.0993ms，已达读带宽天花板。
  - 归约改为 partials+bin3 + last-block 完全覆写（grid 48），顺带免掉调用内 cudaMemsetAsync
    （实测该 1KB memset 固定 17~22us，小图上占原耗时 40%+）。
  - 验证：probe_binh1 31 用例全过（四档尺寸 × 连续/带 pitch 步长/填充步长、非二值回退、全 0/全 255、
    图内单个杂散字节、连续两次调用不清零）；smoke 全绿（hist binary max diff = 0）。
  - 巡检 r35 同日：bench 120/120 全 WIN（零 lose，min 1.068 hist-gray 1080p，中位 4.34x），
    CSV 落库 docs/bench/bench_sm61_r35.csv。
- 优化（2026-10-26，r36）：hist-gray 三路统一 last-block 归约、去掉调用内 memset，全尺寸 1.5~3.8x（原 1.07~1.16x）。
  - probe_hgray 四档实测（含各自必需序列，ms）：原 global-atomic+memset 0.0573/0.0645/0.0788/0.1582 →
    last-block grid48+uint4 0.0174/0.0261/0.0420/0.1044（640/720/1080p/4K）。
  - 关键认识：1KB cudaMemsetAsync 固定 10~42us（Windows/WDDM 每次 launch ~10us），小图上占原耗时 40%+，
    比内核本体还贵；last-block 只多一次 threadfence+atomicInc，净赚。
  - 取数分档：16B 对齐走 uint4（比 4B 快 3~8%），仅 4B 对齐走 4B 版，都不满足走任意宽度通用核。
  - 验证：probe_hgray1 30 用例全过（9 尺寸 × 连续/pitch/填充步长 704 与 700、非对齐基址、全同值单 bin、
    脏输出连续两次调用）；smoke 全绿（0 flagged）。
  - 巡检 r36：120/120 全 WIN，零 lose，min 1.116（dilate k=3 bin 1080p），中位 4.30x；
    CSV 落库 docs/bench/bench_sm61_r36.csv，BENCHMARK.md 增「第九轮（hist 家族两条路径重写）」。
  - 复验 r37（同日第二轮回测）：120/120 全 WIN，零 lose，min 1.103，中位 4.27x；hist 家族
    binary 3.95~7.35x、gray 1.42~3.24x，两轮均无临界摆动 → 原「hist 单轮偶发 lose」问题消失。
    CSV 落库 docs/bench/bench_sm61_r37.csv。
- 优化（2026-10-27，r38~r41）：morph k=3 单遍融合内核，全尺寸 1.1~1.5x → 2.86~3.78x（BENCHMARK.md 第十轮）。
  - 病灶：k≤7 的 rows→cols 两遍路径行遍按 (W/256)×H 启动，4K 是 32400 个小 block（每块 1 行 256B），
    加 4 次全图流量，4K 1.46ms 只有 11 GB/s 有效带宽（纯 uint4 拷贝天花板 0.178ms / 93 GB/s）。
  - 定案：单遍融合 —— 16 像素/线程、uint4 取数写回、水平 3 邻域用 `__vminu4/__vmaxu4` + `__funnelshift`
    在寄存器里做（跨 warp 只有 lane0/lane31 各补一次 4B 读），垂直 3 行滚动寄存器；TH 按高度 8/16/64。
    纯内核 4K 0.208ms（80 GB/s ≈ 拷贝天花板 86%），1080p 0.067、720p 0.039、640 0.024。
  - 原地（dst==src，benchmark 口径）：先试「内核内 halo 行 + warp 边界字预存」→ 正确但寄存器 31→52，
    4K 反而 0.19→0.38ms；最终改「融合写 workspace（16 对齐紧排）+ 整图 uint4 拷回」，4K 0.374ms。
  - 教训：**逐字节尾块处理会让 ptxas 从 31 涨到 148 regs**（占用腰斩、4K 0.19→0.47ms）；
    融合路径因此只接受 W%16==0 + 16B 对齐，其它宽度回退两遍老路径（正确性不变）。
  - 验证：probe_m3v 56 用例全过（14 组尺寸 × pitch/紧排偏移 × in-place/out-of-place × erode/dilate，
    参照 OpenCV CPU；含 W=16/17/31/33/63/127/1279 与 H=1/2/3 极扁图），smoke 120 [OK] 全绿。
  - 巡检 r41：120/120 WIN，零 lose，min 1.131（mean k=3 4K），中位 4.50x；CSV 落库
    docs/bench/bench_sm61_r41.csv（r38/r39/r40 为同日过程口径，均在 docs/bench/）。
- 失败实验（2026-10-27 凌晨 run 断流，改动未提交，已 stash@{0}）：mean k=3 改 256x16 tile +
  每线程固定一列滚 16 行（tile 行跨 258B）→ smoke 全绿但性能回归 0.21~0.40x（4K 1.667ms
  vs OpenCV 0.357ms，原实现 1.13~1.17x），不要再走这条方向；该 run 同时已落地 morph k=3
  融合内核（0d305d0）。
- 优化（2026-10-28，r42）：mean k=3 单遍融合（u16 lane 打包），全尺寸 1.13~1.30x → 2.60~3.25x（BENCHMARK.md 第十一轮）。
  - 病灶：旧 mean3_direct_kernel 是 32x8 smem tile（4K 32400 个小 block、每像素 9 次 smem 字节读），
    4K 1.10ms = 7.5 GB/s；且 tile halo 从 global 读 → 原地（benchmark 口径）跨 block 竞态，
    probe_m3a 实测 4K 69 万像素错 maxdiff 48（smoke 只测 out-of-place，所以一直没抓到）。
  - 定案：16 px/线程 + uint4；`__byte_perm(w,0,0x4140/0x4342)` 把 4 字节拆成 2 个 u16 lane 字，
    水平 3 邻域用 `__funnelshift_r(e[k-1],e[k],16)` + `__vadd2`；垂直 3 行滚动 hp/hc/hn 各 8 字；
    除 9 用 `(v*7282+32768)>>16`（v≤2295 与 round(v/9) 全量等价）；44 regs、无 spill。
    原地 = 融合写 workspace（16 对齐紧排）+ 整图 uint4 拷回，4B/px。
  - **求和边界必须 REFLECT_101 镜像**：morph 那套 clamp 边界在 min/max 下等价（min(a,a,b)==min(a,b)），
    对求和不等价 —— 首版 clamp 导致 4K 11307 px 错 maxdiff 66；改成左右取像素 1/W-2、上下取 row 1/H-2 后 maxdiff=0。
  - 约束：W%16==0 且 W≤4096 且基址/pitch 16B 对齐走融合；其余（含所有奇数宽）回退两遍路径，该路径原地安全。
  - 验证：probe_m3a（四档 × in/out-of-place）全 0 diff；probe_m3b 80 用例（16x1/16x2/32x1/48x5/63x64/
    100x100/127x129/1279x720 等 × pitch/tight+4 × in-place/out-of-place）全过 + DIV9_CHECK 全量 OK；
    smoke 120 [OK] 全绿（新增 mean k=3/15/31 in-place 三条用例）。
  - 巡检 r42：120/120 WIN 零 lose，min 1.235（mean-f32 640 k=3），中位 4.49x，p10 1.41x；
    CSV 落库 docs/bench/bench_sm61_r42.csv。
  - 顺带体检其它算子的原地口径（probe_ipall：gaussian k=5/7/15、mean k=15/31、erode k=15/31 ×
    640/1080p/4K，共 21 组）：in-place 与 out-of-place **逐位一致（diff=0）**，说明只有旧 mean k=3
    那条路径有竞态，其余原地路径（靠 static workspace 分离两遍）都是安全的。
- 优化（2026-10-29，r43）：gaussian k=5 单遍融合（u16 lane 打包），全尺寸 1.29~1.41x → 2.52~3.35x（BENCHMARK.md 第十二轮）。
  - 病灶：两遍定点路径（rows u8→u16 写 2B/px + cols u16→u8 读 2B/px）4K 全图流量 6B/px，
    1.3926ms 只有 36 GB/s（纯拷贝天花板 93 GB/s / 0.178ms）。
  - 定案：16 px/线程 uint4；`__byte_perm(w,0,0x4140/0x4342)` 拆 u16 lane（每 lane 一个字节）；
    水平 5-tap = `A + C + 4*(L1+R1) + 6*E`（±2 邻取相邻 lane 字、±1 邻 `__funnelshift_r`，全 `__vadd2`，
    每 lane ≤4080 不溢出）；垂直 5 行滚动 `h[5][8]`；输出 `(S+127+((S>>8)&1))>>8`
    与 `__float2int_rn(S/256)` 全量等价（round-half-even，S≤65280），与两遍路径逐位一致。
  - 三个坑（都靠 probe 抓到）：① 跨 warp halo **不能用 `*(uint16_t*)` 读**——那会把两字节压进同一 lane
    的 0..15 位，warp 边界每行错 4 像素（x=510..513 条带）；按字节拼 lane 才对。
    ② 垂直镜像必须周期式（`per=2H-2`, `i%=per`）：H=2 时 y=3 要折两次回 1，单次折叠 + clamp 会错。
    ③ 左边界镜像的两个 lane 是 (px2, px1)（来自 e[1].lane0 / e[0].lane1），不是 (px1, px0)。
  - 原地：融合写 workspace + 整图 uint4 拷回（4.1B/px）；4K in-place 0.5437 vs out-of-place 0.3779，
    差值 0.166ms ≈ copy16 0.184ms，即原地多出来的就是那一遍拷贝。
  - 验证：probe_g5d 常量/斜坡/冲激 maxdiff=0；probe_g5f 69 用例（23 形状 × in/out/紧排）全 0；
    smoke 120 [OK]；r43 全量 120/120 WIN 零 lose，min 1.253，中位 4.49x，CSV 落库 docs/bench/bench_sm61_r43.csv。
- 优化（2026-11-02，r45）：f32 单遍融合（mean k=3 / gaussian k=5）+ 权重上传缓存（BENCHMARK.md 第十四轮）。
  - 病灶：两遍 f32 分离路径流量 16B/px（读 4 + 写 tmp 4 + 读 tmp 4 + 写 4）。4K gaussian-f32 k=5
    1.5688ms 折算 84 GB/s，已贴住纯 uint4 拷贝天花板 93 GB/s —— 指令层面没有空间，只能砍流量；
    mean-f32 k=3 1.9528ms 只有 68 GB/s（列遍是 4B 跨行读）。
  - 定案 `gauss32_fuse_kernel<K,TH>` + `mean32_fuse3_kernel<TH>`：4 px/线程（float4 取、float4 写），
    寄存器滚动 K 行「水平部分和」；水平 halo 用 `__shfl_up/down` 从邻线程 float4 分量取（K=5 用 v.z/v.w
    与 v.x/v.y），只有 warp 边界 lane 0/31 与图像边界才落 global 或走镜像。垂直镜像用与旧路径
    **完全相同**的 fold-once + clamp（不是周期式！—— 旧 f32 路径就是单次折叠，改成周期式会让
    f32 高斯的输出不再与旧路径逐位一致）。
  - 算术次序完全沿用旧路径：行遍 `s=0; for q: s=fmaf(c_w[q], p[i+q], s)`；mean 列遍保持
    `acc = 0+h0+h1+h2` 后 `acc += 新行 - 旧行`（先减后加，不是重算三项）→ 融合输出与两遍路径**逐位一致**。
  - 约束：out-of-place、W%4==0、8<=W<=4096、src/dst 基址与 step 16B 对齐；其余回退两遍路径。
  - 顺带：`cudaMemcpyToSymbolAsync` 上传 20B 权重实测 **11.0us**（probe_cwhdr：noop launch 11.7us、
    copy 11.0us、copy+launch 22.7us 串行叠加），比 640x480 的内核本体还贵 → 加 `upload_w()`
    （同 ksize + memcmp 权重相同则跳过上传），三个高斯入口（u8 k=7 定点、u8 通用、f32）统一走它。
  - 结果：mean-f32 k=3 1.24~1.34x → **3.40~3.64x**；gaussian-f32 k=5 1.26~1.33x → **2.27~3.03x**；
    4K 两个内核都到 91~93 GB/s（= 拷贝天花板，物理上限）。probe 纯内核口径 mean 2.6~3.0x、gauss 1.8~2.1x。
  - 验证：probe_f32f **116 条断言**（19 组形状 × {融合 out-of-place, 旧两遍 in-place} + 步长填充 4 组
    + sigma 切换 6 条）：融合 vs 旧路径 diff **全 0**（含 H=1/2/3、W=12/100/127 回退、pad=8/24 非 16B step）；
    vs OpenCV CPU gauss k=5 diff=0、mean k=3 diff=1.5e-5；sigma 交替（0/1.2/0.7001）证明权重缓存能失效；
    smoke 120 [OK] 全绿。
  - 已知既有角落（非本轮引入）：H <= kernel 半径时旧 f32 路径的 fold-once+clamp 与 REFLECT_101 不等价
    （8x2/16x2 的 k=5 vs OpenCV 差 4~6，k=15 老路径在 8x3/12x5/20x7/256x3 差 0.26~11.7）；融合路径与
    旧输出逐位一致（保留既有行为），benchmark 尺寸不受影响。
  - 巡检 r45：120/120 WIN 零 lose，min 1.394（hist-gray 4K 噪声项），p10 2.578，中位 4.266，max 21.9x；
      CSV 落库 docs/bench/bench_sm61_r45.csv。新增 scripts/run_probe.bat（探针跑法统一）。
    - 顺带修 `scripts/push.sh`：cron（无 tty）下 Git Credential Manager 会去开 /dev/tty 提示凭据 →
      push 根本没发出去，而旧 push_once 的退出码来自 `| tail -3`（恒 0）→ 打出假 push_ok=1。
      改为 GIT_TERMINAL_PROMPT=0 + GIT_CONFIG_* 注入 `gh auth git-credential`（先清空 GCM）+ 退出码取自 git。
- 优化（2026-11-03，r46）：u8 高斯**原地路径去整图拷回** —— 只暂存 halo 行（4K 0.544→0.416ms）。
  - 病灶：原地（benchmark 口径 src==dst）一直是「融合写整图 workspace + 整图 uint4 拷回」= 4.1B/px，
    4K 白扔 0.184ms。r45 计划里的 smem tile 只解决「块内」，跨 block 的 halo 行照样被邻居写掉，
    所以改成 **stash**：前置小内核只把「可能被别人覆盖、但本块还要读」的行拷进暂存区 ——
    每条 chunk 边界一个 2R 行窗口 [c-R, c+R)，末块再加 R+2 行底段（底部镜像折回 + 内核预取 y+R+1）。
    4K k=5 = 138 行 = 530KB（整图 8.3MB），拷贝开销 0.184ms → ~0.012ms。
  - **关键坑（probe_stash1 抓到，每行 1~4 像素错、跨全部 block）**：stash 只解决跨 block，
    块内 **warp 会互相超车**：快 warp 在迭代 y+3 已经写掉行 y+3，慢 warp 在迭代 y 预取的正是行 y+3。
    修法 = 原地路径每行一次 `__syncthreads()`（skew 压到 0），另在预载后补一次（预载也读本块自己的行）。
    out-of-place 不加 barrier（STASH 模板参数），性能零影响。
  - 阈值：barrier 的固定代价在小图上反超收益（640x480 0.047→0.060、720p 0.091→0.096），
    所以只有 H>=1000（1080p/4K）走 stash，小图仍走「写 workspace + 整图拷回」。
  - 结果：gauss k=5 1080p 0.170→0.140（3.06→3.48x）、4K 0.544→0.416（2.5→4.47x）；640/720p 不回归。
  - k=7 **未跟着改**：probe_g7f 在 4K 原地偶发 diff（bad 6~50，非确定复现；同一二进制下 probe_stash2
    连跑三次又是 0），R=3 / 8px 每线程 / 15 warp 每块的超车窗口更大；定位清楚前保持原路径
    （0.639ms / 3.89x 仍 WIN），列进「当前下一步」②。
  - 验证：probe_g5f 69 用例（含 4K/H=1/2/3/回退宽度/in-place/tight）连续两次全过、probe_stash1
    4K 原地 vs 原地外 bad=0、probe_g7f 95 条全过、smoke 120 [OK]。
  - 巡检 r46：120/120 WIN 零 lose，min 1.391（hist-gray 4K 噪声项），p10 2.524，中位 4.491，max 21.9；
    CSV 落库 docs/bench/bench_sm61_r46.csv。
- 经验追加（2026-11-03）：**「每线程读永远领先写」不足以保证原地安全** —— warp 之间没有顺序保证，
  块内快 warp 可以把慢 warp 的预取行覆盖掉。原地卷积类内核要么读写完全分离（workspace/快照），
  要么每行同步一次；用 stash 只压掉跨 block 的那部分，块内这部分必须靠 barrier。
- 注意：benchmark 里 morphology/mean/gaussian 都是原地调用（src==dst），OpenCV 都是 out-of-place；
  我们的原地路径天然多一次 temp/拷贝，口径偏保守。morph k=3 的 out-of-place 实测 4K 0.208ms（6.8x）。
- 分支清理状态（2026-10-26 已核实，勿再列为待办）：`spike-arch-scaling` **已删除**——本地 refs 与 GitHub
  远端均不存在（`git ls-remote --heads origin` 只有 main），其内容早已包含在 spike-hist4k→main 链路中。
  本地仍存两个**已合入 main** 的分支 `spike-hist4k`、`spike-clock-lock`（--no-ff 合并，保留未删，非遗留问题）。
- 经验：cmd /c 长命令在 git-bash 下要用 MSYS_NO_PATHCONV=1 cmd /c；.cu 文件写中文注释必须带 BOM；
  同一 .cu 内多个 extern __shared__ 不能重名异型（float ring[] 与 uint16_t ring[] 冲突，改名即可）；
  直方图性能排查先看"计时内预清零"这类隐形固定开销（~20us 比内核本体还贵），再看 smem 原子冲突；
  单轮 lose 判定不可靠——hist/f32 临界项需 ≥10 轮聚合，lose 若不定点复现即为计时噪声而非确定性 lose
