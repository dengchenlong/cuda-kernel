# spike 002: gaussian 归因与 v4 内核

## 问题
u8 gaussian k=15 全尺寸输 OpenCV（0.72-0.75x），gaussian-f32 k≥5 输（0.66-0.84x）。
现版 rows(256 线程 tile+1 输出) + cols(ring 滚动 1 列)。

## 过程（probe1→6）
1. probe1 首跑污染数据作废；probe2 交错公平计时：v3(rows2+cols2, f32 版已有内核移植)
   全尺寸 ≥ v0，4K k=15 0.088→0.060 (-32%)。但仍未过 cv。
2. cv 计时异常排查（probe3/4/5）：主口径 cv k=15@1080p=0.030 稳定复现；
   前序 op（mean k=31）污染状态后 cv 呈双峰 0.03/0.28 —— **OpenCV 4.14 cudafilters
   gaussian 大核路径对 GPU 状态敏感**，主 benchmark 口径下 0.030 是它最强状态。
3. 读 OpenCV 源码（row_filter.hpp/column_filter.hpp）：**4-patch + halo smem 布局**，
   32x8 block，每线程 4 输出。
4. probe6：v4 = rows4(OpenCV 同构) + cols4(OpenCV 同构)：

| 尺寸 | k | cv(此探针口径) | v3 | **v4** |
|---|---|---|---|---|
| 640x480 | 15 | 0.268* | 0.0189 | **0.0062** |
| 1280x720 | 15 | 0.268* | 0.0284 | **0.0104** |
| 1920x1080 | 5 | 0.285* | 0.0308 | **0.0106** |
| 1920x1080 | 15 | 0.278* | 0.0410 | **0.0163** |
| 3840x2160 | 5 | 0.313* | 0.0371 | **0.0308** |
| 3840x2160 | 15 | 0.344* | 0.0575 | **0.0493** |

*本探针里 cv 数字系统性偏大（状态敏感，probe5 已证主口径 0.030@1080p k=15），
与 v3/cv 主口径对照（probe2：1080p k=15 v3=0.0436 vs cv=0.0279）合并估算：
**v4 ≈ cv 主口径的 0.5~0.6 倍时间**，即 1.8-2x 领先。

## v4 设计要点
- rows：32x8 block，每线程 4 输出，smem tile (4+2)*32 floats/行，全 smem MAC，
  kernel 在 constant（OpenCV 用 global+L2，constant 更省一个 cache 层级）。
- cols：32x8 block，每线程 4 输出行，smem tile (4+2*HALO)*8 行 × 32 列，
  halo 行 1-2 行复用（k≤16 halo=1，否则 2），无 ring 滚动。
- 数值路径与现版完全一致（同 make_gauss_kernel、同 f32 中间缓冲、同
  __float2int_rn 舍入）—— 逐位结果不变，smoke 应零差异。
- f32 版同样可移植（rows2→rows4、cols2→cols4，输入 float）。

## 意外发现
OpenCV 4.14 gaussian k=31 直接崩溃（invalid argument，callers 表边界），
本项目 k=31 正常。benchmark 主对照集 {5,7,15} 不受影响。

## Verdict: VALIDATED

### 进真实实现
1. gaussian.cu：新增 rows4/cols4 内核 + host dispatch（u8 与 f32 各一套），
   替换现版两遍路径（保留 k=5/7 定点特化不动 —— 它们已赢）。
2. 跑 smoke 全量比对（预期逐位一致）。
3. 跑 benchmark 验证 18 lose 项中的 gaussian 10 项转 WIN。
