# spike 003: histogram 归因

## 问题
hist-binary 3 项小输（0.92-0.96x）、hist-gray 1280x720 平（1.00x）、4K 输（0.72x）。
现版 fast path 固定 `HIST_GRID=48` block —— 这是 Pascal（6 SM）时代调优值，
RTX 5090 D 有 **170 SM**，48 block 只占 28% SM，大图长尾严重。

## 实验（probe1.cu）
grid 大小 × 每线程 uint4 数，四尺寸对照：

| 尺寸 | cv 主口径* | v0=48blk | 170blk | 256blk | 96blk 4u4 |
|---|---|---|---|---|---|
| 640x480 | 0.007 | 0.0040 | 0.0051 | 0.0051 | 0.0040 |
| 1280x720 | 0.007 | 0.0051 | 0.0050 | 0.0051 | 0.0040 |
| 1920x1080 | 0.007 | 0.0062 | 0.0051 | 0.0053 | 0.0058 |
| 3840x2160 | 0.009 | 0.0147 | 0.0078 | **0.0074** | 0.0092 |

*cv 在本探针中系统性偏大（同 spike 002 的状态敏感现象），主口径取主 benchmark 值。

## 结论
- **grid 是唯一主因**：4K 时 48→256 blk 快 2 倍（0.0147→0.0074），已超 cv 主口径 0.009。
- 每线程多 uint4（2u4/4u4）无一致收益。
- 小图（640x480）大 grid 略慢 1us（0.0040→0.0051），但主 benchmark 里小图差的就是
  这 1us（0.92x）——自适应 grid 可两头兼顾。

## Verdict: VALIDATED

### 进真实实现
`gpu_histogram` fast path 的 grid 改为 `min(SM数, max(48, (H+7)/8))`（运行时取 SM 数，
170blk 在 4K 主口径实测最优 0.0092ms，256blk 反而 0.0109ms）。
partial 缓冲已是 256 槽上限，无需改分配。二值版已是 256 blk 不动。
sm_61 上 grid-stride 语义不变，正确性不受影响；48 下限保留 Pascal 原行为。

---

# spike 003b: warp 级优化尝试（warp-ballot / 私有计数）

## 问题
benchmark 中 hist-gray 偶发 lose（1080p 0.79-0.86x、4K 偶发 0.54x）。
假设：warp 级原语（__match_any_sync 聚合 / per-lane 私有计数）可超越现有
per-byte smem atomicAdd 实现。

## 三个候选（probe4.cu，主口径 + cv 快路径对照）

| 方案 | 1080p | 4K | cv 快路径 |
|---|---|---|---|
| v0 = 现版 per-byte smem atomic | **0.0041** | **0.0071** | 0.0068 / 0.0077 |
| v1 = __match_any_sync warp-aggregated | 0.0152 | 0.0497 | ↑ |
| v2 = per-lane 私有 u16 计数 + 归约 | 0.0090 | 0.0191 | ↑ |

## 结果与归因
- **v1 慢 3-7 倍**：随机 8-bit 数据在 32 lane 上重复 bin 期望仅 ~2 组（256 bins），
  match 指令 + leader 分支开销远超聚合收益。warp-aggregated 适合高碰撞场景
  （同一地址的 atomicInc，如队列分配器），不适合 256-bin 直方图。
- **v2 慢 1.3-2.7 倍**：64KB smem/块把 occupancy 压到每 SM 2 块，加上 128×256
  归约成本，抵消了"零原子"收益。
- **v0 已是最优**：1080p 领先 cv 快路径 66%，4K 领先 8%。

## 踩坑记录
- `__match_any_sync(0xffffffff, ...)` 在 warp 部分活跃（行尾 cols4%32≠0）时**死锁**：
  mask 要求全 warp 到达，inactive lane 永远不会到达。必须用 `__activemask()`
  （NVIDIA warp-aggregated atomic 官方写法）。
- cv calcHist 复用 integral 的 (W+1)×(H+1) 输出 GpuMat 时，输出尺寸反复 8MB↔1KB
  抖动会卡死分配器 —— 探针里必须用独立输出。

## Verdict: INVALIDATED（假设被证伪 —— 两个方向都排除，确认现版 v0 内核已是
该架构上的最优形态；probe 口径下 v0 领先 cv 8%~66%）

### 遗留
benchmark 环境效应导致的偶发 lose（±1-2μs）：非内核差距，内核不动。

---

# spike 003c: 4K 两项边缘项攻坚（hist-gray 0.71x / hist-binary 0.90x → 全 WIN）

## 问题
120 项 benchmark 中仅剩 hist-gray 4K（0.71x）与 hist-binary 4K（0.90x）未达标。
上一轮结论"v0 已最优"建立在 probe4 的对照上，本轮重审发现 probe4 的 cv 快路径
对照里有结构性缺陷，归因需要重做。

## 实验

### probe5：网格 × 归约 × 带宽地板（首次系统对照）
- ro16（uint4 只读+校验和）4K 地板 **0.0053ms**；ro4（uint 只读）0.0023ms（L2 命中）。
- 现版 v0@170 last-block 0.0102ms；cv 同构复刻（4B 读 + per-block global atomicAdd +
  计时内预清零）**0.0064ms** —— last-block 归约是 4K 主瓶颈。
- ⚠️ 本轮踩坑：cv 槽位复用了 integral 的 8MB dHist 输出，cv 计时被毒化
  （半数窗口 0.13ms）；且 probe4 的 cvk4@170 无 grid-stride 只扫 1360/2160 行，
  "4B 读赢 uint4"是假信号。probe6 修正两处后复测。

### probe6（口径修正后）：4K gray
| 方案 | med |
|---|---|
| v0_170（last-block, uint4） | 0.0100 |
| ga4_h8（cv 复刻: 4B + global atomic + 预清零） | 0.0064 |
| ga16_h8（uint4 + global atomic） | 0.0063 |
| lb4_h8（4B + last-block） | 0.0121 |
| bino_256（binary popcount） | 0.0097 |
| bcvk_h8（binary cv 复刻路径） | 0.0060 |

结论：读宽 4B/16B 无显著差；**归约机制是唯一主因**。__threadfence × 170 块 +
末块串行汇总（~13μs×170 块的栅栏代价）在 4K 尾部主导；2160=270×8 时 cv 网格
恰好无迭代尾巴。1080p/720p/640x480 全尺寸同样占优（0.0049 vs 0.0088 等），
不存在小尺寸回归（预清零 1KB 在计时内, 与 cv 同口径）。

### probe8（生产 API 验收, 链接 libgpuops.a, 交错 60 轮）
- gray 4K: mine med 0.0088 / min 0.0083；cv med 0.1345（分配器毒化窗）/ min 0.1237。
- binary 4K: mine med 0.0085 / min 0.0068；cv 快路径 min 0.0076。
- correctness max diff = 0。

## Verdict: VALIDATED

### 进真实实现（src/histogram.cu 已改）
- fast path 弃 last-block，改 cv 同构 per-block global atomicAdd；API 内
  memsetAsync 预清零（与 cv hist.setTo(0) 同代价，覆写契约不变）。
- 对齐条件从 16B 放宽到 4B（读宽无差，覆盖面更广）。
- binary 走灰度同一路径（bin0/bin255 语义天然成立，其余 bin 加 0）。
- 回退路径（任意宽度 vec last-block）保留，grid=48 固定，sm_61 兼容不变。

### 验收（服务器 3 轮 benchmark）
- hist-gray 4K **0.008 vs cv 0.010 = 1.18x WIN**（原 0.71x）
- hist-binary 4K **0.008 vs cv 0.009 = 1.06x WIN**（原 0.90x）
- 全尺寸 hist 8 项全 WIN（640x480 1.66-29.9x，720p 1.28-1.36x，1080p 1.20-1.32x）
- 118 项其余全 WIN 无回归；smoke all pass，hist gray/binary max diff = 0。

### 待办
- 笔记本 sm_61 复验（代码未用任何 sm_61 缺失指令；回退路径未动）。
- cv 分配器毒化窗口现象（med 0.13 vs min 0.0076）与 benchmark 无关但记录在案：
  cv 侧大 buffer 反复重分配会周期性拖慢 calcHist，探针永远用独立输出。
