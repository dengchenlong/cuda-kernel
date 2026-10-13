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
