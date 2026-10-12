// README + verdict: spike 001 integral 归因
//
# spike 001: integral 在 sm_120 上的性能归因

## 问题
现版 integral（96-warp 行扫描 + chunk 列 carry）在 RTX 5090 D (sm_120) 上
0.51x(1080p) / 0.43x(4K) 输给 OpenCV-CUDA 4.14。假设：Pass1 的 launch shape
（固定 12 block × 256 线程 = 96 warp，只占 170 SM 的一小部分）没有吃满 GPU。

## 实验（probe1.cu）
三个实现 + OpenCV，全部 cudaEvents 中位数：

| 实现 | 1280x720 | 1920x1080 | 3840x2160 |
|---|---|---|---|
| v0 = 现版 (12 blk × 256) | 0.108 | 0.216 | 0.531 |
| v1 = rows 128 blk × 256 | 0.098 | 0.191 | 0.423 |
| v1 = rows 256 blk × 256 | 0.082 | 0.158 | 0.279 |
| **v2 = rows 每 block 8 行** | **0.072** | **0.131** | **0.191** |
| OpenCV | 0.294 | 0.339 | 0.414 |

## 结果
- **假设成立**：Pass1 launch shape 是主要瓶颈。96 warp → 2048 warp（v2）,
  4K 从 0.53 → 0.19 ms（2.8x）。
- **v2 已全面跑赢 OpenCV**：720p 0.072 vs 0.294 (4.1x)、1080p 0.131 vs 0.339 (2.6x)、
  4K 0.191 vs 0.414 (2.2x)。此前 benchmark 里 OpenCV 0.230ms@4K 的口径含 GpuMat
  输出分配；本 probe 里 OpenCV 计时含输出分配复用，仍被 v2 碾压。
- probe1 报的 "bad rows=5" 是**对照 bug**：OpenCV cuda::integral 输出是
  (W+1)x(H+1)（probe5 证实），而本库输出 W×H —— probe1 用同索引比较错位。
  v2 全链路 vs CPU 双重前缀 **2160 行全对**（probe4），逐位正确。
- v2 的 Pass2 与现版完全相同（未动）。

## v2 的改动本质
Pass1 从“12 block × 256 线程、grid-stride 行”改为“每 block 8 行（每 warp 一行）、
grid=(H+7)/8”。内核代码逐指令相同，只是调度 shape 变了 —— **算法/数值路径零改动**，
sm_61 上同样受益或持平（warp 数总是 ≥96 的超集，不引入新指令序列）。

## Verdict: VALIDATED

### 建议（进真实实现）
1. `integral_rows_kernel` launch 改为 `<<<(H+7)/8, 256>>>` + kernel 内去 grid-stride
   （保留 grid-stride 作超大 H 的护栏？不需要，grid 维度可到 2^31-1）。
2. 跑 smoke 确认逐位一致（应零差异，数值路径没动）。
3. 笔记本 sm_61 验证：shape 改化对 Pascal 同样是“更多并行”方向，预期只快不慢；
   若担心，可加 `if (H >= 720)` 门 —— 但更倾向无条件替换（96-warp 版在 4K 明显亏）。
