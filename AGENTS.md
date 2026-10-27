# AGENTS.md — cuda-kernel

高性能 CUDA 图像算子库（对标 OpenCV-CUDA），Pascal sm_61 起步，支持 CV_8U 与 CV_32F。

## 项目速览

- **构建**：`scripts\build_all.bat`（nvcc 直接编译，每次全量重编 5 个 TU，约 65s；路径变量在脚本顶部）或 CMake（`-DOpenCV_DIR=...` 必须显式指定）
- **正确性**：`scripts\run_smoke.bat`（vs OpenCV-CUDA 逐像素比对，约 3s）；**性能**：`scripts\run_bench.bat`（全量 120 项，数分钟，CSV 落 `build\bench_results.csv`）
- **探针**：`scripts\build_probe.bat <name>` 编译 `build\<name>.cu`，并自动补 UTF-8 BOM（含中文注释的 .cu 缺 BOM 时 nvcc 必报错）
- **推送**：`scripts\push.sh`（直连 GitHub 被重置时自动走本机代理，并用 `ls-remote` 校验远端 SHA）；只有打印 `SYNC_OK` 才算真的推上去了
- **提交**：`core.hooksPath=scripts/hooks` 的 pre-commit 只在 staged 含 `src/`、`bench/` 时重编 + 跑 smoke；`SKIP_SMOKE=1 git commit` 可跳过
- **CI**：GitHub Actions 只做编译级检查（sm_61/75/86/89 四架构矩阵）——runner 无 GPU，smoke/benchmark 只能本机跑
- **数据**：BENCHMARK.md（十轮，120 项全 WIN）、PROGRESS.md（开发记录；续作从「当前下一步」接）

## Agent skills

### Issue tracker

Issues 走 GitHub Issues（`gh` CLI，仓库 dengchenlong/cuda-kernel）。See `docs/agents/issue-tracker.md`.

### Triage labels

默认五角色标签：`needs-triage` / `needs-info` / `ready-for-agent` / `ready-for-human` / `wontfix`。See `docs/agents/triage-labels.md`.

### Domain docs

单上下文布局：根 `GLOSSARY.md` + `docs/adr/`（按需惰性创建）。See `docs/agents/domain.md`.
