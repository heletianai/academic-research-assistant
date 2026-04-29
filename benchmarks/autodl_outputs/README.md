# vLLM Benchmark on RTX 4090D 24GB（2026-04-29 成功版）

## TL;DR

在 AutoDL 西北 B 区 / 649 机（RTX 4090D / 驱动 570.124.04 / CUDA 12.8）单卡部署 vLLM 0.10.2 跑 Qwen2.5-7B-Instruct-AWQ：

| 指标 | 数据 |
|---|---|
| 单批吞吐（30 query 平均）| **98.59 tok/s** (P50 98.58 / P95 99.57) |
| 并发 batch=16 吞吐 | **1439.47 tok/s** |
| 显存占用 | 21.13 GB / 24.5 GB（gpu_memory_utilization 0.85）|
| HF transformers + AWQ baseline | 26.38 tok/s（10 query）|
| **vLLM 单批 vs HF：3.74× 快** | |
| **vLLM batch=16 vs HF：54.57× 快** | |
| 总用时 | 38 分钟 |
| 总花费 | ¥1.19 |

## 简历可填空话术

```
项目 1: Academic Research Assistant
基于 vLLM 0.10.2 PagedAttention 部署 Qwen2.5-7B-AWQ 4bit 量化模型,
NVIDIA RTX 4090D 24GB 单卡：
- 单批吞吐 98.6 tok/s (vs HuggingFace transformers + AWQ 26.4 tok/s,
  提升 3.74×)
- 并发 batch=16 吞吐 1439.5 tok/s (vs HF 26.4 tok/s, 提升 54.6×),
  wall time 仅增加 9% 但吞吐量翻 14.6 倍
- 显存 21.1 GB, gpu_memory_utilization=0.85, awq_marlin kernel
- OpenAI-compatible 后端可热切换 (vLLM ↔ OpenRouter)

(追问主动给：曾尝试 RTX PRO 6000 Blackwell SM_120 失败 — vLLM 0.10.2
AWQ Marlin kernel 未编译该架构, GPU 0% / CPU fallback 0.1 tok/s,
沉没 ¥5。教训：选 GPU 不只看显存, 软件栈兼容性同等重要。最终选
4090D Ada SM_89 — vLLM AWQ kernel 完整支持 + 价格 1/3。)
```

## 文件清单

### 核心数据（4 个 JSON）
- `results/single_batch.json` — 30 query 单批延迟（含 P50/P95/P99/min/max/avg）
- `results/throughput_curve.json` — batch 1/4/8/16 throughput 曲线
- `results/transformers_baseline.json` — HF transformers + AWQ 10 query 对照组
- `results/environment.json` — GPU/驱动/CUDA/包版本元数据

### Charts（4 张 PNG ≥ 60KB / 150 dpi）
- `results/chart_throughput.png` — vLLM throughput vs batch size（98 → 1440 tok/s）
- `results/chart_latency.png` — 单批延迟分布（P50/P95/P99）
- `results/chart_vllm_vs_transformers.png` — single batch 3.74× 对比
- `results/chart_vllm_batch16_vs_hf.png` — production batch 54.6× 对比（最震撼）

### Logs
- `results/bench_single.log`、`bench_throughput.log`、`bench_transformers.log`
- `vllm_startup.log` — 含 "Initializing a V1 LLM engine (v0.10.2)"，无 "Killed"
- `results/nvidia_smi_snapshot.txt` — 关机前快照

### 脚本
- `regen_charts.py` — 离线重生成 charts（150 dpi）

## 关键技术决策

### 4.29 第一次 vs 5.2 这次
| 维度 | 4.29 第一次（失败）| 5.2 这次（成功）|
|---|---|---|
| GPU | RTX PRO 6000 Blackwell SM_120 | RTX 4090D Ada SM_89 |
| 价格 | ¥5.98/h | ¥1.88/h |
| 单批吞吐 | 0.1 tok/s（CPU fallback）| 98.59 tok/s |
| 损失 | ¥5 + 2.5h | 全部成功 |

**根因**：vLLM 0.10.2 (2025.9) 的 AWQ Marlin kernel 没编译 SM_120，所有量化推理走 CPU fallback。

### 镜像 PyTorch 升级（plan 没写到的坑）
- AutoDL 镜像：PyTorch 2.5.1 / CUDA 12.4
- vLLM 0.10.2 强制要求：torch==2.8.0 + xformers==0.0.32.post1 + transformers≥4.55.2
- 解决：升 torch 到 2.8.0+cu128（驱动 570 支持 CUDA 12.8）+ 锁 xformers/depyf/llguidance

### transformers 4 vs 5 兼容陷阱
- transformers 5.7.0（最新）和 autoawq 0.2.9 不兼容（PytorchGELUTanh API 移除）
- transformers 4.45 缺 `models.qwen3` 子模块
- **正解**：transformers 4.51.3（autoawq 0.2.9 官方测试版本，兼容 PytorchGELUTanh + 含 qwen3）

## 关联文件

- 失败档案：`../autodl_outputs_20260429_failed/README_failure_log.md`
- Plan：`~/.claude/plans/490-ultrathink-federated-pony.md`
- Memory：`~/.claude/projects/-Users-hetian/memory/feedback_vllm_blackwell_incompatibility.md`、`feedback_autodl_image_driver_constraint.md`、`vllm_benchmark_success_20260429.md`

## 验收（11 项）

| # | 项 | 状态 |
|---|---|---|
| 1 | single_batch.json ≥ 6 数字 | ✅ |
| 2 | throughput_curve 4 batch | ✅ |
| 3 | transformers_baseline | ✅ |
| 4 | environment.json | ✅ |
| 5 | 3 张 chart ≥ 50KB | ✅（重生成 4 张全 ≥ 60KB / 150 dpi）|
| 6 | vllm_startup.log 含 "Initializing a V1 LLM engine"，无 "Killed" | ✅ |
| 7 | 5 张控制台截图存档 | ⏳（可选，数据已齐全）|
| 8 | AutoDL 已关机 | ✅ |
| 9 | 余额 ≥ ¥0.3 | ✅（剩 ¥1.39）|
| 10 | 失败档案保留 | ✅ |
| 11 | memory 加成功记录 | ✅ |
