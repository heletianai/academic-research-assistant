"""
高 dpi 重生成 chart PNG（≥ 150 dpi，文件 ≥ 50KB），加第 4 张图（vLLM batch=16 vs HF）
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RES = Path(__file__).parent / "results"
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 11

GPU_TITLE = "RTX 4090D 24GB"

# Chart 1: Throughput vs Concurrency
data = json.loads((RES / "throughput_curve.json").read_text())
batches = [r["concurrent"] for r in data["results"]]
tps = [r["throughput_tokens_per_s"] for r in data["results"]]
fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar([str(b) for b in batches], tps, color=["#4A90E2", "#5DA5DD", "#7DBAE0", "#2E5C8A"])
for b, v in zip(bars, tps):
    ax.text(b.get_x() + b.get_width()/2, v + max(tps)*0.01, f"{v:.1f}",
            ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.set_xlabel("Concurrent Requests (batch size)", fontsize=13)
ax.set_ylabel("Throughput (tokens/s)", fontsize=13)
ax.set_title(f"vLLM PagedAttention: Throughput vs Concurrency\n(Qwen2.5-7B-AWQ on {GPU_TITLE})", fontsize=14)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(tps) * 1.1)
fig.savefig(RES / "chart_throughput.png")
plt.close(fig)
print(f"Saved chart_throughput.png")

# Chart 2: Latency Distribution
data = json.loads((RES / "single_batch.json").read_text())
fig, ax = plt.subplots(figsize=(9, 6))
metrics = ["min", "P50", "avg", "P95", "P99", "max"]
values = [
    data["min_tokens_per_s"], data["p50_tokens_per_s"], data["avg_tokens_per_s"],
    data["p95_tokens_per_s"], data["p99_tokens_per_s"], data["max_tokens_per_s"]
]
bars = ax.bar(metrics, values, color="#5DA5DD")
for b, v in zip(bars, values):
    ax.text(b.get_x() + b.get_width()/2, v + max(values)*0.005, f"{v:.2f}",
            ha="center", va="bottom", fontsize=12)
ax.set_ylabel("Tokens / second", fontsize=13)
ax.set_xlabel("Statistic", fontsize=13)
ax.set_title(f"vLLM Single-Request Latency Distribution\n(30 queries, Qwen2.5-7B-AWQ on {GPU_TITLE})", fontsize=14)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(min(values) * 0.98, max(values) * 1.005)
fig.savefig(RES / "chart_latency.png")
plt.close(fig)
print(f"Saved chart_latency.png")

# Chart 3: vLLM (single) vs transformers
vllm_data = json.loads((RES / "single_batch.json").read_text())
hf_data = json.loads((RES / "transformers_baseline.json").read_text())
fig, ax = plt.subplots(figsize=(9, 6))
frameworks = ["HuggingFace\ntransformers + AWQ", "vLLM 0.10.2\n+ PagedAttention\n(single batch)"]
speeds = [hf_data["avg_tokens_per_s"], vllm_data["avg_tokens_per_s"]]
colors = ["#E07B5B", "#4A90E2"]
bars = ax.bar(frameworks, speeds, color=colors)
speedup = vllm_data["avg_tokens_per_s"] / hf_data["avg_tokens_per_s"]
for b, v in zip(bars, speeds):
    ax.text(b.get_x() + b.get_width()/2, v + max(speeds)*0.01, f"{v:.2f} tok/s",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Tokens / second", fontsize=13)
ax.set_title(f"vLLM vs transformers: {speedup:.2f}× speedup (single batch)\n(Qwen2.5-7B-AWQ on {GPU_TITLE})", fontsize=14)
ax.grid(axis="y", alpha=0.3)
fig.savefig(RES / "chart_vllm_vs_transformers.png")
plt.close(fig)
print(f"Saved chart_vllm_vs_transformers.png (speedup: {speedup:.2f}x)")

# Chart 4: 新增 - vLLM batch=16 vs HF transformers (最震撼的对比)
batch16_tps = data["results"][-1]["throughput_tokens_per_s"] if isinstance(data, dict) and "results" in data else None
# 重新读 throughput data
tp_data = json.loads((RES / "throughput_curve.json").read_text())
batch16 = next(r for r in tp_data["results"] if r["concurrent"] == 16)
batch16_tps = batch16["throughput_tokens_per_s"]
hf_tps = hf_data["avg_tokens_per_s"]
mega_speedup = batch16_tps / hf_tps

fig, ax = plt.subplots(figsize=(9, 6))
frameworks = ["HuggingFace\ntransformers + AWQ\n(batch=1)", f"vLLM 0.10.2\nPagedAttention\n(batch=16)"]
speeds = [hf_tps, batch16_tps]
colors = ["#E07B5B", "#1B4F8F"]
bars = ax.bar(frameworks, speeds, color=colors)
for b, v in zip(bars, speeds):
    ax.text(b.get_x() + b.get_width()/2, v + max(speeds)*0.01, f"{v:.1f} tok/s",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_ylabel("Tokens / second", fontsize=13)
ax.set_title(f"Production Throughput: vLLM batch=16 vs transformers\n{mega_speedup:.1f}× speedup (Qwen2.5-7B-AWQ on {GPU_TITLE})", fontsize=14)
ax.grid(axis="y", alpha=0.3)
fig.savefig(RES / "chart_vllm_batch16_vs_hf.png")
plt.close(fig)
print(f"Saved chart_vllm_batch16_vs_hf.png (speedup: {mega_speedup:.2f}x)")

print("\nAll charts generated at 150 dpi.")
