"""只用 GPU kernel 时间画图：推理耗时、加速比、训练耗时与显存。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(__file__).parent / "results"
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root
runs = {r["num_chunks"]: r for r in (json.loads(p.read_text()) for p in root.glob("k*.json"))}
ks = sorted(runs)
colors = {"compressed": "tab:blue", "raw": "tab:red"}
labels = {"compressed": "compressed (K x 8 tokens)", "raw": "raw ViT tokens (K x 49)"}


def gpu(section, mode, batch, k):
    item = runs[k][section].get(f"{mode}_b{batch}")
    if not item or item.get("oom"):
        return None
    return item["gpu_kernel_ms"]["total"]


fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
for ax, batch in zip(axes[:3], (1, 8, 32)):
    for mode in ("compressed", "raw"):
        ax.plot(ks, [gpu("inference", mode, batch, k) for k in ks], "-o", color=colors[mode], label=labels[mode])
    ax.set_title(f"Inference GPU time, batch={batch}")
    ax.set_ylabel("ms / sample_actions")
ax = axes[3]
for batch, marker in ((1, "o"), (8, "s"), (32, "^")):
    speedup = [gpu("inference", "raw", batch, k) / gpu("inference", "compressed", batch, k) for k in ks]
    ax.plot(ks, speedup, "-" + marker, label=f"batch={batch}")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_title("Speedup from compression (raw / compressed)")
ax.set_ylabel("x")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("bp_num_chunks K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "bp_speed_gpu_inference.png", dpi=130)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for batch, marker in ((8, "o"), (16, "s"), (32, "^")):
    for mode in ("compressed", "raw"):
        style = "-" if mode == "compressed" else "--"
        pts = [(k, gpu("training", mode, batch, k)) for k in ks]
        pts = [p for p in pts if p[1] is not None]
        mem = [(k, runs[k]["training"][f"{mode}_b{batch}"]) for k in ks]
        mem = [(k, v["peak_mem_gb"]) for k, v in mem if not v.get("oom")]
        if pts:
            axes[0].plot(*zip(*pts), style + marker, color=colors[mode], label=f"{mode} B={batch}")
        if mem:
            axes[2].plot(*zip(*mem), style + marker, color=colors[mode], label=f"{mode} B={batch}")
    pts = [(k, gpu("training", "raw", batch, k), gpu("training", "compressed", batch, k)) for k in ks]
    pts = [(k, r / c) for k, r, c in pts if r and c]
    axes[1].plot(*zip(*pts), "-" + marker, label=f"batch={batch}")
axes[1].axhline(1.0, color="k", lw=0.8, ls=":")
axes[2].axhline(95.6, color="k", lw=0.8, ls=":")
axes[2].text(4.2, 91, "96GB (missing points = OOM)", fontsize=8)
for ax, title, ylabel in ((axes[0], "Training fwd+bwd GPU time", "ms / step"),
                          (axes[1], "Training speedup from compression (raw / compressed)", "x"),
                          (axes[2], "Training peak memory", "GB")):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("bp_num_chunks K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2 if ax is not axes[1] else 1)
fig.tight_layout()
fig.savefig(out_dir / "bp_speed_gpu_training.png", dpi=130)
print("ok")
