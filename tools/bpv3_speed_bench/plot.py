"""生成速度报告配图。"""

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
inter = json.loads((root / "interleave.json").read_text())
ks = sorted(runs)
colors = {"compressed": "tab:blue", "raw": "tab:red"}
labels = {"compressed": "compressed (K x 8 query tokens)", "raw": "raw ViT tokens (K x 49)"}

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, batch in zip(axes, (1, 8, 32)):
    for mode in ("compressed", "raw"):
        wall = [inter["inference"][f"k{k}_{mode}_b{batch}"]["mean"] for k in ks]
        gpu = [runs[k]["inference"][f"{mode}_b{batch}"]["gpu_kernel_ms"]["total"] for k in ks]
        ax.plot(ks, wall, "-o", color=colors[mode], label=f"{labels[mode]} wall")
        ax.plot(ks, gpu, "--s", color=colors[mode], alpha=0.7, label=f"{labels[mode]} GPU kernel")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("bp_num_chunks K")
    ax.set_ylabel("ms / sample_actions call")
    ax.set_title(f"Inference latency, batch={batch}")
    ax.grid(alpha=0.3)
axes[0].legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(out_dir / "bp_speed_inference.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
stages = [("vit", "ViT (cur + BP)"), ("bp_compressor", "BP compressor"), ("vlm_prefill", "VLM prefill"),
          ("denoise_10steps", "denoise x10")]
stage_colors = ["tab:gray", "tab:green", "tab:orange", "tab:purple"]
for ax, batch in zip(axes, (1, 32)):
    x = 0
    ticks, names = [], []
    for k in ks:
        for mode in ("compressed", "raw"):
            g = runs[k]["inference"][f"{mode}_b{batch}"]["gpu_kernel_ms"]
            bottom = 0.0
            for (key, name), c in zip(stages, stage_colors):
                v = g.get(key, 0.0)
                ax.bar(x, v, bottom=bottom, color=c, label=name if x == 0 else None, width=0.8)
                bottom += v
            ticks.append(x)
            names.append(f"K{k}\n{'C' if mode == 'compressed' else 'R'}")
            x += 1
        x += 0.6
    ax.set_xticks(ticks, names, fontsize=8)
    ax.set_ylabel("GPU kernel ms")
    ax.set_title(f"Inference GPU time by stage, batch={batch} (C=compressed, R=raw)")
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "bp_speed_stage_gpu.png", dpi=130)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
for batch, marker in ((8, "o"), (16, "s"), (32, "^")):
    for mode in ("compressed", "raw"):
        pts = [(k, inter["training"][f"k{k}_{mode}_b{batch}"]) for k in ks]
        pts = [(k, v["mean"]) for k, v in pts if not v.get("oom")]
        mem = [(k, runs[k]["training"][f"{mode}_b{batch}"]) for k in ks]
        mem = [(k, v["peak_mem_gb"]) for k, v in mem if not v.get("oom")]
        style = "-" if mode == "compressed" else "--"
        if pts:
            axes[0].plot(*zip(*pts), style + marker, color=colors[mode], label=f"{mode} B={batch}")
        if mem:
            axes[1].plot(*zip(*mem), style + marker, color=colors[mode], label=f"{mode} B={batch}")
for ax, title, ylabel in ((axes[0], "Training fwd+bwd step (wall)", "ms / step"),
                          (axes[1], "Training peak memory", "GB")):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("bp_num_chunks K")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
axes[1].axhline(95.6, color="k", lw=0.8, ls=":")
axes[1].text(4.2, 92, "GPU capacity 96GB (missing points = OOM)", fontsize=8)
axes[0].legend(fontsize=7.5, ncol=2)
fig.tight_layout()
fig.savefig(out_dir / "bp_speed_training.png", dpi=130)
print("ok")
