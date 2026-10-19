"""BP KV cache 实验配图（GPU kernel 时间）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(__file__).parent / "results"
out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else root
ks = [4, 8, 16, 32]
data = {k: json.loads((root / f"cache_k{k}.json").read_text()) for k in ks}


def t(section, mode, batch, k, stage="total"):
    return data[k][section][f"k{k}_{mode}_b{batch}"].get(stage, 0.0)


styles = {
    ("no_cache", "compressed"): ("tab:blue", "--", "compressed, no cache"),
    ("no_cache", "raw"): ("tab:red", "--", "raw, no cache"),
    ("cache_step", "compressed"): ("tab:blue", "-", "compressed + BP cache"),
    ("cache_step", "raw"): ("tab:red", "-", "raw + BP cache"),
}
fig, axes = plt.subplots(1, 4, figsize=(21, 4.6))
for ax, batch in zip(axes[:3], (1, 8, 32)):
    for (section, mode), (color, ls, label) in styles.items():
        ax.plot(ks, [t(section, mode, batch, k) for k in ks], ls + "o", color=color, label=label)
    ax.set_title(f"Per-step GPU time, batch={batch}")
    ax.set_ylabel("ms / inference step")
ax = axes[3]
for batch, color in ((1, "tab:green"), (8, "tab:orange"), (32, "tab:purple")):
    for section, ls, tag in (("no_cache", "--", "no cache"), ("cache_step", "-", "BP cache")):
        ratio = [t(section, "raw", batch, k) / t(section, "compressed", batch, k) for k in ks]
        ax.plot(ks, ratio, ls + "o", color=color, label=f"B={batch}, {tag}")
ax.axhline(1.0, color="k", lw=0.8, ls=":")
ax.set_title("Speedup from compression (raw / compressed)")
ax.set_ylabel("x")
for ax in axes:
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.set_xlabel("bp_num_chunks K")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)
fig.tight_layout()
fig.savefig(out_dir / "bp_cache_speed.png", dpi=130)

stages = [("vit", "ViT", "tab:gray"), ("bp_compressor", "BP compressor", "tab:green"),
          ("vlm_prefill", "VLM prefill", "tab:orange"), ("denoise_10steps", "denoise x10", "tab:purple")]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, batch in zip(axes, (1, 32)):
    x, ticks, names = 0.0, [], []
    for k in ks:
        for section, tag in (("no_cache", "NC"), ("cache_step", "C")):
            for mode in ("compressed", "raw"):
                bottom = 0.0
                for key, name, color in stages:
                    v = t(section, mode, batch, k, key)
                    ax.bar(x, v, bottom=bottom, color=color, width=0.8, label=name if x == 0 else None)
                    bottom += v
                ticks.append(x)
                names.append(f"{'C' if mode == 'compressed' else 'R'}\n{tag}")
                x += 1
        ax.text(x - 2.5, ax.get_ylim()[1] * 0.0, "", fontsize=8)
        x += 0.8
    ax.set_xticks(ticks, names, fontsize=7)
    ax.set_ylabel("GPU ms / step")
    ax.set_title(f"batch={batch}: K=4 | K=8 | K=16 | K=32  (C/R=compressed/raw, NC=no cache, C=BP cache)",
                 fontsize=9)
    ax.grid(axis="y", alpha=0.3)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "bp_cache_stage.png", dpi=130)
print("ok")
