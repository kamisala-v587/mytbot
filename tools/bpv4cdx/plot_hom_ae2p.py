"""HOM curves (held-out matched MSE and E_alt / E_match vs step) of the AE-2 supplement next to the AE-2 large runs.
usage: python tools/bpv4cdx/plot_hom_ae2p.py <outputs/BPv4CDX/stage_a/v2> <out.png>"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, PNG = Path(sys.argv[1]), Path(sys.argv[2])
runs = {"S-A P5": ("ae2l_main", "C0", "-"), "S-U P5": ("ae2l_su", "C1", "-"), "S-UF P5": ("ae2l_suf", "C2", "-"),
        "S-UF P64": ("ae2l_suf_p64", "C2", "--"), "S-U P64": ("ae2l_su_p64", "C1", "--"),
        "S-A P5 seed 43": ("ae2l_main_seed43", "C0", ":"), "S-U P5 seed 43": ("ae2l_su_seed43", "C1", ":")}
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for label, (name, color, ls) in runs.items():
    d = sorted(OUT.glob(f"{name}/*/heldout_monitor.jsonl"))[-1]
    rows = [json.loads(line) for line in d.read_text().splitlines()]
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, [r["mse_matched"] for r in rows], color=color, ls=ls, marker="o", ms=3, label=label)
    axes[1].plot(steps, [r["alt_over_match"] for r in rows], color=color, ls=ls, marker="o", ms=3, label=label)
axes[0].set_title("held-out generation MSE (matched, 750 episodes)")
axes[0].set_yscale("log")
axes[1].set_title("held-out E_alt / E_match")
axes[1].axhline(1.1, color="gray", ls="--", lw=0.8)
for ax in axes:
    ax.set_xlabel("step")
    ax.grid(alpha=0.3)
axes[1].legend(fontsize=8)
fig.tight_layout()
PNG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(PNG, dpi=120)
print(PNG)
