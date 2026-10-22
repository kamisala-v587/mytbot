"""Plot HOM (held-out generation error and E_alt / E_match vs step) and the training loss of the AE-2 large runs."""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
runs = {"S-A main": "ae2l_main", "S-U": "ae2l_su", "text-only": "ae2l_textonly"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for label, name in runs.items():
    d = sorted(OUT.glob(f"{name}/*/heldout_monitor.jsonl"))[-1]
    rows = [json.loads(line) for line in d.read_text().splitlines()]
    steps = [r["step"] for r in rows]
    axes[0].plot(steps, [r["mse_matched"] for r in rows], marker="o", label=label)
    if name != "ae2l_textonly":
        axes[1].plot(steps, [r["alt_over_match"] for r in rows], marker="o", label=label)
    log = (OUT / "logs" / f"train_{name}.log").read_text().replace("\r", "\n")
    pts = {}
    for m in re.finditer(r"(\d+)/20000 \[[^\]]*loss=([0-9.]+)", log):
        pts[int(m.group(1))] = float(m.group(2))
    xs = sorted(k for k in pts if k % 250 == 0)
    axes[2].plot(xs, [pts[k] for k in xs], lw=1, label=label)
axes[0].set_title("held-out generation MSE (matched, 750 episodes)")
axes[1].set_title("held-out E_alt / E_match")
axes[1].axhline(1.1, color="gray", ls="--", lw=0.8)
axes[2].set_title("training loss (flow matching)")
axes[2].set_yscale("log")
for ax in axes:
    ax.set_xlabel("step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.tight_layout()
f = OUT / "ae2l_eval" / "hom_curves.png"
fig.savefig(f, dpi=110)
print(f)
