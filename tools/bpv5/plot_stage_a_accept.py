#!/usr/bin/env python3
"""Plot Stage A offline acceptance curves: loss, HOM, HOM ratios, linear-probe R²."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUN = Path("/vla/workspace/my_tbot/outputs/BPv5/stage_a/main/2026-11-17_23-32")
PROBE = Path("/vla/workspace/code/conclusion/bpv5/assets/stage_a_accept_probe")
OUT = Path("/vla/workspace/code/conclusion/bpv5/assets/stage_a_accept")


def ratio_val(x):
    return float(x["ratio"] if isinstance(x, dict) else x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--probe_dir", type=Path, default=PROBE)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--max_step", type=int, default=40000)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loss_rows = list(csv.DictReader((args.run / "loss.log").open()))
    steps = np.array([int(r["step"]) for r in loss_rows])
    loss = np.array([float(r["loss"]) for r in loss_rows])
    m = steps <= args.max_step
    steps, loss = steps[m], loss[m]

    hom = []
    for line in (args.run / "heldout_monitor.jsonl").read_text().splitlines():
        d = json.loads(line)
        if d["step"] <= args.max_step:
            hom.append(d)
    hom.sort(key=lambda d: d["step"])

    probes = []
    for p in sorted(args.probe_dir.glob("probe_step_*.json")):
        d = json.loads(p.read_text())
        if d["step"] <= args.max_step:
            probes.append(d)
    probes.sort(key=lambda d: d["step"])

    # F1 loss
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(steps, loss, color="#1f4e79", lw=1.2)
    ax.set_xlabel("step")
    ax.set_ylabel("flow loss (100-step mean)")
    ax.set_title("Stage A train loss (8-GPU)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "F1_loss.png", dpi=140)
    plt.close(fig)

    # F2 HOM MSE
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    hs = [d["step"] for d in hom]
    for key, lab, c in [
        ("mse_matched", "matched", "#2ca02c"),
        ("mse_blank", "blank", "#ff7f0e"),
        ("mse_alternate", "alt", "#9467bd"),
        ("mse_wrong", "wrong", "#d62728"),
    ]:
        ax.plot(hs, [d[key] for d in hom], marker="o", label=lab, color=c)
    ax.set_xlabel("step")
    ax.set_ylabel("action MSE")
    ax.set_title("HOM action MSE")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "F2_hom_mse.png", dpi=140)
    plt.close(fig)

    # F3 HOM ratios
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, lab, c in [
        ("blank_over_match", "blank/matched", "#ff7f0e"),
        ("alt_over_match", "alt/matched", "#9467bd"),
        ("wrong_over_match", "wrong/matched", "#d62728"),
    ]:
        ax.plot(hs, [ratio_val(d[key]) for d in hom], marker="o", label=lab, color=c)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("ratio")
    ax.set_title("HOM demo gain ratios")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out / "F3_hom_ratio.png", dpi=140)
    plt.close(fig)

    # F4 probe R² — descriptive titles (no CJK font on this host); drop shuffle Z
    label_names = {
        "A": "full-traj EE delta",
        "B": "first-close EE delta",
        "C": "first-open EE delta",
    }
    if probes:
        fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4), sharey=True)
        methods = [
            ("bp", "trained BP", "#1f77b4"),
            ("random_qf", "random QF", "#7f7f7f"),
        ]
        for ax, lab in zip(axes, ("A", "B", "C")):
            ps = [d["step"] for d in probes]
            for mk, name, c in methods:
                ys = [(d["labels"].get(lab, {}).get("r2") or {}).get(mk, float("nan")) for d in probes]
                ax.plot(ps, ys, marker="o", label=name, color=c)
            leaf_ys = [(d["labels"].get(lab, {}).get("r2") or {}).get("leaf_mean", float("nan")) for d in probes]
            leaf_v = next((y for y in leaf_ys if y == y), float("nan"))
            if leaf_v == leaf_v:
                ax.axhline(leaf_v, color="#2ca02c", ls="--", lw=1.2, label=f"leaf mean {leaf_v:.2f}")
            ax.set_title(label_names[lab])
            ax.set_xlabel("step")
            ax.axhline(0.0, color="gray", ls=":", lw=0.8)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("test R² (mean over xyz)")
        axes[0].legend(frameon=False, fontsize=8)
        fig.suptitle("Linear probe on Z_demo (displacement readouts)", y=1.02)
        fig.tight_layout()
        fig.savefig(args.out / "F4_probe_r2.png", dpi=140, bbox_inches="tight")
        plt.close(fig)

        # F5 by source at max step
        last = probes[-1]
        fig, ax = plt.subplots(figsize=(7.6, 3.4))
        labels = ["A", "B", "C"]
        x = np.arange(len(labels))
        w = 0.35
        agi = [((last["labels"][k].get("r2_by_source") or {}).get("bp") or {}).get("agibot", np.nan) for k in labels]
        idn = [((last["labels"][k].get("r2_by_source") or {}).get("bp") or {}).get("interndata", np.nan) for k in labels]
        ax.bar(x - w / 2, agi, w, label="agibot (real)", color="#1f4e79")
        ax.bar(x + w / 2, idn, w, label="interndata (sim)", color="#6fa8dc")
        ax.set_xticks(x)
        ax.set_xticklabels([label_names[k] for k in labels], fontsize=9)
        ax.set_ylabel("trained BP test R²")
        ax.set_title(f"Probe R² by source @ step {last['step']}")
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.legend(frameon=False)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out / "F5_probe_by_source.png", dpi=140)
        plt.close(fig)

    summary = {
        "max_step": args.max_step,
        "loss_last": float(loss[-1]) if len(loss) else None,
        "hom": [
            {
                "step": d["step"],
                "matched": d["mse_matched"],
                "blank_over_match": ratio_val(d["blank_over_match"]),
                "alt_over_match": ratio_val(d["alt_over_match"]),
                "wrong_over_match": ratio_val(d["wrong_over_match"]),
                "mse_blank": d["mse_blank"],
                "mse_alternate": d["mse_alternate"],
                "mse_wrong": d["mse_wrong"],
            }
            for d in hom
        ],
        "probes": [{"step": d["step"], "r2": {k: (v.get("r2") or {}) for k, v in d["labels"].items()}} for d in probes],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"out": str(args.out), "n_hom": len(hom), "n_probe": len(probes)}, indent=2))


if __name__ == "__main__":
    main()
