#!/usr/bin/env python3
"""Compare frozen-LoRA B-1 vs LoRA-layer-0 B-1 at a shared step (default 10000).

Writes:
  outputs/BPv5/stage_b/reports/b1_vs_lora_l0_10k.md
  outputs/BPv5/stage_b/reports/b1_vs_lora_l0_10k.png
  outputs/BPv5/stage_b/reports/b1_vs_lora_l0_10k.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/vla/workspace/my_tbot")
DEFAULT_BASE = REPO / "outputs/BPv5/stage_b/b1/2026-11-19_02-10"
REPORTS = REPO / "outputs/BPv5/stage_b/reports"


def newest_run(parent: Path) -> Path:
    cands = sorted(parent.glob("*/loss.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"no loss.log under {parent}")
    return cands[0].parent


def load_loss(run: Path) -> list[dict]:
    rows = list(csv.DictReader((run / "loss.log").open()))
    out = []
    for r in rows:
        out.append(
            {
                "step": int(float(r["step"])),
                "loss": float(r["loss"]),
                "grad_norm": float(r["grad_norm"]),
                "lr": float(r["lr"]),
                "update_s": float(r.get("update_s") or 0),
            }
        )
    win = 5
    for i, s in enumerate(out):
        chunk = out[max(0, i - win + 1) : i + 1]
        s["loss_ma"] = sum(x["loss"] for x in chunk) / len(chunk)
    return out


def load_hom(run: Path) -> list[dict]:
    p = run / "heldout_monitor.jsonl"
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.open() if line.strip()]


def at_step(series: list[dict], step: int) -> dict | None:
    exact = [s for s in series if s["step"] == step]
    if exact:
        return exact[-1]
    if not series:
        return None
    return min(series, key=lambda s: abs(s["step"] - step))


def hom_at(hom: list[dict], step: int) -> dict | None:
    exact = [h for h in hom if int(h["step"]) == step]
    if exact:
        return exact[-1]
    if not hom:
        return None
    return min(hom, key=lambda h: abs(int(h["step"]) - step))


def mean_loss(series: list[dict], lo: int, hi: int) -> float | None:
    xs = [s["loss"] for s in series if lo <= s["step"] <= hi]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--lora", type=Path, default=None)
    ap.add_argument("--step", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=REPORTS)
    args = ap.parse_args()

    lora_run = args.lora or newest_run(REPO / "outputs/BPv5/stage_b/b1_lora_l0")
    base_series = load_loss(args.baseline)
    lora_series = load_loss(lora_run)
    base_hom = load_hom(args.baseline)
    lora_hom = load_hom(lora_run)

    last_lora = lora_series[-1]["step"] if lora_series else 0
    if last_lora < args.step:
        raise SystemExit(f"LoRA run only at step {last_lora} < {args.step}; not ready")

    b10 = at_step(base_series, args.step)
    l10 = at_step(lora_series, args.step)
    bh = hom_at(base_hom, args.step)
    lh = hom_at(lora_hom, args.step)

    payload = {
        "compare_step": args.step,
        "baseline": {
            "run": str(args.baseline),
            "at_step": b10,
            "mean_8k_10k": mean_loss(base_series, args.step - 2000, args.step),
            "hom": bh,
        },
        "lora_l0": {
            "run": str(lora_run),
            "at_step": l10,
            "mean_8k_10k": mean_loss(lora_series, args.step - 2000, args.step),
            "hom": lh,
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "b1_vs_lora_l0_10k.json"
    json_path.write_text(json.dumps(payload, indent=2))

    # figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for series, color, label in (
        (base_series, "#1f77b4", "B-1 frozen LoRA"),
        (lora_series, "#ff7f0e", "B-1 LoRA layer0"),
    ):
        steps = [s["step"] for s in series if s["step"] <= args.step + 500]
        axes[0, 0].plot(steps, [s["loss_ma"] for s in series if s["step"] <= args.step + 500], color=color, lw=1.6, label=label)
        axes[0, 1].plot(steps, [s["grad_norm"] for s in series if s["step"] <= args.step + 500], color=color, lw=1.2, label=label)
    axes[0, 0].axvline(args.step, color="#d62728", ls="--", lw=1)
    axes[0, 0].set_title("Train loss (MA5)"); axes[0, 0].set_xlabel("step"); axes[0, 0].legend(fontsize=8)
    axes[0, 1].axvline(args.step, color="#d62728", ls="--", lw=1)
    axes[0, 1].set_title("grad_norm"); axes[0, 1].set_xlabel("step"); axes[0, 1].legend(fontsize=8)

    # HOM bars at compare step
    labels = ["matched", "blank", "alt", "wrong"]
    keys = ["mse_matched", "mse_blank", "mse_alternate", "mse_wrong"]
    x = list(range(len(labels)))
    w = 0.35
    if bh:
        axes[1, 0].bar([i - w / 2 for i in x], [bh[k] for k in keys], width=w, color="#1f77b4", label="frozen")
    if lh:
        axes[1, 0].bar([i + w / 2 for i in x], [lh[k] for k in keys], width=w, color="#ff7f0e", label="lora_l0")
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(labels)
    axes[1, 0].set_title(f"HOM MSE @ step {args.step}"); axes[1, 0].legend(fontsize=8)

    # ratios
    ratio_keys = [
        ("blank_over_match", "blank/match"),
        ("alt_over_match", "alt/match"),
        ("wrong_over_match", "wrong/match"),
    ]
    names = [n for _, n in ratio_keys]
    xr = list(range(len(names)))
    if bh:
        axes[1, 1].bar([i - w / 2 for i in xr], [bh[k]["ratio"] for k, _ in ratio_keys], width=w, color="#1f77b4", label="frozen")
    if lh:
        axes[1, 1].bar([i + w / 2 for i in xr], [lh[k]["ratio"] for k, _ in ratio_keys], width=w, color="#ff7f0e", label="lora_l0")
    axes[1, 1].axhline(1.0, color="#888", ls=":", lw=1)
    axes[1, 1].set_xticks(xr); axes[1, 1].set_xticklabels(names)
    axes[1, 1].set_title(f"HOM ratios @ step {args.step}"); axes[1, 1].legend(fontsize=8)

    fig.suptitle(f"BPv5 B-1 comparison at {args.step} steps", fontsize=12)
    png_path = args.out_dir / "b1_vs_lora_l0_10k.png"
    fig.savefig(png_path, dpi=140)

    def fmt(x):
        return f"{x:.4f}" if isinstance(x, float) else str(x)

    md = []
    md.append(f"# BPv5 B-1 vs LoRA-L0 @ {args.step} steps\n")
    md.append("## Setup\n")
    md.append("| | frozen LoRA B-1 | LoRA layer-0 B-1 |")
    md.append("|---|---|---|")
    md.append(f"| run | `{args.baseline}` | `{lora_run}` |")
    md.append("| LoRA grads | none | `lora_train_layer_ids=[0]` |")
    md.append("| phase | b1 | b1 |")
    md.append("| AE / Q-Former | AE train, Q-Former freeze | same |")
    md.append("\n## Train loss\n")
    md.append("| metric | frozen | lora_l0 | Δ (lora−frozen) |")
    md.append("|---|---:|---:|---:|")
    if b10 and l10:
        d = l10["loss"] - b10["loss"]
        md.append(f"| loss @ {args.step} | {fmt(b10['loss'])} | {fmt(l10['loss'])} | {d:+.4f} |")
        d2 = (payload["lora_l0"]["mean_8k_10k"] or 0) - (payload["baseline"]["mean_8k_10k"] or 0)
        md.append(
            f"| mean loss [{args.step-2000},{args.step}] | {fmt(payload['baseline']['mean_8k_10k'])} | {fmt(payload['lora_l0']['mean_8k_10k'])} | {d2:+.4f} |"
        )
        md.append(f"| grad_norm @ {args.step} | {fmt(b10['grad_norm'])} | {fmt(l10['grad_norm'])} | {l10['grad_norm']-b10['grad_norm']:+.4f} |")
    md.append("\n## Held-out monitor (HOM)\n")
    if bh and lh:
        md.append("| metric | frozen | lora_l0 | Δ |")
        md.append("|---|---:|---:|---:|")
        for k, name in (
            ("mse_matched", "mse matched"),
            ("mse_blank", "mse blank"),
            ("mse_alternate", "mse alt"),
            ("mse_wrong", "mse wrong"),
        ):
            md.append(f"| {name} | {fmt(bh[k])} | {fmt(lh[k])} | {lh[k]-bh[k]:+.4f} |")
        for k, name in ratio_keys:
            md.append(
                f"| {name} | {fmt(bh[k]['ratio'])} | {fmt(lh[k]['ratio'])} | {lh[k]['ratio']-bh[k]['ratio']:+.4f} |"
            )
        md.append("\n### Advice\n")
        md.append(f"- frozen: `{bh.get('advice')}`")
        md.append(f"- lora_l0: `{lh.get('advice')}`")
    else:
        md.append("_HOM missing for one or both runs at this step._\n")
    md.append("\n## Artifacts\n")
    md.append(f"- curves: `{png_path}`")
    md.append(f"- json: `{json_path}`")
    md.append("\n## Reading\n")
    md.append(
        "- Lower matched HOM is better. blank/alt/wrong ratios > 1 suggest the demo signal is used; ≈1 means weak demo dependence.\n"
        "- Train loss alone is not decisive for Stage B; prefer HOM matched + ratios.\n"
    )
    md_path = args.out_dir / "b1_vs_lora_l0_10k.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"wrote {md_path}")
    print(f"wrote {png_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
