#!/usr/bin/env python3
"""Plot Stage A overfit diagnostic loss curves (AE-1 G3 style)."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STAGE = Path("/vla/workspace/my_tbot/outputs/BPv5/stage_a")
OUT = STAGE / "reports" / "overfit_diag"
RUNS = [
    ("rt1  place_shoe×1", STAGE / "overfit_rt1", "#0072B2"),
    ("rt8  baseline", STAGE / "overfit_rt8", "#56B4E9"),
    ("idn32 interndata×32", STAGE / "overfit_idn32", "#009E73"),
    ("agi32 agibot×32", STAGE / "overfit_agi32", "#D55E00"),
    ("rt1_alldims", STAGE / "overfit_rt1_alldims", "#E69F00"),
    ("rt1_ae3loss", STAGE / "overfit_rt1_ae3loss", "#CC79A7"),
    ("agi32_alldims", STAGE / "overfit_agi32_alldims", "#F0E442"),
    ("rt1_fullprefix", STAGE / "overfit_rt1_fullprefix", "#000000"),
    ("rt1_ae05", STAGE / "overfit_rt1_ae05", "#999999"),
    ("rt8_nocausal", STAGE / "stage_a_overfit_rt8_nocausal", "#0072B2"),
    ("rt8_ae05", STAGE / "stage_a_overfit_rt8_ae05", "#D55E00"),
    ("rt8_fullprefix", STAGE / "stage_a_overfit_rt8_fullprefix", "#009E73"),
    ("rt8_nocausal_ae05", STAGE / "stage_a_overfit_rt8_nocausal_ae05", "#CC79A7"),
    ("rt8_bs8", STAGE / "overfit_rt8_bs8", "#0072B2"),
    ("rt8_bs16", STAGE / "overfit_rt8_bs16", "#56B4E9"),
    ("rt8_bs8_ae3loss", STAGE / "overfit_rt8_bs8_ae3loss", "#E69F00"),
    ("rt8_bs8_alldims", STAGE / "overfit_rt8_bs8_alldims", "#CC79A7"),
    ("S0 ae3recipe", STAGE / "ladder_s0_ae3recipe", "#0072B2"),
    ("S5 lr3e4", STAGE / "ladder_s5_lr3e4", "#D55E00"),
    ("S7 lora4", STAGE / "ladder_s7_lora4", "#009E73"),
    ("S1 causal", STAGE / "ladder_s1_causal", "#E69F00"),
    ("S2 ae025", STAGE / "ladder_s2_ae025", "#56B4E9"),
    ("S3 statsmask", STAGE / "ladder_s3_statsmask", "#CC79A7"),
    ("S4 ideal", STAGE / "ladder_s4_ideal", "#000000"),
    ("S6 fullprefix", STAGE / "ladder_s6_fullprefix", "#999999"),
    ("src_rt64", STAGE / "srcdiag_src_rt64", "#0072B2"),
    ("src_idn64", STAGE / "srcdiag_src_idn64", "#009E73"),
    ("src_agi64", STAGE / "srcdiag_src_agi64", "#D55E00"),
    ("mix_rt_idn", STAGE / "srcdiag_mix_rt_idn", "#56B4E9"),
    ("mix_rt_agi", STAGE / "srcdiag_mix_rt_agi", "#E69F00"),
    ("mix_idn_agi", STAGE / "srcdiag_mix_idn_agi", "#CC79A7"),
    ("mix_rt_idn_agi", STAGE / "srcdiag_mix_rt_idn_agi", "#000000"),
]
SMOOTH = 100


def load_loss(run: Path) -> pd.DataFrame:
    cands = sorted(run.glob("*/loss.log")) if run.is_dir() else []
    if (run / "loss.log").is_file():
        path = run / "loss.log"
    elif cands:
        path = cands[-1]
    else:
        return pd.DataFrame()
    text = path.read_text()
    text = text[: text.rfind("\n") + 1]
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        try:
            rows.append({"step": int(r["step"]), "loss": float(r["loss"])})
        except (KeyError, ValueError, TypeError):
            continue
    return pd.DataFrame(rows).drop_duplicates("step").sort_values("step").reset_index(drop=True)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.facecolor": "white",
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.3))
    summary = []
    for name, root, color in RUNS:
        df = load_loss(root)
        if df.empty:
            print(f"skip {name}: no loss.log under {root}")
            continue
        ema = df["loss"].ewm(span=SMOOTH).mean()
        axs[0].plot(df["step"], df["loss"], color=color, lw=0.6, alpha=0.25)
        axs[0].plot(df["step"], ema, color=color, lw=1.8, label=name)
        axs[1].plot(df["step"], ema, color=color, lw=1.8, label=name)
        last = float(df["loss"].iloc[-1])
        mid = float(df.loc[df["step"] >= df["step"].max() // 2, "loss"].mean())
        early = float(df.head(max(1, len(df) // 10))["loss"].mean())
        summary.append(
            {
                "run": name,
                "steps": int(df["step"].iloc[-1]),
                "loss_early": early,
                "loss_last": last,
                "drop_ratio": 1.0 - last / early if early > 0 else float("nan"),
            }
        )
        print(f"{name}: step {int(df.step.iloc[-1])} last={last:.4f} early≈{early:.4f} drop={1-last/early:.1%}")

    axs[0].set(title="Overfit diag — train loss", xlabel="step", ylabel="flow MSE", yscale="log")
    axs[0].legend(frameon=False, fontsize=8)
    axs[1].set(title="Same (EMA only, linear y)", xlabel="step", ylabel="flow MSE (EMA)")
    axs[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT / f"overfit_loss.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print("→", OUT)


if __name__ == "__main__":
    main()
