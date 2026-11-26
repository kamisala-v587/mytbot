"""Reusable Stage A main-train monitor plots.

Reads the newest 8-GPU and 4-GPU runs under outputs/BPv5/stage_a/main
(or --dist8 / --local4). Safe to re-run while training is still writing logs.

Writes PNGs and summary.json next to the report assets.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/vla/workspace/my_tbot")
MAIN = ROOT / "outputs/BPv5/stage_a/main"
STATS_INDEX = ROOT / "norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d/stats_index.json"
SOURCE_ORDER = ("interndata", "agibot", "droid", "molmoact")

OUT = Path("/vla/workspace/code/conclusion/bpv5/assets/stage_a_main_monitor")

# 80D slot semantics. See code/conclusion/bpv5/80D_action_norm_stats_delta_abs.md
SLOTS = [
    (0, 7, "左臂关节"),
    (7, 10, "左EE xyz"),
    (10, 16, "左EE quat"),
    (16, 17, "左夹爪"),
    (17, 29, "预留"),
    (29, 30, "腰"),
    (30, 39, "预留"),
    (39, 40, "腰"),
    (40, 47, "右臂关节"),
    (47, 50, "右EE xyz"),
    (50, 56, "右EE quat"),
    (56, 57, "右夹爪"),
    (57, 74, "预留"),
    (74, 76, "头部"),
    (76, 77, "预留"),
    (77, 80, "底盘速度"),
]


def latest_run(world: int) -> Path:
    hits = []
    for train in MAIN.glob("*/train.log"):
        text = train.read_text(errors="ignore")[:200_000]
        if f"dist_loading rank 0/{world}:" in text:
            hits.append(train.parent)
    if not hits:
        raise SystemExit(f"no rank-0/{world} run under {MAIN}")
    return max(hits, key=lambda p: p.stat().st_mtime)


def read_loss(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f"empty loss log: {path}")
    step = np.array([int(r["step"]) for r in rows])
    loss = np.array([float(r["loss"]) for r in rows])
    return {"step": step, "loss": loss}


def read_hom(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def plot_loss(series: dict, title: str, path: Path, late_after: int = 2000) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(series["step"], series["loss"], color="#1f4e79", lw=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel("flow loss (100-step mean)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    late = series["step"] >= late_after
    if late.any():
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.plot(series["step"][late], series["loss"][late], color="#1f4e79", lw=1.4)
        ax.set_xlabel("step")
        ax.set_ylabel("flow loss (100-step mean)")
        ax.set_title(title + f" (step ≥ {late_after})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path.with_name(path.stem + "_late.png"), dpi=140)
        plt.close(fig)


def _hom_ratio(row: dict, key: str) -> float | None:
    x = row.get(key)
    if isinstance(x, dict):
        r = x.get("ratio")
        return float(r) if r is not None else None
    if isinstance(x, (int, float)):
        return float(x)
    return None


def plot_hom(rows: list[dict], title: str, path: Path) -> None:
    if not rows:
        return
    step = [r["step"] for r in rows]
    mse_keys = [
        ("mse_matched", "matched", "#1f4e79"),
        ("mse_blank", "blank", "#7f8c8d"),
        ("mse_alternate", "alt", "#d68910"),
        ("mse_wrong", "wrong", "#922b21"),
    ]
    ratio_keys = [
        ("blank_over_match", "blank / matched", "#7f8c8d"),
        ("alt_over_match", "alt / matched", "#d68910"),
        ("wrong_over_match", "wrong / matched", "#922b21"),
    ]
    ratios = {k: [_hom_ratio(r, k) for r in rows] for k, _, _ in ratio_keys}
    has_ratio = any(any(v is not None for v in vals) for vals in ratios.values())

    if has_ratio:
        fig, (ax0, ax1) = plt.subplots(
            2, 1, figsize=(8.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.15, 1.0]}
        )
    else:
        fig, ax0 = plt.subplots(figsize=(8.2, 4.2))
        ax1 = None

    for key, label, color in mse_keys:
        if key not in rows[0]:
            continue
        ax0.plot(step, [r[key] for r in rows], marker="o", color=color, label=label)
    ax0.set_ylabel("held-out action MSE")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=9)

    if ax1 is not None:
        for key, label, color in ratio_keys:
            ys = ratios[key]
            if all(v is None for v in ys):
                continue
            ax1.plot(step, ys, marker="o", color=color, label=label)
        ax1.axhline(1.0, color="#666", ls="--", lw=1, alpha=0.6)
        ax1.set_xlabel("step")
        ax1.set_ylabel("ratio")
        ax1.set_title(title.replace("action MSE", "condition ratios"))
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", fontsize=9)
    else:
        ax0.set_xlabel("step")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mask_text(mask: list[bool]) -> str:
    parts = []
    for lo, hi, name in SLOTS:
        if name == "预留":
            continue
        width = hi - lo
        on = sum(1 for i in range(lo, hi) if i < len(mask) and mask[i])
        if on:
            span = f"{lo}–{hi - 1}" if width > 1 else f"{lo}"
            parts.append(f"{name} {span}（{on}/{width}）")
    return "、".join(parts)


def robot_masks(robots: set[str], repos_by_robot: dict[str, str]) -> dict[str, dict]:
    index = json.loads(STATS_INDEX.read_text())
    out = {}
    cache: dict[str, list[bool]] = {}
    for robot in sorted(robots):
        repo = repos_by_robot.get(robot)
        stats_path = index.get(repo or "")
        if not stats_path:
            out[robot] = {"n_valid": None, "slots": "", "stats": None}
            continue
        if stats_path not in cache:
            meta = json.loads(Path(stats_path).read_text()).get("metadata") or {}
            cache[stats_path] = [bool(v > 0.5) for v in meta.get("action_mask") or []]
        mask = cache[stats_path]
        out[robot] = {
            "n_valid": int(sum(mask)),
            "slots": mask_text(mask),
            "stats": stats_path,
        }
    return out


def aggregate_samples(run: Path) -> dict:
    files = sorted(run.glob("sample_loss_rank*.csv"))
    if not files:
        return {}
    by_source = defaultdict(lambda: [0, 0.0])
    by_robot = defaultdict(lambda: [0, 0.0])
    repos_by_robot: dict[str, str] = {}
    # step-bin mean by source, bin = 1000 steps
    bins: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    n = 0
    for path in files:
        with path.open() as fh:
            for row in csv.DictReader(fh):
                repo = row["repo"]
                source = repo.split("/", 1)[0]
                robot = f"{row['robot']} ({source})"
                loss = float(row["loss"])
                step = int(row["step"])
                by_source[source][0] += 1
                by_source[source][1] += loss
                by_robot[robot][0] += 1
                by_robot[robot][1] += loss
                repos_by_robot.setdefault(robot, repo)
                b = bins[source][step // 1000]
                b[0] += 1
                b[1] += loss
                n += 1
    sources = {
        k: {"n": v[0], "share": v[0] / n, "mean_loss": v[1] / v[0]}
        for k, v in sorted(by_source.items(), key=lambda kv: (SOURCE_ORDER.index(kv[0]) if kv[0] in SOURCE_ORDER else len(SOURCE_ORDER), kv[0]))
    }
    robots = {
        k: {"n": v[0], "share": v[0] / n, "mean_loss": v[1] / v[0]}
        for k, v in sorted(by_robot.items(), key=lambda kv: -kv[1][0])
    }
    curves = {}
    for source, buckets in bins.items():
        xs, ys = [], []
        for b in sorted(buckets):
            c, s = buckets[b]
            xs.append((b + 0.5) * 1000)
            ys.append(s / c)
        curves[source] = {"step": xs, "loss": ys}
    return {
        "n": n,
        "sources": sources,
        "robots": robots,
        "curves": curves,
        "masks": robot_masks(set(robots), repos_by_robot),
    }


def plot_bars(items: list[tuple[str, float, int]], title: str, ylabel: str, path: Path) -> None:
    labels = [k for k, _, _ in items]
    vals = [v for _, v, _ in items]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(labels[::-1], vals[::-1], color="#1f4e79")
    ax.set_xlabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_source_curves(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    colors = {"agibot": "#1f4e79", "interndata": "#1e8449", "droid": "#d68910", "molmoact": "#6c3483"}
    for source in sorted(curves, key=lambda s: (SOURCE_ORDER.index(s) if s in SOURCE_ORDER else len(SOURCE_ORDER), s)):
        c = curves[source]
        ax.plot(c["step"], c["loss"], label=source, color=colors.get(source), lw=1.4)
    ax.set_xlabel("step (bin center, 1000-step bins)")
    ax.set_ylabel("per-trajectory flow loss")
    ax.set_title("Local 4-GPU loss by data source")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pack_run(run: Path) -> dict:
    loss = read_loss(run / "loss.log")
    hom = read_hom(run / "heldout_monitor.jsonl")
    return {
        "run": str(run),
        "last_step": int(loss["step"][-1]),
        "last_loss": float(loss["loss"][-1]),
        "loss_at": {
            str(int(s)): float(v)
            for s, v in zip(loss["step"], loss["loss"])
            if int(s) in {500, 1000, 2000, 5000, 10000, 15000} or int(s) == int(loss["step"][-1])
        },
        "hom": [
            {
                "step": r["step"],
                "mse_matched": r.get("mse_matched"),
                "mse_blank": r.get("mse_blank"),
                "mse_alternate": r.get("mse_alternate"),
                "mse_wrong": r.get("mse_wrong"),
                "blank_over_match": (r.get("blank_over_match") or {}).get("ratio"),
                "alt_over_match": (r.get("alt_over_match") or {}).get("ratio"),
                "wrong_over_match": (r.get("wrong_over_match") or {}).get("ratio"),
                "by_source": {
                    k: {"segments": v.get("segments"), "mse_matched": v.get("mse_matched")}
                    for k, v in (r.get("by_source") or {}).items()
                },
            }
            for r in hom
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist8", type=Path, default=None)
    ap.add_argument("--local4", type=Path, default=None)
    ap.add_argument("--skip-local4", action="store_true", help="Only plot 8-GPU main (local4 run may be deleted).")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    dist = args.dist8 or latest_run(8)
    local = None
    if not args.skip_local4:
        try:
            local = args.local4 or latest_run(4)
        except SystemExit:
            local = None
            print("warn: no 4-GPU run; plotting dist8 only", flush=True)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    d_loss = read_loss(dist / "loss.log")
    d_hom = read_hom(dist / "heldout_monitor.jsonl")
    plot_loss(d_loss, "8-GPU main flow loss", out / "F1_dist8_loss.png")
    plot_hom(d_hom, "8-GPU main held-out action MSE", out / "F3_dist8_hom_mse.png")

    summary: dict = {"dist8": pack_run(dist), "local4": None, "local4_samples": None}
    if local is not None:
        l_loss = read_loss(local / "loss.log")
        l_hom = read_hom(local / "heldout_monitor.jsonl")
        plot_loss(l_loss, "4-GPU local flow loss", out / "F4_local4_loss.png")
        plot_hom(l_hom, "4-GPU local held-out action MSE", out / "F6_local4_hom_mse.png")
        samples = aggregate_samples(local)
        if samples:
            plot_source_curves(samples["curves"], out / "F7_local4_source_loss.png")
            plot_bars(
                [(k, v["mean_loss"], v["n"]) for k, v in samples["sources"].items()],
                "Local 4-GPU mean flow loss by source",
                "mean per-trajectory flow loss",
                out / "F8_local4_source_mean.png",
            )
            plot_bars(
                [(k, v["mean_loss"], v["n"]) for k, v in samples["robots"].items()],
                "Local 4-GPU mean flow loss by robot",
                "mean per-trajectory flow loss",
                out / "F9_local4_robot_mean.png",
            )
            plot_bars(
                [(k, v["share"], v["n"]) for k, v in samples["sources"].items()],
                "Local 4-GPU sample share by source",
                "share of logged trajectories",
                out / "F10_local4_source_share.png",
            )
        summary["local4"] = pack_run(local)
        summary["local4_samples"] = {
            "n": samples.get("n"),
            "sources": samples.get("sources"),
            "robots": {
                k: {**v, **samples["masks"].get(k, {})} for k, v in samples.get("robots", {}).items()
            },
        }
    # curves are large; keep them out of the report sidecar
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({"dist8": str(dist), "local4": str(local) if local else None, "out": str(out)}, ensure_ascii=False))
    print("dist8", summary["dist8"]["last_step"], summary["dist8"]["last_loss"])
    if summary["local4"]:
        print("local4", summary["local4"]["last_step"], summary["local4"]["last_loss"])

if __name__ == "__main__":
    main()
