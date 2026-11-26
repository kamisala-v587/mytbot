#!/usr/bin/env python3
"""Scan Stage A leaves and verify ABC probe labels can be computed from state_80D.

A: end-start EE xyz of the chosen arm
B: first gripper-close EE xyz minus start
C: first gripper-open (after close) EE xyz minus start

Arm side is chosen per episode (left=16/7:10, right=56/47:50); not fixed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from lerobot.policies.BPV5QFormer import data80d as D

LEFT = {"ee": slice(7, 10), "grip": 16, "name": "left"}
RIGHT = {"ee": slice(47, 50), "grip": 56, "name": "right"}
ARMS = (LEFT, RIGHT)


def _finite_ee(state: np.ndarray, sl: slice) -> bool:
    x = state[:, sl]
    return bool(np.isfinite(x).all() and np.any(np.abs(x).sum(axis=1) > 0))


def _arm_ok(s_mask: np.ndarray, arm: dict) -> bool:
    ee = s_mask[arm["ee"]]
    return bool(ee.all() and s_mask[arm["grip"]])


def _thresholds2(grip: np.ndarray) -> tuple[float, float] | None:
    lo, hi = float(np.quantile(grip, 0.10)), float(np.quantile(grip, 0.90))
    if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-4:
        return None
    t_open = lo + 0.25 * (hi - lo)
    t_close = lo + 0.75 * (hi - lo)
    return t_open, t_close


def _polarity(grip: np.ndarray, t_open: float, t_close: float) -> str:
    """Return 'high_is_closed' or 'low_is_closed' from which band is visited after motion starts."""
    # Heuristic: the more extreme band that is occupied in the middle half is "closed"
    # if the start is nearer the other band.
    n = len(grip)
    a, b = max(1, n // 4), max(2, 3 * n // 4)
    mid = grip[a:b]
    start = float(np.median(grip[: max(1, n // 10)]))
    frac_high = float(np.mean(mid >= t_close))
    frac_low = float(np.mean(mid <= t_open))
    if start <= (t_open + t_close) / 2:
        # started open-ish low → closed is high, or vice versa
        return "high_is_closed" if frac_high >= frac_low else "low_is_closed"
    return "low_is_closed" if frac_low >= frac_high else "high_is_closed"


def _first_cross(grip: np.ndarray, enter_closed: bool, pol: str, t_open: float, t_close: float, start_i: int = 0):
    """First index >= start_i where gripper enters closed or open band."""
    if pol == "high_is_closed":
        closed = grip >= t_close
        opened = grip <= t_open
    else:
        closed = grip <= t_open
        opened = grip >= t_close
    target = closed if enter_closed else opened
    # require previously not in that band
    prev = ~target
    for i in range(max(start_i, 1), len(grip)):
        if prev[i - 1] and target[i]:
            return i
    return None


def labels_for_episode(state: np.ndarray, s_mask: np.ndarray) -> dict:
    n = int(state.shape[0])
    out = {
        "n": n,
        "A": None,
        "B": None,
        "C": None,
        "arm": None,
        "t_close": None,
        "t_open": None,
        "reason": None,
    }
    if n < 16:
        out["reason"] = "too_short"
        return out

    candidates = []
    for arm in ARMS:
        if not _arm_ok(s_mask, arm):
            continue
        ee = state[:, arm["ee"]].astype(np.float64)
        grip = state[:, arm["grip"]].astype(np.float64)
        if not np.isfinite(ee).all() or not np.isfinite(grip).all():
            continue
        thr = _thresholds2(grip)
        if thr is None:
            # still allow A if EE moves
            disp = ee[-1] - ee[0]
            if float(np.linalg.norm(disp)) >= 0.01:
                candidates.append(
                    {
                        "arm": arm["name"],
                        "A": disp.tolist(),
                        "B": None,
                        "C": None,
                        "t_close": None,
                        "t_open": None,
                        "grip_span": float(np.ptp(grip)),
                        "A_norm": float(np.linalg.norm(disp)),
                    }
                )
            continue
        t_open, t_close = thr
        pol = _polarity(grip, t_open, t_close)
        # margin: events not in first/last 5%
        lo_i, hi_i = int(0.05 * n), int(0.95 * n)
        tc = _first_cross(grip, True, pol, t_open, t_close, 0)
        if tc is not None and not (lo_i <= tc <= hi_i):
            tc = None
        # C only after a valid close (never search open from t=0 alone).
        to = _first_cross(grip, False, pol, t_open, t_close, tc + 1) if tc is not None else None
        if to is not None and not (lo_i <= to <= hi_i):
            to = None
        if to is not None and tc is not None and to <= tc:
            to = None
        disp_a = ee[-1] - ee[0]
        disp_b = (ee[tc] - ee[0]) if tc is not None else None
        disp_c = (ee[to] - ee[0]) if to is not None else None
        if disp_b is not None and float(np.linalg.norm(disp_b)) < 0.01:
            disp_b, tc = None, None
            disp_c, to = None, None
        if disp_b is None:
            disp_c, to = None, None
        if disp_c is not None and float(np.linalg.norm(disp_c)) < 0.01:
            disp_c, to = None, None
        candidates.append(
            {
                "arm": arm["name"],
                "A": disp_a.tolist(),
                "B": None if disp_b is None else disp_b.tolist(),
                "C": None if disp_c is None else disp_c.tolist(),
                "t_close": tc,
                "t_open": to,
                "grip_span": float(np.ptp(grip)),
                "A_norm": float(np.linalg.norm(disp_a)),
                "polarity": pol,
                "t_open_thr": t_open,
                "t_close_thr": t_close,
            }
        )

    if not candidates:
        out["reason"] = "no_valid_arm"
        return out

    # Prefer arm with B+C, then B, then larger grip span / A motion.
    def score(c):
        return (
            int(c["B"] is not None) + int(c["C"] is not None),
            int(c["B"] is not None),
            c["grip_span"],
            c["A_norm"],
        )

    best = max(candidates, key=score)
    if best["A_norm"] < 0.01 and best["B"] is None:
        out["reason"] = "static_ee"
        return out
    out.update(
        {
            "A": best["A"],
            "B": best["B"],
            "C": best["C"],
            "arm": best["arm"],
            "t_close": best["t_close"],
            "t_open": best["t_open"],
            "reason": "ok",
            "A_norm_m": best["A_norm"],
            "B_norm_m": None if best["B"] is None else float(np.linalg.norm(best["B"])),
            "C_norm_m": None if best["C"] is None else float(np.linalg.norm(best["C"])),
            "polarity": best.get("polarity"),
        }
    )
    return out


def leaves_from_stats_index(index: dict[str, str], sources: list[str]) -> dict[str, list[str]]:
    by = {s: [] for s in sources}
    for rel in index:
        src = rel.split("/", 1)[0]
        if src in by:
            by[src].append(rel)
    for s in sources:
        by[s].sort()
    return by


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/vla/workspace/cache/bpv5_cleaned_root")
    ap.add_argument("--stats_root", default="/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d")
    ap.add_argument(
        "--sources",
        default="agibot,interndata",
        help="Comma-separated sources to scan (default: agibot,interndata; small probe pool).",
    )
    ap.add_argument("--leaves_per_source", type=int, default=8)
    ap.add_argument(
        "--rank_pool",
        type=int,
        default=32,
        help="Per source: randomly sample at most this many leaves before ranking by episode count (avoid full-tree info.json reads).",
    )
    ap.add_argument("--eps_per_leaf", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out",
        default="/vla/workspace/code/conclusion/bpv5/assets/stage_a_probe_label_feasibility.json",
    )
    args = ap.parse_args()

    D.configure(root=args.data_root, stats_root=args.stats_root, action_layout="action80d", delta_mode="bpv5_80d")
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    rng = np.random.default_rng(args.seed)
    index = D.stats_index()
    leaves = leaves_from_stats_index(index, sources)

    rows = []
    summary = {
        "sources": {},
        "totals": {"scanned": 0, "A": 0, "B": 0, "C": 0, "fail": 0},
        "fail_reasons": defaultdict(int),
        "arm_choice": defaultdict(int),
    }

    reader = D.StateReader()
    for src in sources:
        cands = leaves[src]
        if not cands:
            summary["sources"][src] = {"leaves_available": 0}
            continue
        # Small-scale: do NOT read info.json for every leaf. Sample a rank pool first.
        pick_n = min(args.leaves_per_source, len(cands))
        pool_n = min(max(args.rank_pool, pick_n), len(cands))
        rank_cands = list(rng.choice(cands, size=pool_n, replace=False)) if pool_n else []
        scored = []
        for rel in rank_cands:
            try:
                info = D.info(rel)
                n_ep = int(info.get("total_episodes") or 0)
            except Exception:
                continue
            if n_ep >= 2:
                scored.append((n_ep, rel))
        scored.sort(reverse=True)
        # Prefer larger leaves among the sampled pool; then random among top band.
        top = [r for _, r in scored[: max(pick_n * 2, pick_n)]]
        chosen = list(rng.choice(top, size=min(pick_n, len(top)), replace=False)) if top else []

        src_stat = {
            "leaves_available": len(cands),
            "leaves": len(chosen),
            "scanned": 0,
            "A": 0,
            "B": 0,
            "C": 0,
            "fail": 0,
            "leaf_detail": [],
        }
        for rel in chosen:
            print(f"scan {rel}", flush=True)
            eps_map = D.episodes(rel)
            ep_ids = sorted(eps_map)
            info = D.info(rel)
            robot = info.get("robot_type") or src
            _, _, _, _, a_mask, s_mask, _ = D.norm_vectors(rel)
            n_take = min(args.eps_per_leaf, len(ep_ids))
            # last episodes of the leaf (holdout spirit)
            idxs = ep_ids[-n_take:] if ep_ids else []
            leaf_row = {
                "rel": rel,
                "robot": robot,
                "n_eps_leaf": len(ep_ids),
                "arms_mask": {
                    "left": _arm_ok(s_mask, LEFT),
                    "right": _arm_ok(s_mask, RIGHT),
                },
                "episodes": [],
            }
            for ei in idxs:
                ep = eps_map[ei]
                try:
                    state, _ = reader.episode(rel, ep)
                except Exception as e:
                    leaf_row["episodes"].append({"ep": int(ei), "reason": f"read_fail:{e}"})
                    src_stat["fail"] += 1
                    summary["totals"]["fail"] += 1
                    summary["fail_reasons"]["read_fail"] += 1
                    continue
                lab = labels_for_episode(state, s_mask)
                src_stat["scanned"] += 1
                summary["totals"]["scanned"] += 1
                rec = {
                    "ep": int(ei),
                    "n": lab["n"],
                    "arm": lab["arm"],
                    "reason": lab["reason"],
                    "has_A": lab["A"] is not None,
                    "has_B": lab["B"] is not None,
                    "has_C": lab["C"] is not None,
                    "A_norm_cm": None if lab.get("A_norm_m") is None else round(100 * lab["A_norm_m"], 2),
                    "B_norm_cm": None if lab.get("B_norm_m") is None else round(100 * lab["B_norm_m"], 2),
                    "C_norm_cm": None if lab.get("C_norm_m") is None else round(100 * lab["C_norm_m"], 2),
                    "t_close": lab["t_close"],
                    "t_open": lab["t_open"],
                    "A": lab["A"],
                    "B": lab["B"],
                    "C": lab["C"],
                    "y_ok": {
                        "A": lab["A"] is not None and all(math.isfinite(x) for x in lab["A"]),
                        "B": lab["B"] is not None and all(math.isfinite(x) for x in lab["B"]),
                        "C": lab["C"] is not None and all(math.isfinite(x) for x in lab["C"]),
                    },
                }
                leaf_row["episodes"].append(rec)
                if lab["A"] is not None:
                    src_stat["A"] += 1
                    summary["totals"]["A"] += 1
                if lab["B"] is not None:
                    src_stat["B"] += 1
                    summary["totals"]["B"] += 1
                if lab["C"] is not None:
                    src_stat["C"] += 1
                    summary["totals"]["C"] += 1
                if lab["reason"] != "ok" and lab["A"] is None:
                    src_stat["fail"] += 1
                    summary["totals"]["fail"] += 1
                    summary["fail_reasons"][lab["reason"] or "unknown"] += 1
                if lab["arm"]:
                    summary["arm_choice"][lab["arm"]] += 1
            src_stat["leaf_detail"].append(leaf_row)
            rows.append(leaf_row)
        summary["sources"][src] = {k: v for k, v in src_stat.items() if k != "leaf_detail"}
        summary["sources"][src]["leaf_detail"] = src_stat["leaf_detail"]

    summary["fail_reasons"] = dict(summary["fail_reasons"])
    summary["arm_choice"] = dict(summary["arm_choice"])
    # selection recommendation: leaves with >=2 episodes that have B (and ideally C)
    keep_leaves = []
    for leaf in rows:
        b_eps = [e for e in leaf["episodes"] if e.get("has_B")]
        c_eps = [e for e in leaf["episodes"] if e.get("has_C")]
        a_eps = [e for e in leaf["episodes"] if e.get("has_A")]
        if len(a_eps) >= 2:
            keep_leaves.append(
                {
                    "rel": leaf["rel"],
                    "robot": leaf["robot"],
                    "n_A": len(a_eps),
                    "n_B": len(b_eps),
                    "n_C": len(c_eps),
                    "recommend": "ABC"
                    if len(b_eps) >= 2 and len(c_eps) >= 2
                    else ("AB" if len(b_eps) >= 2 else "A_only"),
                }
            )
    summary["recommended_leaves"] = keep_leaves
    summary["recommended_counts"] = {
        "ABC": sum(1 for x in keep_leaves if x["recommend"] == "ABC"),
        "AB": sum(1 for x in keep_leaves if x["recommend"] == "AB"),
        "A_only": sum(1 for x in keep_leaves if x["recommend"] == "A_only"),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"out": str(out), "totals": summary["totals"], "recommended_counts": summary["recommended_counts"], "arm_choice": summary["arm_choice"], "sources": {k: {kk: vv for kk, vv in v.items() if kk != "leaf_detail"} for k, v in summary["sources"].items()}}, indent=2))


if __name__ == "__main__":
    main()
