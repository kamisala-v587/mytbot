#!/usr/bin/env python3
"""Compute mixed (delta joints / abs gripper·quat·vel) action_80D norm stats for cleaned 80D roots.

Follows code/conclusion/bpv5/80D_action_norm_stats_delta_abs.md and BPV5QFormer.data80d
BPV5_DELTA_DIMS / BPV5_ABS_DIMS. Occupied dims come from each leaf's mapping80 (fallback:
parquet action_mask / state_mask). Inactive dims are not updated, so heterogeneous layouts
can be merged safely when desired.

Examples:
  # AgiBot (single robot_type → one stats file)
  python tools/bpv5/compute_80d_mixed_norm_stats.py \\
    --root /physical-ai-data/cleaned_dataset/agibotworld/full-20261106-1943 \\
    --output-dir norm_stats_bpv5_pt/agibotworld_full_20261106_mixed_80d

  # InternData (group by robot_type)
  python tools/bpv5/compute_80d_mixed_norm_stats.py \\
    --root /physical-ai-data/cleaned_dataset/full-InternData-A1-v3.0/full-20261114-2126 \\
    --output-dir norm_stats_bpv5_pt/interndata_a1_v3_full_20261114_mixed_80d \\
    --group-by robot_type
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

TOOLS = Path(__file__).resolve().parents[1]
REPO = TOOLS.parent
sys.path[:0] = [str(TOOLS), str(REPO / "src")]

from lerobot.policies.BPV5QFormer.data80d import (  # noqa: E402
    ACTION_DIM,
    BPV5_ABS_DIMS,
    BPV5_DELTA_DIMS,
)
from norm_stats.core import RunningStats  # noqa: E402

DELTA_DIMS = np.asarray(BPV5_DELTA_DIMS, dtype=np.int64)
ABS_DIMS = np.asarray(BPV5_ABS_DIMS, dtype=np.int64)
DELTA_SET = set(int(x) for x in DELTA_DIMS)
ABS_SET = set(int(x) for x in ABS_DIMS)
KNOWN_SET = DELTA_SET | ABS_SET
CHUNK_DEFAULT = 50


def _slug(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return s.strip("_") or "group"


def discover_leaves(root: Path) -> list[dict[str, Any]]:
    """Prefer cleaning-result.json leaves; else scan for meta/info.json."""
    cr = root / "cleaning-result.json"
    out: list[dict[str, Any]] = []
    if cr.is_file():
        payload = json.loads(cr.read_text())
        for item in payload.get("leaves") or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("status") or "CONVERTED").upper() not in {"CONVERTED", "OK", "SUCCESS"}:
                continue
            leaf = item.get("leaf") or item.get("output")
            if not leaf:
                continue
            out.append({"leaf": str(leaf), "episodes": int(item.get("episodes") or 0), "frames": int(item.get("frames") or 0)})
        if out:
            return out
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # skip hidden / pipeline staging dirs (e.g. DROID .data_clean)
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        if "info.json" in filenames and Path(dirpath).name == "meta":
            leaf = str(Path(dirpath).parent.relative_to(root))
            if any(part.startswith(".") for part in Path(leaf).parts):
                continue
            out.append({"leaf": leaf, "episodes": 0, "frames": 0})
            dirnames.clear()
    return out


def mapping_occupied(repo: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (action_occupied[80], state_occupied[80]) bool masks from mapping80."""
    path = repo / "meta" / "mapping80.json"
    a = np.zeros(ACTION_DIM, dtype=bool)
    s = np.zeros(ACTION_DIM, dtype=bool)
    if not path.is_file():
        return a, s
    mapping = json.loads(path.read_text())
    for key, mask in (("action", a), ("state", s)):
        for entry in mapping.get(key) or []:
            dest = entry.get("destination")
            if not dest or len(dest) != 2:
                continue
            lo, hi = int(dest[0]), int(dest[1])
            mask[lo:hi] = True
    return a, s


def leaf_meta(repo: Path) -> dict[str, Any]:
    info = json.loads((repo / "meta" / "info.json").read_text())
    a_occ, s_occ = mapping_occupied(repo)
    if not a_occ.any():
        # fallback: first parquet action_mask row
        files = list_parquets(repo)
        if files:
            table = pq.read_table(files[0], columns=["action_mask", "state_mask"])
            a_occ = np.asarray(table.column("action_mask")[0].as_py(), dtype=np.float64) > 0.5
            s_occ = np.asarray(table.column("state_mask")[0].as_py(), dtype=np.float64) > 0.5
    return {
        "robot_type": str(info.get("robot_type") or "unknown"),
        "action_occupied": a_occ,
        "state_occupied": s_occ,
        "layout": tuple(sorted(int(i) for i in np.flatnonzero(a_occ))),
    }


def list_parquets(repo: Path) -> list[Path]:
    data = repo / "data"
    if not data.is_dir():
        return []
    files: list[Path] = []
    for dirpath, _, filenames in os.walk(data, followlinks=True):
        for name in filenames:
            if name.endswith(".parquet"):
                files.append(Path(dirpath) / name)
    return sorted(files)


def targets_for_episode(
    action: np.ndarray,
    state: np.ndarray,
    *,
    chunk: int,
    window_stride: int,
    delta_dims: np.ndarray,
) -> np.ndarray:
    length = len(action)
    if length < chunk:
        return np.zeros((0, ACTION_DIM), dtype=np.float64)
    starts = np.arange(0, length - chunk + 1, window_stride, dtype=np.int64)
    out = np.zeros((len(starts) * chunk, ACTION_DIM), dtype=np.float64)
    row = 0
    for t0 in starts:
        base = state[t0]
        chunk_a = action[t0 : t0 + chunk]
        tgt = chunk_a.copy()
        if delta_dims.size:
            tgt[:, delta_dims] = chunk_a[:, delta_dims] - base[delta_dims]
        out[row : row + chunk] = tgt
        row += chunk
    return out


def _update_episode(
    action: np.ndarray,
    state: np.ndarray,
    *,
    chunk: int,
    window_stride: int,
    delta_dims: np.ndarray,
    active_action: np.ndarray,
    active_state: np.ndarray,
    action_stats: list[RunningStats],
    state_stats: list[RunningStats],
) -> int:
    for d in np.flatnonzero(active_state):
        state_stats[int(d)].update(state[:, int(d)])
    tgt = targets_for_episode(
        action, state, chunk=chunk, window_stride=window_stride, delta_dims=delta_dims
    )
    if not len(tgt):
        return 0
    for d in np.flatnonzero(active_action):
        action_stats[int(d)].update(tgt[:, int(d)])
    return len(tgt) // chunk


def compute_leaf(
    repo: Path,
    *,
    chunk: int,
    window_stride: int,
    max_episodes: int | None,
) -> dict[str, Any]:
    meta = leaf_meta(repo)
    a_occ: np.ndarray = meta["action_occupied"]
    s_occ: np.ndarray = meta["state_occupied"]
    # Only known semantic slots participate; unknown reserved dims stay masked out.
    known = np.zeros(ACTION_DIM, dtype=bool)
    known[list(KNOWN_SET)] = True
    active_action = a_occ & known
    active_state = s_occ & known
    delta_dims = np.asarray([d for d in DELTA_DIMS if active_state[int(d)]], dtype=np.int64)

    action_stats = [RunningStats() for _ in range(ACTION_DIM)]
    state_stats = [RunningStats() for _ in range(ACTION_DIM)]
    files = list_parquets(repo)
    if not files:
        raise FileNotFoundError(f"no parquet under {repo}/data")

    # Episodes may span parquet files; buffer the trailing episode across files.
    pending_ep: int | None = None
    pending_a: list[np.ndarray] = []
    pending_s: list[np.ndarray] = []
    n_ep = 0
    n_frames = 0
    n_win = 0

    def flush_pending() -> None:
        nonlocal n_ep, n_win, pending_ep, pending_a, pending_s
        if pending_ep is None or not pending_a:
            pending_ep, pending_a, pending_s = None, [], []
            return
        a = np.concatenate(pending_a, axis=0)
        s = np.concatenate(pending_s, axis=0)
        n_win += _update_episode(
            a,
            s,
            chunk=chunk,
            window_stride=window_stride,
            delta_dims=delta_dims,
            active_action=active_action,
            active_state=active_state,
            action_stats=action_stats,
            state_stats=state_stats,
        )
        n_ep += 1
        pending_ep, pending_a, pending_s = None, [], []

    for path in files:
        if max_episodes is not None and n_ep >= max_episodes:
            break
        table = pq.read_table(path, columns=["episode_index", "action_80D", "state_80D"])
        ep = np.asarray(table.column("episode_index").to_numpy())
        act = np.stack(table.column("action_80D").to_numpy(zero_copy_only=False)).astype(np.float64)
        st = np.stack(table.column("state_80D").to_numpy(zero_copy_only=False)).astype(np.float64)
        n_frames += int(len(ep))
        if len(ep) == 0:
            continue
        # contiguous runs of the same episode_index
        cuts = np.flatnonzero(ep[1:] != ep[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [len(ep)]))
        for lo, hi in zip(starts.tolist(), ends.tolist()):
            e = int(ep[lo])
            a_seg, s_seg = act[lo:hi], st[lo:hi]
            if pending_ep is None:
                pending_ep = e
                pending_a, pending_s = [a_seg], [s_seg]
                continue
            if e == pending_ep:
                pending_a.append(a_seg)
                pending_s.append(s_seg)
                continue
            flush_pending()
            if max_episodes is not None and n_ep >= max_episodes:
                break
            pending_ep = e
            pending_a, pending_s = [a_seg], [s_seg]
        if max_episodes is not None and n_ep >= max_episodes:
            pending_ep, pending_a, pending_s = None, [], []
            break
    flush_pending()

    def pack(stats_list: list[RunningStats]) -> dict[str, Any]:
        mean = np.zeros(ACTION_DIM, dtype=np.float64)
        mean_sq = np.zeros(ACTION_DIM, dtype=np.float64)
        vmin = np.zeros(ACTION_DIM, dtype=np.float64)
        vmax = np.zeros(ACTION_DIM, dtype=np.float64)
        count = np.zeros(ACTION_DIM, dtype=np.int64)
        for d, st_d in enumerate(stats_list):
            if st_d.count == 0:
                continue
            mean[d] = float(st_d.mean[0])
            mean_sq[d] = float(st_d.mean_sq[0])
            vmin[d] = float(st_d.min[0])
            vmax[d] = float(st_d.max[0])
            count[d] = int(st_d.count)
        return {
            "count": count.tolist(),
            "mean": mean.tolist(),
            "mean_sq": mean_sq.tolist(),
            "min": vmin.tolist(),
            "max": vmax.tolist(),
        }

    return {
        "repo": str(repo),
        "robot_type": meta["robot_type"],
        "layout": meta["layout"],
        "episodes": n_ep,
        "frames": n_frames,
        "delta_windows": n_win,
        "action_occupied": active_action.astype(np.float64).tolist(),
        "state_occupied": active_state.astype(np.float64).tolist(),
        "action_payload": pack(action_stats),
        "state_payload": pack(state_stats),
    }


def _merge_dim_payloads(parts: list[dict[str, Any]], key: str) -> dict[str, Any]:
    mean = np.zeros(ACTION_DIM, dtype=np.float64)
    mean_sq = np.zeros(ACTION_DIM, dtype=np.float64)
    vmin = np.full(ACTION_DIM, np.inf, dtype=np.float64)
    vmax = np.full(ACTION_DIM, -np.inf, dtype=np.float64)
    count = np.zeros(ACTION_DIM, dtype=np.float64)
    for part in parts:
        payload = part[key]
        c = np.asarray(payload["count"], dtype=np.float64)
        m = np.asarray(payload["mean"], dtype=np.float64)
        ms = np.asarray(payload["mean_sq"], dtype=np.float64)
        mn = np.asarray(payload["min"], dtype=np.float64)
        mx = np.asarray(payload["max"], dtype=np.float64)
        total = count + c
        # weighted merge where either side has samples
        both = (count > 0) & (c > 0)
        only_new = (count == 0) & (c > 0)
        mean[only_new] = m[only_new]
        mean_sq[only_new] = ms[only_new]
        mean[both] = (mean[both] * count[both] + m[both] * c[both]) / total[both]
        mean_sq[both] = (mean_sq[both] * count[both] + ms[both] * c[both]) / total[both]
        vmin = np.where(c > 0, np.minimum(vmin, mn), vmin)
        vmax = np.where(c > 0, np.maximum(vmax, mx), vmax)
        count = total
    # dims never seen → zeros
    never = count == 0
    vmin[never] = 0.0
    vmax[never] = 0.0
    std = np.sqrt(np.maximum(mean_sq - np.square(mean), 0.0))
    # consumer schema uses a single count list entry (max over dims)
    return {
        "min": vmin.tolist(),
        "max": vmax.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [int(count.max()) if count.size else 0],
        "count_per_dim": count.astype(np.int64).tolist(),
    }


def build_group_output(
    parts: list[dict[str, Any]],
    *,
    group_name: str,
    robot_type: str,
    chunk: int,
    window_stride: int,
) -> dict[str, Any]:
    action = _merge_dim_payloads(parts, "action_payload")
    state = _merge_dim_payloads(parts, "state_payload")
    a_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    s_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    for part in parts:
        a_mask = np.maximum(a_mask, np.asarray(part["action_occupied"], dtype=np.float64))
        s_mask = np.maximum(s_mask, np.asarray(part["state_occupied"], dtype=np.float64))
    delta_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    abs_mask = np.zeros(ACTION_DIM, dtype=np.float64)
    for d in DELTA_DIMS:
        if s_mask[int(d)] > 0.5:
            delta_mask[int(d)] = 1.0
    for d in ABS_DIMS:
        if a_mask[int(d)] > 0.5:
            abs_mask[int(d)] = 1.0
    return {
        "action_80D": {k: action[k] for k in ("min", "max", "mean", "std", "count")},
        "state_80D": {k: state[k] for k in ("min", "max", "mean", "std", "count")},
        "metadata": {
            "robot_type": robot_type,
            "group_name": group_name,
            "action_mode": "mixed",
            "chunk_size": chunk,
            "window_stride": window_stride,
            "action_definition": (
                f"delta: action[t+k]-state[t] on BPV5_DELTA_DIMS {list(map(int, DELTA_DIMS))}; "
                f"abs: action[t+k] on BPV5_ABS_DIMS {list(map(int, ABS_DIMS))}"
            ),
            "delta_mask": delta_mask.tolist(),
            "abs_mask": abs_mask.tolist(),
            "action_mask": a_mask.tolist(),
            "state_mask": s_mask.tolist(),
            "count_per_dim_action": action["count_per_dim"],
            "count_per_dim_state": state["count_per_dim"],
            "datasets": [p["repo"] for p in parts],
            "episodes": int(sum(p["episodes"] for p in parts)),
            "frames": int(sum(p["frames"] for p in parts)),
            "delta_windows": int(sum(p["delta_windows"] for p in parts)),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="cleaned dataset root (contains cleaning-result.json)")
    ap.add_argument("--output-dir", type=Path, required=True, help="directory to write <group>/stats.json")
    ap.add_argument("--group-by", choices=("robot_type", "layout", "all"), default="robot_type")
    ap.add_argument("--chunk-size", type=int, default=CHUNK_DEFAULT)
    ap.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="sliding-window stride (default: chunk-size = non-overlapping)",
    )
    ap.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    ap.add_argument("--max-episodes-per-leaf", type=int, default=None)
    ap.add_argument("--leaf-limit", type=int, default=None, help="debug: only first N leaves")
    ap.add_argument("--repo-id-file", type=Path, default=None, help="optional explicit leaf list (one relative path per line)")
    args = ap.parse_args()

    root = args.root.resolve()
    window_stride = int(args.window_stride or args.chunk_size)
    if window_stride < 1:
        raise SystemExit("--window-stride must be >= 1")

    if args.repo_id_file:
        leaf_ids = [ln.strip() for ln in args.repo_id_file.read_text().splitlines() if ln.strip()]
        leaves = [{"leaf": x, "episodes": 0, "frames": 0} for x in leaf_ids]
    else:
        leaves = discover_leaves(root)
    if args.leaf_limit is not None:
        leaves = leaves[: args.leaf_limit]
    if not leaves:
        raise SystemExit(f"no leaves under {root}")

    # annotate robot_type / layout
    annotated: list[dict[str, Any]] = []
    for item in leaves:
        repo = root / item["leaf"]
        if not (repo / "meta" / "info.json").is_file():
            print(f"[skip] missing info.json: {item['leaf']}", flush=True)
            continue
        meta = leaf_meta(repo)
        annotated.append({**item, **meta, "repo": str(repo)})
    print(f"leaves={len(annotated)} group_by={args.group_by} workers={args.workers} stride={window_stride}", flush=True)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotated:
        if args.group_by == "all":
            key = "all"
        elif args.group_by == "layout":
            key = "layout_" + _slug("-".join(map(str, item["layout"][:8])) + f"_n{len(item['layout'])}")
        else:
            key = _slug(item["robot_type"])
        groups[key].append(item)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"root": str(root), "groups": {}}

    for gname, items in sorted(groups.items()):
        print(f"\n=== group {gname}  leaves={len(items)} ===", flush=True)
        jobs = [
            (it["repo"], args.chunk_size, window_stride, args.max_episodes_per_leaf)
            for it in items
        ]
        parts: list[dict[str, Any]] = []
        if args.workers <= 1:
            for i, job in enumerate(jobs, 1):
                print(f"[{i}/{len(jobs)}] {items[i - 1]['leaf']}", flush=True)
                part = compute_leaf(
                    Path(job[0]),
                    chunk=job[1],
                    window_stride=job[2],
                    max_episodes=job[3],
                )
                part["leaf"] = items[i - 1]["leaf"]
                parts.append(part)
        else:
            done = 0
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                fut_map = {pool.submit(compute_leaf, Path(job[0]), chunk=job[1], window_stride=job[2], max_episodes=job[3]): idx for idx, job in enumerate(jobs)}
                for fut in as_completed(fut_map):
                    idx = fut_map[fut]
                    part = fut.result()
                    part["leaf"] = items[idx]["leaf"]
                    parts.append(part)
                    done += 1
                    if done % 10 == 0 or done == len(jobs):
                        print(f"[{done}/{len(jobs)}] last={items[idx]['leaf']} eps={part['episodes']} win={part['delta_windows']}", flush=True)

        robot_types = sorted({p["robot_type"] for p in parts})
        robot_type = robot_types[0] if len(robot_types) == 1 else "mixed"
        out = build_group_output(
            parts,
            group_name=gname,
            robot_type=robot_type,
            chunk=args.chunk_size,
            window_stride=window_stride,
        )
        out_path = args.output_dir / gname / "stats.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print("wrote", out_path, flush=True)
        a = out["action_80D"]
        active = [i for i, v in enumerate(out["metadata"]["action_mask"]) if v > 0.5]
        print("active action mean/std:")
        for d in active:
            kind = "delta" if out["metadata"]["delta_mask"][d] > 0.5 else "abs"
            print(f"  d{d:02d} {kind:5} mean={a['mean'][d]: .4f} std={a['std'][d]:.4f}", flush=True)
        summary["groups"][gname] = {
            "stats": str(out_path),
            "robot_type": robot_type,
            "leaves": len(parts),
            "episodes": out["metadata"]["episodes"],
            "frames": out["metadata"]["frames"],
            "delta_windows": out["metadata"]["delta_windows"],
            "active_dims": active,
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("wrote", summary_path, flush=True)


if __name__ == "__main__":
    main()
