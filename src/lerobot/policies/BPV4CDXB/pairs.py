"""Stage B pairing index from a bp_tasks LeRobot export (development doc 3.2 / 3.3, B-0).

Reads `<root>/candidate_sets.json` and every repo's `meta/bp_candidate_set.json` and writes one json with
  episodes        every trajectory: candidate set, task, goal, anchor, arm, layout class, instance, jitter, layout, length
  matched         per query episode: demos of the same candidate set whose objects moved >= --min_shift_cm (max over objects)
  contrast        per anchor: its candidate sets (same scene, different goals)
  equivalent      per candidate set: sets with the same task, instance, goal and layout class at another anchor
  split           held-out anchors per task (fixed by sha1 of the anchor id); demos and queries never cross the split
  divergence      per anchor: t_div = first frame where the jitter-0 trajectories of any two goals differ by more than
                  --div_threshold (joint L2, rad; also recorded for 0.05 / 0.2 / 0.5); the jitter-0 initial scenes of
                  all goals must be identical. Per episode also t_grasp (first gripper close)
  checks          B-0 pairing checks (see b0 section of the development doc)

usage: python -m lerobot.policies.BPV4CDXB.pairs --root /vla/workspace/my_robotwin/data_bp/p1_n10_lerobot \
         --out outputs/BPv4CDX/stage_b/stage_b_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HELDOUT_ANCHORS = {"place_shoe_bp": 3, "move_stapler_pad_bp": 2, "place_a2b_bp": 2, "place_mouse_pad_bp": 1,
                   "place_fan_bp": 1}
JOINTS = [*range(6), *range(7, 13)]


def episode_states(repo: Path) -> dict[int, np.ndarray]:
    files = sorted((repo / "data").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files)
    df = df.sort_values(["episode_index", "frame_index"])
    return {int(e): np.stack(g["observation.state"].to_numpy()) for e, g in df.groupby("episode_index")}


def xy(layout: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v[:2], dtype=float) for k, v in layout.items()}


def shift_cm(a: dict, b: dict) -> float:
    keys = sorted(set(a) & set(b))
    return float(max(np.linalg.norm(a[k] - b[k]) for k in keys) * 100) if keys else 0.0


def first_divergence(a: np.ndarray, b: np.ndarray, threshold: float) -> int:
    n = min(len(a), len(b))
    d = np.linalg.norm(a[:n, JOINTS] - b[:n, JOINTS], axis=1)
    idx = np.nonzero(d > threshold)[0]
    return int(idx[0]) if len(idx) else n


def first_grasp(s: np.ndarray) -> int | None:
    """First frame where either aloha gripper (dims 6, 13) closes below half of its opening at frame 0."""
    best = None
    for g in (6, 13):
        if s[0, g] <= 0:
            continue
        closed = s[:, g] < 0.5 * s[0, g]
        idx = np.nonzero(closed)[0]
        if len(idx) and (best is None or idx[0] < best):
            best = int(idx[0])
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_shift_cm", type=float, default=2.0)
    # 0.2 rad: shoe / mouse pad / fan stay together until the grasp, a2b and stapler split as soon as they pick a
    # different object or action (0.05 rad flags the planner's small approach differences as a divergence)
    ap.add_argument("--div_threshold", type=float, default=0.2)
    args = ap.parse_args()
    root = Path(args.root)
    sets = json.loads((root / "candidate_sets.json").read_text())

    episodes, by_set, meta_by_set = {}, defaultdict(list), {}
    states_j0 = {}
    for key, s in sets.items():
        repo = root / s["path"]
        meta = json.loads((repo / "meta" / "bp_candidate_set.json").read_text())
        meta_by_set[key] = meta
        states = episode_states(repo)
        for e in meta["episodes"]:
            idx = int(e["index"])
            eid = f"{s['path']}|{idx}"
            episodes[eid] = {
                "repo": s["path"], "episode": idx, "set": key, "task": meta["task"], "goal": meta["goal"],
                "anchor": meta["anchor_id"], "arm": meta["arm"], "layout_class": meta["layout_class"],
                "instance": meta["instance"], "jitter": int(e["factors"]["jitter"]), "length": int(len(states[idx])),
                "layout": {k: list(map(float, v[:3])) for k, v in e["layout"].items()},
                "t_grasp": first_grasp(states[idx]),
                "object_pose_init": e.get("object_pose_init", {}),
            }
            by_set[key].append(eid)
            if int(e["factors"]["jitter"]) == 0:
                states_j0[key] = states[idx]

    matched = {}
    for key, eids in by_set.items():
        for q in eids:
            lq = xy(episodes[q]["layout"])
            demos = [(d, shift_cm(lq, xy(episodes[d]["layout"]))) for d in eids if d != q]
            matched[q] = [{"demo": d, "shift_cm": round(s, 2)} for d, s in demos if s >= args.min_shift_cm]

    contrast = defaultdict(list)
    for key, meta in meta_by_set.items():
        contrast[meta["anchor_id"]].append(key)

    equivalent = {}
    for key, meta in meta_by_set.items():
        sig = (meta["task"], json.dumps(meta["instance"], sort_keys=True), meta["goal"], meta["layout_class"])
        equivalent[key] = sorted(k for k, m in meta_by_set.items() if k != key and m["anchor_id"] != meta["anchor_id"]
                                 and (m["task"], json.dumps(m["instance"], sort_keys=True), m["goal"], m["layout_class"]) == sig)

    anchors_by_task = defaultdict(set)
    for meta in meta_by_set.values():
        anchors_by_task[meta["task"]].add(meta["anchor_id"])
    heldout = set()
    for task, anchors in anchors_by_task.items():
        k = HELDOUT_ANCHORS.get(task, max(1, round(0.17 * len(anchors))))
        heldout |= set(sorted(anchors, key=lambda a: hashlib.sha1(a.encode()).hexdigest())[:k])
    split = {a: ("val" if a in heldout else "train") for a in sorted(set().union(*anchors_by_task.values()))}

    divergence, scene_identical = {}, {}
    for anchor, keys in contrast.items():
        j0 = {k: states_j0[k] for k in keys if k in states_j0}
        pairs = list(itertools.combinations(sorted(j0), 2))
        t = min((first_divergence(j0[a], j0[b], args.div_threshold) for a, b in pairs), default=None)
        lengths = [len(v) for v in j0.values()]
        divergence[anchor] = {"t_div": t, "tau_div": round(t / max(1, min(lengths) - 1), 3) if t is not None else None,
                              "goals": len(keys), "j0_sets": len(j0),
                              "t_div_by_threshold": {str(th): min((first_divergence(j0[a], j0[b], th) for a, b in pairs), default=None)
                                                     for th in (0.05, 0.2, 0.5)}}
        inits = [episodes[e]["layout"] for k in keys for e in by_set[k] if episodes[e]["jitter"] == 0]
        diffs = [max(abs(x - y) for o in a for x, y in zip(a[o], b.get(o, a[o]))) for a, b in itertools.combinations(inits, 2)]
        scene_identical[anchor] = float(max(diffs, default=0.0))

    # B-0 checks
    n_pairs = sum(len(v) for v in matched.values())
    cross = sum(1 for q, ds in matched.items() for d in ds
                if split[episodes[q]["anchor"]] != split[episodes[d["demo"]]["anchor"]]
                or episodes[q]["set"] != episodes[d["demo"]]["set"] or q == d["demo"])
    arm_mismatch = sum(1 for q, ds in matched.items() for d in ds if episodes[q]["arm"] != episodes[d["demo"]]["arm"])
    no_demo = sorted(q for q, ds in matched.items() if not ds)
    total_frames = sum(e["length"] for e in episodes.values())
    pre_div = sum(min(e["length"], divergence[e["anchor"]]["t_div"] or 0) for e in episodes.values())
    checks = {
        "episodes": len(episodes), "candidate_sets": len(sets), "anchors": len(split),
        "heldout_anchors": sorted(heldout), "heldout_episodes": sum(1 for e in episodes.values() if split[e["anchor"]] == "val"),
        "matched_pairs": n_pairs, "pairs_crossing_set_or_split_or_self": cross, "pairs_arm_mismatch": arm_mismatch,
        "queries_without_demo": no_demo, "contrast_groups": len(contrast),
        "contrast_groups_single_goal": sorted(a for a, k in contrast.items() if len(k) < 2),
        "j0_scene_max_abs_diff_m": max(scene_identical.values(), default=0.0),
        "anchors_j0_scene_not_identical": sorted(a for a, v in scene_identical.items() if v > 1e-6),
        "frames_before_t_div_fraction": round(pre_div / max(1, total_frames), 3),
        "shift_cm": {"min": float(min(d["shift_cm"] for ds in matched.values() for d in ds)),
                     "median": float(np.median([d["shift_cm"] for ds in matched.values() for d in ds])),
                     "max": float(max(d["shift_cm"] for ds in matched.values() for d in ds))},
    }
    checks["pass"] = (cross == 0 and arm_mismatch == 0 and not no_demo and not checks["contrast_groups_single_goal"]
                      and not checks["anchors_j0_scene_not_identical"])
    out = {"root": str(root), "min_shift_cm": args.min_shift_cm, "div_threshold": args.div_threshold,
           "episodes": episodes, "sets": {k: by_set[k] for k in sorted(by_set)}, "matched": matched,
           "contrast": dict(contrast), "equivalent": equivalent, "split": split, "divergence": divergence,
           "checks": checks}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(checks, indent=1, ensure_ascii=False))
    td = [v["tau_div"] for v in divergence.values() if v["tau_div"] is not None]
    by_task = defaultdict(list)
    for a, v in divergence.items():
        by_task[a.split("/")[0]].append(v["t_div"])
    print("t_div (frames) by task:", {t: (int(np.min(v)), int(np.median(v)), int(np.max(v))) for t, v in by_task.items()})
    print(f"tau_div median {np.median(td):.2f}")
    for th in ("0.05", "0.2", "0.5"):
        fr = sum(min(e["length"], divergence[e["anchor"]]["t_div_by_threshold"][th] or 0) for e in episodes.values())
        per = defaultdict(list)
        for a, v in divergence.items():
            per[a.split("/")[0]].append(v["t_div_by_threshold"][th])
        print(f"threshold {th} rad: frames before divergence {fr / total_frames:.3f};",
              {t: int(np.median(v)) for t, v in per.items()})
    tg = [e["t_grasp"] for e in episodes.values() if e["t_grasp"] is not None]
    pre_grasp = sum(min(e["length"], e["t_grasp"]) for e in episodes.values() if e["t_grasp"] is not None)
    per = defaultdict(list)
    for e in episodes.values():
        if e["t_grasp"] is not None:
            per[e["task"]].append(e["t_grasp"] / max(1, e["length"] - 1))
    print(f"first grasp: {len(tg)} / {len(episodes)} episodes; frames before grasp {pre_grasp / total_frames:.3f};",
          "median tau by task", {t: round(float(np.median(v)), 2) for t, v in per.items()})


if __name__ == "__main__":
    main()
