"""Main-path Stage B pairing index (`bpv4cdx Stage B 开发文档 v1.md` 3.1 / 3.2).

Buckets = (repo, task description string) over a split manifest (AE-3 format). A bucket enters Stage B when it has
more than `--min_size - 1` trajectories (default: > 3). matched = another episode of the same bucket and the same
split (the AE-3 held-out episodes form their own pool; demo and query never cross train / held-out).
Arm per episode: from the description text when it names the arm(s) ("left arm" -> L, both arms -> both); for an
arm-agnostic description on a dual-arm 6+1 robot, the arm whose gripper closes first; else "unk".
Cross-arm matched = same bucket, one L and one R episode (only arm-agnostic descriptions can have both).

Output (compact, one entry per bucket):
  {"version", "manifest", "manifest_sha256", "roots", "min_size", "repos": [{rel, robot_type, dual_arm_61}],
   "buckets": [{"repo": repo index, "text", "train": [ep], "heldout": [ep], "length": {ep: L}, "arm": {ep: L/R/unk}}],
   "summary": {...}}

usage: python -m lerobot.policies.BPV4CDXB.pairs_pretrain --manifest configs/ds_ids/local/et3_split.json \
         --data_roots /public-data/dataset:/vla/workspace/data:/public-data/dataset/agibot-world3.0-fix/beta2025 \
         --array_store /vla/workspace/cache/ae3_arrays_dev --out outputs/BPv4CDX/stage_b/stage_b_pretrain_index.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.policies.BPV4CDX.embodiment import embodiment_from_meta, first_close
from lerobot.policies.BPV4CDX.paths import load_manifest, manifest_repos
from lerobot.policies.BPV4CDX.repo_data import EpView, read_arrays, store_dir

_LEFT = re.compile(r"\bleft\s+(arm|hand|gripper)", re.I)
_RIGHT = re.compile(r"\bright\s+(arm|hand|gripper)", re.I)


def text_arm(text: str) -> str:
    left, right = bool(_LEFT.search(text)), bool(_RIGHT.search(text))
    if left and not right:
        return "L"
    if right and not left:
        return "R"
    return "both" if left and right else "unk"


def episode_tasks(repo: str) -> pd.DataFrame:
    files = sorted(Path(repo, "meta", "episodes").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "tasks", "length"]) for f in files)

    def first(t):
        if isinstance(t, str):
            return t
        t = list(t)
        return str(t[0]) if t else ""

    df["text"] = df["tasks"].map(first)
    return df[["episode_index", "text", "length"]]


def repo_entry(path: str, entry: dict, array_store: str | None, min_size: int) -> tuple[dict, list[dict]]:
    info_json = json.loads(Path(path, "meta", "info.json").read_text())
    robot_type = info_json["robot_type"]
    emb = embodiment_from_meta(robot_type, info_json["features"])
    df = episode_tasks(path)
    held = set(int(e) for e in entry["heldout"])
    states = None
    if emb.dual_arm_61:
        sd = store_dir(array_store, path) if array_store else None
        if sd is not None and (sd / "index.json").is_file():
            index = json.loads((sd / "index.json").read_text())
            states = EpView(np.load(sd / "state.npy", mmap_mode="r"), {int(e): (int(s), int(t)) for e, s, t in index["episodes"]})
        else:
            state, _, spans = read_arrays(path, emb)
            states = EpView(state, spans)
    buckets = []
    for text, g in df.groupby("text", sort=True):
        if len(g) < min_size:
            continue
        eps = sorted(int(e) for e in g["episode_index"])
        arm = {}
        named = text_arm(text)
        for ep in eps:
            a = None
            # a description that names the arm(s) decides; first-close only for arm-agnostic descriptions
            if named == "unk" and states is not None and ep in states:
                fc = first_close(states[ep], emb)
                a = fc[0] if fc else None
            arm[str(ep)] = a or named
        buckets.append({"text": text, "train": [e for e in eps if e not in held], "heldout": [e for e in eps if e in held],
                        "length": {str(int(e)): int(n) for e, n in zip(g["episode_index"], g["length"])}, "arm": arm})
    info = {"rel": entry["rel"], "robot_type": robot_type, "dual_arm_61": bool(emb.dual_arm_61),
            "episodes": int(len(df)), "texts": int(df["text"].nunique())}
    return info, buckets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--data_roots", required=True)
    ap.add_argument("--array_store", default=None)
    ap.add_argument("--min_size", type=int, default=4, help="keep buckets with at least this many trajectories (> 3)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    roots = args.data_roots.split(":")
    pairs = manifest_repos(load_manifest(args.manifest), roots)
    logging.info("%d repos", len(pairs))

    def one(pe):
        try:
            return repo_entry(pe[0], pe[1], args.array_store, args.min_size)
        except Exception as e:  # noqa: BLE001 - report and skip one broken repo
            logging.warning("skip %s: %s: %s", pe[1]["rel"], type(e).__name__, e)
            return None

    with ThreadPoolExecutor(args.threads) as pool:
        results = list(pool.map(one, pairs))
    repos, buckets, skipped = [], [], []
    dropped_eps = 0
    for (path, entry), res in zip(pairs, results):
        if res is None:
            skipped.append(entry["rel"])
            continue
        info, bs = res
        kept = sum(len(b["train"]) + len(b["heldout"]) for b in bs)
        dropped_eps += info["episodes"] - kept
        if not bs:
            continue
        ri = len(repos)
        repos.append(info)
        for b in bs:
            buckets.append({"repo": ri, **b})

    n_train = sum(len(b["train"]) for b in buckets)
    n_held = sum(len(b["heldout"]) for b in buckets)
    arms = Counter(a for b in buckets for a in b["arm"].values())
    cross = sum(1 for b in buckets if len({a for a in b["arm"].values() if a in ("L", "R")}) == 2)
    held_pool = sum(1 for b in buckets if len(b["heldout"]) >= 2)
    sizes = np.array([len(b["train"]) + len(b["heldout"]) for b in buckets])
    summary = {"repos_in": len(pairs), "repos_kept": len(repos), "repos_skipped": skipped, "buckets": len(buckets),
               "episodes_train": n_train, "episodes_heldout": n_held, "episodes_dropped_small_bucket": dropped_eps,
               "bucket_size": {"min": int(sizes.min()), "median": int(np.median(sizes)), "max": int(sizes.max())},
               "arm_labels": dict(arms), "buckets_with_cross_arm": cross, "buckets_heldout_ge2": held_pool,
               "robot_types": dict(Counter(r["robot_type"] for r in repos))}
    out = {"version": 1, "manifest": str(Path(args.manifest).resolve()),
           "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(), "roots": roots,
           "min_size": args.min_size, "repos": repos, "buckets": buckets, "summary": summary}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
