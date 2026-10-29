"""Stage B main-path pairing index (StageB实施.md, S1): demo = another episode with the same robot type and the
identical task description text (across repos), never across the train / held-out split.

  bucket key   robot_type + task text (meta/tasks.parquet + meta/episodes `tasks`); junk task texts of <= 2
               characters (genie1 pick_and_place 'P') are relabelled to the repo's only real text
  split        AE-3 split manifest (held-out episodes); train pool = train episodes of buckets with > --min_bucket
               train episodes; val pool = held-out episodes of buckets with >= 2 held-out episodes
  arm          'left' / 'right' when the text names one arm, else null (then demo and query may use different arms)

Output json: {"buckets": [{"key", "robot_type", "text", "arm", "train": [eid], "val": [eid]}],
              "episodes": {eid: [rel, episode, length, bucket index, split]}}, eid = "<rel>|<episode>".

usage: python -m lerobot.policies.BPV4CDXB.pairs_main --repo_file configs/et3.txt \
         --manifest configs/ds_ids/B200/ae3_split.json --zdemo_dir <cache>/bpv4cdxb_zdemo/<tag> --out <index.json>
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from lerobot.policies.BPV4CDX.paths import rel_id

ARM = re.compile(r"\b(left|right)\b", re.I)


def read_repo(repo: str):
    info = json.loads(Path(repo, "meta/info.json").read_text())
    tasks = pd.read_parquet(Path(repo, "meta/tasks.parquet"))
    real = [t for t in tasks.index if len(str(t).strip()) > 2]
    files = sorted(Path(repo, "meta/episodes").rglob("*.parquet"))
    ep = pd.concat(pd.read_parquet(f, columns=["episode_index", "tasks", "length"]) for f in files)
    rows, relabelled, dropped = [], 0, 0
    for e, ts, n in zip(ep["episode_index"], ep["tasks"], ep["length"]):
        ts = [str(t) for t in ts]
        good = [t for t in ts if len(t.strip()) > 2]
        if not good:
            if len(real) == 1:
                good, relabelled = real, relabelled + 1
            else:
                dropped += 1
                continue
        rows.append((int(e), " || ".join(good), int(n)))
    return rel_id(repo), info.get("robot_type"), rows, relabelled, dropped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--zdemo_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_bucket", type=int, default=3, help="train pool: buckets with more than this many episodes")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--check", type=int, default=2000, help="random (query, demo) draws checked after writing")
    args = ap.parse_args()
    repos = [l.strip().rstrip("/") for l in Path(args.repo_file).read_text().splitlines() if l.strip()]
    held = {e["rel"]: set(e["heldout"]) for e in json.loads(Path(args.manifest).read_text())["repos"]}
    with ProcessPoolExecutor(args.workers) as pool:
        res = list(pool.map(read_repo, repos, chunksize=4))
    missing = [rel for rel, *_ in res if rel not in held]
    if missing:
        raise SystemExit(f"repos not in the split manifest: {missing[:3]}")
    groups = collections.defaultdict(lambda: {"train": [], "val": []})
    episodes, relabelled, dropped = {}, 0, 0
    for rel, robot, rows, rl, dr in res:
        relabelled, dropped = relabelled + rl, dropped + dr
        for ep, text, n in rows:
            split = "val" if ep in held[rel] else "train"
            key = f"{robot}||{text}"
            eid = f"{rel}|{ep}"
            groups[key][split].append(eid)
            episodes[eid] = [rel, ep, n, key, split]
    buckets, kidx = [], {}
    for key in sorted(groups):
        g = groups[key]
        tr = g["train"] if len(g["train"]) > args.min_bucket else []
        va = g["val"] if len(g["val"]) >= 2 else []
        if not tr and not va:
            continue
        robot, text = key.split("||", 1)
        m = ARM.findall(text)
        arm = m[0].lower() if m and len({a.lower() for a in m}) == 1 else None
        kidx[key] = len(buckets)
        buckets.append({"key": key, "robot_type": robot, "text": text, "arm": arm, "train": tr, "val": va})
    kept = {}
    for b in buckets:
        for s in ("train", "val"):
            for eid in b[s]:
                rel, ep, n, key, split = episodes[eid]
                kept[eid] = [rel, ep, n, kidx[key], split]
    index = {"version": 1, "repo_file": str(args.repo_file), "manifest": str(args.manifest),
             "zdemo_dir": str(args.zdemo_dir), "min_bucket": args.min_bucket,
             "rule": "demo: same robot_type + identical task text, other episode, same split",
             "buckets": buckets, "episodes": kept}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(index))

    # summary
    ntr = sum(len(b["train"]) for b in buckets)
    nva = sum(len(b["val"]) for b in buckets)
    sizes = sorted((len(b["train"]) for b in buckets if b["train"]), reverse=True)
    print(f"repos {len(repos)}, episodes read {len(episodes)}, relabelled junk text {relabelled}, dropped {dropped}")
    print(f"buckets kept {len(buckets)} (train pools {len(sizes)}, val pools {sum(bool(b['val']) for b in buckets)})")
    print(f"train queries {ntr}, val queries {nva}, not used {len(episodes) - ntr - nva}")
    print(f"train bucket size max {sizes[0]}, median {sizes[len(sizes) // 2]}, min {sizes[-1]}")
    arm_tr = sum(len(b["train"]) for b in buckets if b["arm"])
    print(f"train episodes in buckets whose text names one arm: {arm_tr} ({100 * arm_tr / max(1, ntr):.1f}%); "
          f"robot types: {dict(collections.Counter(b['robot_type'] for b in buckets))}")

    # checks: same robot + text, other episode, same split, bucket size, Z_demo file present
    rng, zroot, bad = random.Random(0), Path(args.zdemo_dir), []
    eids = list(kept)
    for _ in range(args.check):
        q = rng.choice(eids)
        rel, ep, n, bi, split = kept[q]
        pool = buckets[bi][split]
        d = rng.choice([e for e in pool if e != q])
        drel, dep, _, dbi, dsplit = kept[d]
        ok = (dbi == bi and dsplit == split and d != q
              and (len(pool) > args.min_bucket if split == "train" else len(pool) >= 2)
              and (zroot / drel / f"ep{dep:05d}.pt").is_file())
        if not ok:
            bad.append((q, d))
    texts_ok = all(len({kept[e][3] for e in b["train"] + b["val"]}) == 1 for b in buckets)
    split_ok = all(kept[e][4] == "train" for b in buckets for e in b["train"]) and \
        all(kept[e][4] == "val" for b in buckets for e in b["val"])
    print(f"checks: {args.check} random pairs, {len(bad)} bad {bad[:2]}; one key per bucket {texts_ok}; "
          f"no cross-split pools {split_ok}")
    print(args.out)


if __name__ == "__main__":
    main()
