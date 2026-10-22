"""AE-3 data tools (bpv4cdx AE-3 plan, sections 1 - 4). Sub-commands:

  manifest        repo list -> split manifest (relative ids, embodiment, held-out episodes) + training repo list;
                  egodex (and --exclude patterns) removed; repos whose embodiment cannot be resolved are excluded
                  with the reason
  prepare-arrays  per-repo memory-mapped state / action arrays (<store>/<rel>/state.npy, action.npy, index.json),
                  process pool over repos
  audit           G0 for AE-3 from the states only (no GPU): per source the episode lengths, gripper events, and for
                  S-U and S-UF the gripper-event hit rate (+-1 frame) and the joint interpolation RMS; prints the
                  sampler decision of the plan (S-U if its hit rate >= 0.8 over all audited episodes, else S-UF)
  cache-index     list `<rel>|<episode>` of the cache files that exist (dataset.cache_index)
  progress        cache build progress per source: done / expected, failures, rate since the previous call, ETA

Usage (B200):
  python -m lerobot.policies.BPV4CDX.ae3_data manifest --repo_file configs/ds_ids/B200/pretrain-data.txt \
      --out configs/ds_ids/B200/ae3_split.json --train_list configs/ds_ids/B200/ae3_train.txt
  python -m lerobot.policies.BPV4CDX.ae3_data prepare-arrays --manifest ... --data_roots /share --store <dir> --workers 64
  python -m lerobot.policies.BPV4CDX.ae3_data audit --manifest ... --data_roots /share --array_store <dir> --out <dir>
  python -m lerobot.policies.BPV4CDX.ae3_data progress --manifest ... --data_roots /share --cache_dir <dir>
  python -m lerobot.policies.BPV4CDX.ae3_data cache-index --manifest ... --data_roots /share --cache_dir <dir> --out <txt>
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .paths import ae3_heldout, load_manifest, manifest_repos, read_repo_list, rel_id

EXCLUDE_DEFAULT = ("egodex",)


def source_of(rel: str) -> str:
    """Reporting group: top-level dataset; InternData split by robot (very different sizes)."""
    parts = rel.split("/")
    if parts[0].startswith("full-InternData") and len(parts) > 2:
        return f"InternData/{parts[2]}"
    return parts[0]


# --- manifest ------------------------------------------------------------------------------------------------------
def _manifest_entry(repo: str) -> dict:
    from .embodiment import embodiment_from_meta

    rel = rel_id(repo)
    try:
        info = json.loads(Path(repo, "meta", "info.json").read_text())
        emb = embodiment_from_meta(info["robot_type"], info["features"])
        total = int(info["total_episodes"])
        return {"rel": rel, "robot_type": info["robot_type"], "resolved": emb.resolved, "total_episodes": total,
                "total_frames": int(info.get("total_frames", 0)), "fps": int(info.get("fps", 30)),
                "heldout": ae3_heldout(rel, total), "cameras": emb.cameras, "state_dim": int(sum(emb.state_sizes)),
                "action_dim": int(sum(emb.action_sizes)), "gripper_dims": emb.gripper_dims.tolist(),
                "dual_arm_61": bool(emb.dual_arm_61)}
    except Exception as e:  # noqa: BLE001
        return {"rel": rel, "error": f"{type(e).__name__}: {e}"}


def cmd_manifest(args) -> None:
    repos = read_repo_list(args.repo_file, args.data_root)
    excl = tuple(EXCLUDE_DEFAULT) + tuple(x for x in args.exclude.split(",") if x)
    kept = [r for r in repos if not any(x in r for x in excl)]
    with ThreadPoolExecutor(args.workers) as pool:
        entries = list(pool.map(_manifest_entry, kept))
    ok = [e for e in entries if "error" not in e]
    bad = [e for e in entries if "error" in e]
    rels = [e["rel"] for e in ok]
    if len(set(rels)) != len(rels):
        raise SystemExit("duplicate relative ids in the repo list")
    manifest = {
        "version": 1, "generated_on": socket.gethostname(), "generated_at": time.strftime("%Y-%m-%d %H:%M"),
        "repo_file": str(args.repo_file),
        "rule": "RoboTwin clean_50: last 5; RoboTwin randomized_*: episodes 90-99; other repos: last "
                "min(10, max(2, round(1% of the episodes)))",
        "excluded_patterns": list(excl), "excluded_by_pattern": len(repos) - len(kept),
        "excluded_errors": bad, "repos": ok,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(manifest, indent=1))
    if args.train_list:
        Path(args.train_list).write_text("\n".join(r for r, e in zip(kept, entries) if "error" not in e) + "\n")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for e in ok:
        b = by[source_of(e["rel"])]
        b[0] += 1
        b[1] += e["total_episodes"]
        b[2] += len(e["heldout"])
        b[3] += e["total_frames"] / max(1, e["fps"]) / 3600
    print(f"{'source':34s} {'repos':>6s} {'episodes':>9s} {'held-out':>9s} {'hours':>8s}")
    for s, (n, eps, held, h) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"{s:34s} {n:6d} {eps:9d} {held:9d} {h:8.1f}")
    print(f"kept {len(ok)} repos, excluded {len(repos) - len(kept)} by pattern, {len(bad)} by error -> {args.out}")
    for e in bad[:20]:
        print("  error:", e["rel"], e["error"])


# --- arrays ----------------------------------------------------------------------------------------------------------
def _prepare_one(job) -> tuple[str, str]:
    repo, store, force = job
    from .repo_data import store_dir, write_store

    if not force and (store_dir(store, repo) / "index.json").is_file():
        return repo, "exists"
    try:
        write_store(repo, store)
        return repo, "ok"
    except Exception as e:  # noqa: BLE001
        return repo, f"error {type(e).__name__}: {e}"


def cmd_prepare_arrays(args) -> None:
    pairs = manifest_repos(load_manifest(args.manifest), args.data_roots.split(":"), _sources(args))
    jobs = [(p, args.store, args.force) for p, _ in pairs]
    t0, n_ok, n_err = time.time(), 0, 0
    with ProcessPoolExecutor(args.workers) as pool:
        for i, (repo, status) in enumerate(pool.map(_prepare_one, jobs, chunksize=1), 1):
            n_ok += status in ("ok", "exists")
            if status.startswith("error"):
                n_err += 1
                print(rel_id(repo), status, flush=True)
            if i % 50 == 0 or i == len(jobs):
                print(f"{i} / {len(jobs)} repos, {n_err} errors, {time.time() - t0:.0f} s", flush=True)


# --- audit (G0) ------------------------------------------------------------------------------------------------------
SAMPLERS_AUDITED = ("sa", "su", "suf")


def _audit_one(job) -> list[dict]:
    repo, store, n_eps, num_slots, seed = job
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    from .build_cache import repo_states
    from .embodiment import embodiment_from_meta, gripper_events
    from .sampling import interpolation_error, sample

    try:
        meta = LeRobotDatasetMetadata(repo)
        emb = embodiment_from_meta(meta.robot_type, meta.features)
        states = repo_states(repo, emb, store)
    except Exception as e:  # noqa: BLE001
        return [{"rel": rel_id(repo), "error": f"{type(e).__name__}: {e}"}]
    eps = sorted(states.keys())
    rng = np.random.default_rng([seed, len(eps)])
    pick = sorted(rng.choice(eps, size=min(n_eps, len(eps)), replace=False).tolist())
    rows = []
    for ep in pick:
        s = states[ep]
        events = gripper_events(s, emb)
        row = {"rel": rel_id(repo), "episode": int(ep), "length": int(len(s)), "events": len(events),
               "robot": emb.resolved}
        for name in SAMPLERS_AUDITED:
            plan = sample(name, s, num_slots, emb=emb)
            fr = set(int(f) for f in plan.frames)
            lo, hi = plan.active
            ev = [e for e in events if lo <= e <= hi]
            row[f"{name}_hit"] = float(np.mean([any(abs(e - f) <= 1 for f in fr) for e in ev])) if ev else None
            row[f"{name}_rms"] = interpolation_error(s[:, emb.joint_dims], plan.frames) if len(emb.joint_dims) else None
            row[f"{name}_slot_gap"] = float((hi - lo + 1) / num_slots)
            # largest gap between consecutive slots, as a fraction of the active span (1 / 31 for perfectly uniform)
            row[f"{name}_max_gap"] = float(np.diff(np.unique(plan.frames)).max() / max(1, hi - lo)) if len(fr) > 1 else None
            row[f"{name}_repeats"] = int(plan.repeats)
        rows.append(row)
    return rows


def cmd_audit(args) -> None:
    pairs = manifest_repos(load_manifest(args.manifest), args.data_roots.split(":"), _sources(args))
    if args.repos_per_source:
        rng = np.random.default_rng(args.seed)
        groups = defaultdict(list)
        for p in pairs:
            groups[source_of(p[1]["rel"])].append(p)
        pairs = [g[i] for g in groups.values() for i in sorted(rng.choice(len(g), min(len(g), args.repos_per_source), replace=False))]
    jobs = [(p, args.array_store, args.episodes_per_repo, args.num_slots, args.seed) for p, _ in pairs]
    rows, errors = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for i, res in enumerate(pool.map(_audit_one, jobs, chunksize=1), 1):
            for r in res:
                (errors if "error" in r else rows).append(r)
            if i % 100 == 0:
                print(f"{i} / {len(jobs)} repos", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit_episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    def summary(sel):
        def mean(k):
            v = [r[k] for r in sel if r.get(k) is not None]
            return float(np.mean(v)) if v else None
        lengths = np.array([r["length"] for r in sel])
        res = {"episodes": len(sel), "length_mean": float(lengths.mean()), "length_p90": float(np.percentile(lengths, 90)),
               "events_mean": mean("events"), "slot_gap_frames": mean("su_slot_gap")}
        for name in SAMPLERS_AUDITED:
            for k in ("hit", "rms", "max_gap", "repeats"):
                res[f"{name}_{k}"] = mean(f"{name}_{k}")
        return res

    by = defaultdict(list)
    for r in rows:
        by[source_of(r["rel"])].append(r)
    res = {"all": summary(rows), "by_source": {s: summary(v) for s, v in sorted(by.items())}, "errors": errors,
           "rule": "S-U if su_hit >= 0.8 over all audited episodes with gripper events, else S-UF (S-A: reported for "
                   "the user's choice, the full cache is built for one sampler only)"}
    su_hit = res["all"]["su_hit"]
    res["recommendation"] = "su" if su_hit is not None and su_hit >= 0.8 else "suf"
    (out / "audit_summary.json").write_text(json.dumps(res, indent=1))
    f = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"  # noqa: E731
    head = "".join(f" {n.upper() + ' ' + k:>11s}" for n in SAMPLERS_AUDITED for k in ("hit", "rms", "maxgap"))
    print(f"{'source':30s} {'eps':>5s} {'len':>6s} {'p90':>6s} {'events':>6s} {'gap':>5s}{head}")
    for s, v in list(res["by_source"].items()) + [("ALL", res["all"])]:
        cols = "".join(f" {f(v[f'{n}_{k}']):>11s}" for n in SAMPLERS_AUDITED for k in ("hit", "rms", "max_gap"))
        print(f"{s:30s} {v['episodes']:5d} {v['length_mean']:6.0f} {v['length_p90']:6.0f} {f(v['events_mean'], 1):>6s} "
              f"{f(v['slot_gap_frames'], 1):>5s}{cols}")
    print(f"recommendation: {res['recommendation']} (S-U hit {f(su_hit)}); {len(errors)} repo errors -> {out}")


# --- cache index / progress ------------------------------------------------------------------------------------------
def _scan(cache_dir: str, rel: str) -> list[int]:
    d = Path(cache_dir) / rel
    if not d.is_dir():
        return []
    out = []
    with os.scandir(d) as it:
        for e in it:
            if e.name.startswith("ep") and e.name.endswith(".pt") and ".tmp" not in e.name:
                out.append(int(e.name[2:7]))
    return out


def cmd_cache_index(args) -> None:
    pairs = manifest_repos(load_manifest(args.manifest), args.data_roots.split(":"), _sources(args))
    with ThreadPoolExecutor(args.workers) as pool:
        found = list(pool.map(lambda pe: (pe[1]["rel"], _scan(args.cache_dir, pe[1]["rel"])), pairs))
    lines = [f"{rel}|{ep}" for rel, eps in found for ep in sorted(eps)]
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} cached trajectories in {sum(1 for _, e in found if e)} repos -> {args.out}")


def cmd_progress(args) -> None:
    pairs = manifest_repos(load_manifest(args.manifest), args.data_roots.split(":"), _sources(args))
    with ThreadPoolExecutor(args.workers) as pool:
        found = list(pool.map(lambda pe: (pe[1], len(_scan(args.cache_dir, pe[1]["rel"]))), pairs))
    by = defaultdict(lambda: [0, 0, 0, 0])
    for entry, n in found:
        b = by[source_of(entry["rel"])]
        b[0] += n
        b[1] += int(entry["total_episodes"])
        held = set(entry["heldout"])
        b[3] += len(held)
        b[2] += sum(1 for ep in _scan(args.cache_dir, entry["rel"]) if ep in held) if held else 0
    fails = 0
    fail_dir = Path(args.cache_dir) / "_failures"
    if fail_dir.is_dir():
        fails = sum(sum(1 for _ in f.open()) for f in fail_dir.glob("*.jsonl"))
    done = sum(b[0] for b in by.values())
    total = sum(b[1] for b in by.values())
    snap = Path(args.cache_dir) / "_progress_last.json"
    now = time.time()
    rate = None
    if snap.is_file():
        prev = json.loads(snap.read_text())
        if now > prev["time"] and done >= prev["done"]:
            rate = (done - prev["done"]) / (now - prev["time"])
    snap.write_text(json.dumps({"time": now, "done": done}))
    print(f"{'source':34s} {'done':>9s} {'total':>9s} {'%':>6s} {'held-out':>12s}")
    for s, (d, t, hd, ht) in sorted(by.items(), key=lambda x: -x[1][1]):
        print(f"{s:34s} {d:9d} {t:9d} {100 * d / max(1, t):6.1f} {hd:5d}/{ht:<6d}")
    eta = f", ETA {(total - done - fails) / rate / 3600:.1f} h" if rate else ""
    print(f"TOTAL {done} / {total} ({100 * done / max(1, total):.1f}%), failures {fails}"
          f"{f', {rate:.1f} ep/s since last call' if rate else ''}{eta}")


def _sources(args):
    return args.sources.split(",") if getattr(args, "sources", "") else None


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--repo_file", required=True)
    m.add_argument("--data_root", default=None)
    m.add_argument("--exclude", default="", help="extra ','-separated path patterns to drop (egodex always)")
    m.add_argument("--out", required=True)
    m.add_argument("--train_list", default=None)
    m.add_argument("--workers", type=int, default=32)
    for name in ("prepare-arrays", "audit", "cache-index", "progress"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--data_roots", required=True, help="':'-separated local roots")
        p.add_argument("--sources", default="", help="','-separated top-level sources (default all)")
        p.add_argument("--workers", type=int, default=32)
        if name == "prepare-arrays":
            p.add_argument("--store", required=True)
            p.add_argument("--force", action="store_true")
        if name == "audit":
            p.add_argument("--array_store", default=None)
            p.add_argument("--episodes_per_repo", type=int, default=5)
            p.add_argument("--repos_per_source", type=int, default=0, help="at most N repos per source (0 = all)")
            p.add_argument("--num_slots", type=int, default=32)
            p.add_argument("--seed", type=int, default=0)
            p.add_argument("--out", required=True)
        if name in ("cache-index", "progress"):
            p.add_argument("--cache_dir", required=True)
        if name == "cache-index":
            p.add_argument("--out", required=True)
    args = ap.parse_args()
    {"manifest": cmd_manifest, "prepare-arrays": cmd_prepare_arrays, "audit": cmd_audit,
     "cache-index": cmd_cache_index, "progress": cmd_progress}[args.cmd](args)


if __name__ == "__main__":
    main()
