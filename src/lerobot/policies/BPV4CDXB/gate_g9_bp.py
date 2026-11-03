"""G9 on BP RoboTwin candidate sets: does the policy follow the demonstration? (offline, no simulator)

Queries: episodes of chosen candidate sets of BP RoboTwin split v1 (train / held-out sets / held-out tasks), each at
2 time steps (25% / 65% of the episode). For every query the SAME flow noise is used under every condition; only the
demonstration changes:
  matched  another episode of the query's candidate set
  blank    no demonstration (Z_demo set to zero)
  alt      the candidate-set episode whose actions differ most from the query's (64-point resampled actions, L2)
  wrong    an episode of a candidate set of another task
  goal     an episode of a candidate set with the same anchor and another goal (contrast group: same initial
           scene, different correct behaviour); the BP-specific test of reading the goal from the demonstration
Error: MSE of the sampled 50-step chunk vs the query's true chunk (normalized delta space, valid steps, real action
dims), as in gate_g9_main. Ratios E_cond / E_matched with a query-episode bootstrap 95% CI; overall, per split group
and per task. Every (query, condition) error is also written to <out>/<name>_rows.jsonl.

  index  python -m lerobot.policies.BPV4CDXB.gate_g9_bp index --out <g9 index json>
  run    python -m lerobot.policies.BPV4CDXB.gate_g9_bp run --ckpt <run>/checkpoints/004000/pretrained_model \
           --index <g9 index json> --out <dir> --name b4sft_004k
The index fixes the queries, so results of different checkpoints are paired.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from .gate_g9_main import P5_KEYS, chunk_error, resampled

COND = ("alt", "wrong", "goal")
GROUPS = ("train", "heldout_sets", "heldout_tasks")
ROBOTWIN = Path("/vla/workspace/my_robotwin")
DATA_ROOT = Path("/vla/workspace/data")
REL_PREFIX = "robotwin2.5_bp"


# ---------------------------------------------------------------------------------------------------------- index
def _split_groups(split_dir: Path) -> dict[str, str]:
    """'<task>/<set dir>' -> group, from bp_tasks/splits/<name>/<group>_repos.txt."""
    out = {}
    for group in GROUPS:
        for line in (split_dir / f"{group}_repos.txt").read_text().splitlines():
            if line.strip():
                p = Path(line.strip())
                out[f"{p.parent.name}/{p.name}"] = group
    return out


def _episodes(repo: Path) -> list[tuple[int, int]]:
    files = sorted((repo / "meta" / "episodes").rglob("*.parquet"))
    ep = pd.concat(pd.read_parquet(f, columns=["episode_index", "length"]) for f in files)
    return [(int(e), int(n)) for e, n in zip(ep["episode_index"], ep["length"])]


def build_index(args) -> None:
    rng = np.random.default_rng(args.seed)
    sets = json.loads((ROBOTWIN / "data_bp" / "bp_robotwin_lerobot" / "candidate_sets.json").read_text())
    group_of = _split_groups(ROBOTWIN / "bp_tasks" / "splits" / args.split)
    by_anchor: dict[str, list[str]] = {}
    for cs, e in sets.items():
        by_anchor.setdefault(e["anchor_id"], []).append(cs)
    by_group: dict[str, dict[str, list[str]]] = {g: {} for g in GROUPS}
    for cs, e in sets.items():
        g = group_of.get(e["path"])
        if g:
            by_group[g].setdefault(cs.split("/")[0], []).append(cs)
    per_task = {"train": args.train_sets_per_task, "heldout_sets": 10**6, "heldout_tasks": 10**6}
    targets = []
    for g in GROUPS:
        for task in sorted(by_group[g]):
            cands = sorted(by_group[g][task])
            # prefer candidate sets with a contrast partner (same anchor, another goal)
            cands.sort(key=lambda c: len(by_anchor[sets[c]["anchor_id"]]) < 2)
            k = min(per_task[g], len(cands))
            with_partner = [c for c in cands if len(by_anchor[sets[c]["anchor_id"]]) >= 2]
            pool = with_partner if len(with_partner) >= k else cands
            targets += [(g, c) for c in sorted(rng.choice(pool, size=k, replace=False).tolist())]
    need = {c for _, c in targets}
    for _, c in targets:
        need.update(x for x in by_anchor[sets[c]["anchor_id"]] if x != c)
    tasks = sorted({c.split("/")[0] for c in sets})
    wrong_of = {}
    for _, c in targets:
        others = [x for x in sets if x.split("/")[0] != c.split("/")[0]]
        wrong_of[c] = others[int(rng.integers(len(others)))]
        need.add(wrong_of[c])

    buckets, episodes, bidx = [], {}, {}
    for cs in sorted(need):
        e = sets[cs]
        rel = f"{REL_PREFIX}/{e['path']}"
        eids = []
        for ep, n in _episodes(DATA_ROOT / rel):
            eid = f"{rel}|{ep}"
            eids.append(eid)
            episodes[eid] = [rel, ep, n, len(buckets), "val"]
        bidx[cs] = len(buckets)
        buckets.append({"key": f"aloha||{rel}||{e['instruction']}", "robot_type": "aloha", "text": e["instruction"],
                        "arm": None, "train": [], "val": eids, "candidate_set": cs, "anchor_id": e["anchor_id"],
                        "goal": e["goal"], "group": group_of.get(e["path"])})
    queries = []
    for g, c in targets:
        pool = buckets[bidx[c]]["val"]
        n = {"train": args.train_queries_per_set, "heldout_sets": args.heldout_set_queries,
             "heldout_tasks": args.heldout_task_queries}[g]
        for q in sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()):
            queries.append({"q": q, "candidate_set": c, "group": g, "task": c.split("/")[0],
                            "goal_sets": [bidx[x] for x in by_anchor[sets[c]["anchor_id"]] if x != c],
                            "wrong_set": bidx[wrong_of[c]]})
    index = {"version": 1, "kind": "bp_g9", "split": args.split, "seed": args.seed, "zdemo_dir": "",
             "rule": "queries of chosen candidate sets; buckets = candidate sets (val = all episodes)",
             "buckets": buckets, "episodes": episodes, "g9_queries": queries, "tasks": tasks}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(index))
    count = {g: (sum(1 for x in targets if x[0] == g), sum(1 for q in queries if q["group"] == g)) for g in GROUPS}
    print(f"candidate sets loaded {len(buckets)}; (target sets, queries) per group {count}; "
          f"queries with a goal partner {sum(bool(q['goal_sets']) for q in queries)} / {len(queries)}")
    print(args.out)


# ---------------------------------------------------------------------------------------------------------- run
def make_jobs(ds, index: dict, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    jobs = []
    for qi in index["g9_queries"]:
        q = qi["q"]
        rel, ep, length, bi = ds.ep[q][0], int(ds.ep[q][1]), int(ds.ep[q][2]), ds.ep[q][3]
        pool = [e for e in ds.buckets[bi]["val"] if e != q]
        matched = pool[int(rng.integers(len(pool)))]
        job = {"q": q, "matched": matched, "frames": [int(round(0.25 * (length - 1))), int(round(0.65 * (length - 1)))],
               "raw_dim": int(sum(ds.repos[rel].emb.action_sizes)), "group": qi["group"], "task": qi["task"],
               "candidate_set": qi["candidate_set"], "pool": len(pool)}
        others = [e for e in pool if e != matched]
        if others:
            qa = resampled(np.asarray(ds.repos[rel].actions[ep]))
            dist = [float(np.linalg.norm(resampled(np.asarray(ds.repos[ds.ep[e][0]].actions[int(ds.ep[e][1])])) - qa))
                    for e in others]
            job["alt"] = others[int(np.argmax(dist))]
        wb = ds.buckets[qi["wrong_set"]]["val"]
        job["wrong"] = wb[int(rng.integers(len(wb)))]
        if qi["goal_sets"]:
            gb = ds.buckets[qi["goal_sets"][int(rng.integers(len(qi["goal_sets"])))]]
            job["goal"] = gb["val"][int(rng.integers(len(gb["val"])))]
            job["goal_set"] = gb["candidate_set"]
        jobs.append(job)
    return jobs


class Jobs(torch.utils.data.Dataset):
    def __init__(self, ds, jobs) -> None:
        self.ds, self.jobs = ds, jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i):
        j = self.jobs[i]
        item = self.ds.build(j["q"], j["matched"], j["frames"])
        for k in COND:
            d = j.get(k)
            # placeholder = the matched demo (an all-False camera mask would give NaN attention in the encoder)
            rec = self.ds.demo_p5(d) if d else {key: item[key].clone() for key in P5_KEYS}
            item.update({f"{key}.{k}": rec[key] for key in P5_KEYS})
            item[f"has.{k}"] = torch.tensor(d is not None)
        item["job"] = torch.tensor(i)
        item["raw_dim"] = torch.tensor(j["raw_dim"])
        return item


def summarize(rows: list[dict], key: str | None) -> dict:
    from lerobot.policies.BPV4CDX.common import paired_bootstrap_ratio

    out = {}
    for name in sorted({r[key] for r in rows}) if key else [None]:
        sel = [r for r in rows if key is None or r[key] == name]
        entry = {"samples": len(sel), "queries": len({r["q"] for r in sel}),
                 "matched_mse": float(np.mean([r["matched"] for r in sel]))}
        for k in ("blank",) + COND:
            sk = [r for r in sel if k in r]
            if sk:
                entry[f"{k}_mse"] = float(np.mean([r[k] for r in sk]))
                entry[k] = paired_bootstrap_ratio(np.array([r[k] for r in sk]), np.array([r["matched"] for r in sk]),
                                                  np.array([r["q"] for r in sk]))
        if key is None:
            return entry
        out[name] = entry
    return out


def run(args) -> None:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV4CDX.common import write_result
    from lerobot.policies.BPV4CDXB.dataset_main import make_main_dataset
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    ckpt = Path(args.ckpt)
    cfg_dir = ckpt
    # exported checkpoints (e.g. LoRA merged) carry no train_config.json; merge_info.json names the source checkpoint
    if not (ckpt / "train_config.json").is_file() and (ckpt / "merge_info.json").is_file():
        cfg_dir = Path(json.loads((ckpt / "merge_info.json").read_text())["source_checkpoint"])
        logging.info("train_config.json from the merge source %s", cfg_dir)
    cfg = TrainPipelineConfig.from_pretrained(str(cfg_dir))
    cfg.dataset.split = "val"
    cfg.dataset.index = args.index
    # the observation ViT cache of a run covers its training episodes only; G9 also queries held-out ones (online
    # ViT tokens are bit-identical to the cache)
    cfg.dataset.obs_cache_dir = ""
    index = json.loads(Path(args.index).read_text())
    index["load_procs"] = args.load_procs
    t0 = time.time()
    ds, _ = make_main_dataset(cfg, index)
    jobs = make_jobs(ds, index, args.seed)
    logging.info("dataset + jobs in %.0f s: %d queries (alt %d, goal %d)", time.time() - t0, len(jobs),
                 sum("alt" in j for j in jobs), sum("goal" in j for j in jobs))
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    policy_cfg.init_from = ""  # every weight is in the checkpoint; init_from was only the training start point
    policy = BPV4CDXBPolicy.from_pretrained(str(ckpt), config=policy_cfg).cuda().eval()
    if getattr(policy.model, "demo_encoder", None) is None:
        raise RuntimeError("gate_g9_bp needs a checkpoint with the online demo encoder (B-4, demo_features='p5')")
    loader = torch.utils.data.DataLoader(Jobs(ds, jobs), batch_size=args.batch, num_workers=args.workers,
                                         prefetch_factor=2 if args.workers else None)
    rows = []
    H, D = int(policy.config.chunk_size), int(policy.config.max_action_dim)
    for bi, b in enumerate(loader):
        b = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        B, S = b["job"].shape[0], b[ACTION].shape[1]
        noise = torch.cat([torch.randn(S, H, D, generator=torch.Generator().manual_seed(args.seed * 100003 + int(j)))
                           for j in b["job"].cpu()]).cuda()
        target = torch.nn.functional.pad(b[ACTION].reshape(B * S, H, -1), (0, D - b[ACTION].shape[-1]))
        valid = b[ACTION_VALID_MASK].reshape(B * S, H).bool()
        raw = b["raw_dim"].repeat_interleave(S)
        preds = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds["matched"] = policy.model.sample_actions(b, noise=noise)
            preds["blank"] = policy.model.sample_actions(b, noise=noise, blank_demo=torch.ones(B, dtype=torch.bool, device="cuda"))
            for k in COND:
                if bool(b[f"has.{k}"].any()):
                    preds[k] = policy.model.sample_actions({**b, **{key: b[f"{key}.{k}"] for key in P5_KEYS}}, noise=noise)
        err = {k: chunk_error(p, target, valid, raw).cpu().numpy() for k, p in preds.items()}
        for i in range(B):
            j = jobs[int(b["job"][i])]
            for s in range(S):
                n = i * S + s
                row = {"q": j["q"], "candidate_set": j["candidate_set"], "task": j["task"], "group": j["group"],
                       "frame": j["frames"][s], "phase": ("25%", "65%")[s], "pool": j["pool"],
                       "matched": float(err["matched"][n]), "blank": float(err["blank"][n])}
                for k in COND:
                    if k in err and bool(b[f"has.{k}"][i]):
                        row[k] = float(err[k][n])
                if "goal_set" in j:
                    row["goal_set"] = j["goal_set"]
                rows.append(row)
        if bi % 10 == 0:
            logging.info("batch %d / %d", bi + 1, len(loader))

    overall = summarize(rows, None)
    res = {"ckpt": str(ckpt), "index": str(args.index), "queries": len(jobs), "samples": len(rows),
           "mean_error": {k: overall[f"{k}_mse"] for k in ("matched", "blank") + COND if f"{k}_mse" in overall},
           "ratios": {k: overall[k] for k in ("blank",) + COND if k in overall},
           "by_group": summarize(rows, "group"), "by_task": summarize(rows, "task"), "by_phase": summarize(rows, "phase"),
           "note": "MSE in normalized delta action space; ratio = E_cond / E_matched (> 1: the matched demo helps)"}
    rb = res["ratios"].get("blank")
    res["G9_blank_gain"] = bool(rb and rb["ratio"] >= 1.10 and rb["ci95"][0] > 1.0)
    path = write_result(args.out, args.name, res)
    with (Path(args.out) / f"{args.name}_rows.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    logging.info("written %s", path)
    print(args.name, {k: (round(v["ratio"], 3), [round(x, 3) for x in v["ci95"]]) for k, v in res["ratios"].items()},
          "matched", round(res["mean_error"]["matched"], 4))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("index")
    pi.add_argument("--out", required=True)
    pi.add_argument("--split", default="v1")
    pi.add_argument("--seed", type=int, default=0)
    pi.add_argument("--train_sets_per_task", type=int, default=2)
    pi.add_argument("--train_queries_per_set", type=int, default=2)
    pi.add_argument("--heldout_set_queries", type=int, default=8)
    pi.add_argument("--heldout_task_queries", type=int, default=15)
    pr = sub.add_parser("run")
    pr.add_argument("--ckpt", required=True)
    pr.add_argument("--index", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--name", required=True)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--batch", type=int, default=8, help="queries per forward (x 2 time steps)")
    pr.add_argument("--workers", type=int, default=6)
    pr.add_argument("--load_procs", type=int, default=8)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    build_index(args) if args.cmd == "index" else run(args)


if __name__ == "__main__":
    main()
