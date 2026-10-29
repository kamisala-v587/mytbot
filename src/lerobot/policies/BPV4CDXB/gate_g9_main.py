"""Compact G9 for the Stage B main path (StageB实施.md S4): does the B-2 policy read the demo?

Held-out queries (val pools of the bucket index), each at 2 time steps (25% / 65% of the episode). For every query the
SAME flow noise is used under every condition; only the demo changes:
  matched     another held-out episode of the query's bucket (same robot type + identical task text)
  blank       Z_demo set to zero
  alternate   the bucket episode whose actions differ most from the query's (64-point resampled actions, L2);
              skipped when the bucket has no third episode
  wrong       an episode of another bucket of the same robot type
  left/right  cross-arm follow (dual-arm 6+1 robots, buckets whose text names no arm): one demo whose motion is mostly
              the left arm and one mostly the right arm; follow = the predicted chunk moves the demo's arm more
Error: MSE of the sampled 50-step chunk vs the query's true chunk (normalized delta space, valid steps, real action
dims). Ratios E_cond / E_matched with a query-episode bootstrap 95% CI; per robot type for blank.

usage: python -m lerobot.policies.BPV4CDXB.gate_g9_main --ckpt <run>/checkpoints/0005000/pretrained_model \
         --out outputs/BPv4CDX/stage_b/g9 --name b2_005k [--queries 1000 --workers 16 --batch 8]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

COND = ("alt", "wrong", "left", "right")
P5_KEYS = ("demo.feats", "demo.time", "demo.camera_valid")


def resampled(actions: np.ndarray, n: int = 64) -> np.ndarray:
    t = np.linspace(0, len(actions) - 1, n)
    return np.stack([np.interp(t, np.arange(len(actions)), actions[:, d]) for d in range(actions.shape[1])], 1)


def arm_of(repo, episode: int) -> str | None:
    """'L' / 'R' when one arm moves at least twice as much as the other over the episode (dual-arm 6+1 only)."""
    arms = repo.emb.arms
    if not arms:
        return None
    a = np.asarray(repo.actions[episode])
    energy = {k: float(np.abs(np.diff(a[:, j], axis=0)).sum() + np.abs(np.diff(a[:, g])).sum()) for k, (j, g) in arms.items()}
    if energy["L"] > 2 * energy["R"]:
        return "L"
    if energy["R"] > 2 * energy["L"]:
        return "R"
    return None


class Jobs(torch.utils.data.Dataset):
    def __init__(self, ds, jobs) -> None:
        self.ds, self.jobs = ds, jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __getitem__(self, i):
        j = self.jobs[i]
        item = self.ds.build(j["q"], j["matched"], j["frames"])
        p5 = "demo.feats" in item  # B-4: the demo enters as P5 features encoded online, so swap those
        for k in COND:
            d = j.get(k)
            if p5:
                # placeholder = the matched demo (an all-False camera mask would give NaN attention in the encoder)
                rec = self.ds.demo_p5(d) if d else {key: item[key].clone() for key in P5_KEYS}
                item.update({f"{key}.{k}": rec[key] for key in P5_KEYS})
            else:
                item[f"z.{k}"] = self.ds.zdemo(d) if d else torch.zeros_like(item["demo.z"])
            item[f"has.{k}"] = torch.tensor(d is not None)
        item["job"] = torch.tensor(i)
        item["raw_dim"] = torch.tensor(j["raw_dim"])
        return item


def make_jobs(ds, n_queries: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    queries = [ds.items[i] for i in sorted(rng.choice(len(ds.items), size=min(n_queries, len(ds.items)), replace=False))]
    by_robot: dict[str, list[int]] = {}
    for bi, b in enumerate(ds.buckets):
        if b["val"]:
            by_robot.setdefault(b["robot_type"], []).append(bi)
    arm_cache: dict[str, str | None] = {}

    def arm(eid):
        if eid not in arm_cache:
            rel, ep = ds.ep[eid][0], int(ds.ep[eid][1])
            arm_cache[eid] = arm_of(ds.repos[rel], ep)
        return arm_cache[eid]

    jobs = []
    for q in queries:
        rel, ep, length, bi = ds.ep[q][0], int(ds.ep[q][1]), int(ds.ep[q][2]), ds.ep[q][3]
        b, repo = ds.buckets[bi], ds.repos[rel]
        pool = [e for e in b["val"] if e != q]
        matched = pool[int(rng.integers(len(pool)))]
        parts = rel.split("/")
        source = f"InternData/{parts[2]}" if parts[0].startswith("full-InternData") and len(parts) > 2 else parts[0]
        job = {"q": q, "matched": matched, "frames": [int(round(0.25 * (length - 1))), int(round(0.65 * (length - 1)))],
               "raw_dim": int(sum(repo.emb.action_sizes)), "robot": b["robot_type"], "bucket": bi, "text_arm": b["arm"],
               "pool": len(pool), "source": source}
        others = [e for e in pool if e != matched]
        if others:
            qa = resampled(np.asarray(repo.actions[ep]))
            dist = []
            for e in others:
                r2 = ds.repos[ds.ep[e][0]]
                dist.append(float(np.linalg.norm(resampled(np.asarray(r2.actions[int(ds.ep[e][1])])) - qa)))
            job["alt"] = others[int(np.argmax(dist))]
        cand = [x for x in by_robot.get(b["robot_type"], []) if x != bi]
        if cand:
            wb = ds.buckets[cand[int(rng.integers(len(cand)))]]
            job["wrong"] = wb["val"][int(rng.integers(len(wb["val"])))]
        if b["arm"] is None and repo.emb.dual_arm_61:
            lefts = [e for e in pool if arm(e) == "L"]
            rights = [e for e in pool if arm(e) == "R"]
            if lefts and rights:
                job["left"], job["right"] = lefts[int(rng.integers(len(lefts)))], rights[int(rng.integers(len(rights)))]
                job["q_arm"] = arm(q)
        jobs.append(job)
    return jobs


def chunk_error(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
    """[N, H, D] x [N, H, D] x [N, H] x [N] -> per-sample MSE over valid steps and the first raw dims."""
    d = torch.arange(pred.shape[-1], device=pred.device)[None, None, :] < raw[:, None, None]
    m = valid[:, :, None] & d
    se = ((pred.float() - target.float()) ** 2) * m
    return se.sum((1, 2)) / m.sum((1, 2)).clamp(min=1)


def active_arm(pred: torch.Tensor, arms: dict) -> list[str]:
    """Arm whose joints move more in the predicted chunk (normalized delta space)."""
    out = []
    for p in pred:
        e = {k: float(p[:, j].abs().mean() + (p[:, g] - p[0, g]).abs().mean()) for k, (j, g) in arms.items()}
        out.append(max(e, key=e.get))
    return out


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV4CDX.common import paired_bootstrap_ratio, write_result
    from lerobot.policies.BPV4CDXB.dataset import make_bpv4cdxb_dataset
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=8, help="queries per forward (x 2 time steps)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ckpt = Path(args.ckpt)
    cfg = TrainPipelineConfig.from_pretrained(str(ckpt))
    cfg.dataset.split = "val"
    ds, _ = make_bpv4cdxb_dataset(cfg)
    t0 = time.time()
    jobs = make_jobs(ds, args.queries, args.seed)
    logging.info("jobs %d (alt %d, wrong %d, cross-arm %d) in %.0f s", len(jobs), sum("alt" in j for j in jobs),
                 sum("wrong" in j for j in jobs), sum("left" in j for j in jobs), time.time() - t0)
    policy = BPV4CDXBPolicy.from_pretrained(str(ckpt)).cuda().eval()
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
                    bb = dict(b)
                    if "demo.feats" in b:
                        bb.update({key: b[f"{key}.{k}"] for key in P5_KEYS})
                    else:
                        bb["demo.z"] = b[f"z.{k}"]
                    preds[k] = policy.model.sample_actions(bb, noise=noise)
        err = {k: chunk_error(p, target, valid, raw).cpu().numpy() for k, p in preds.items()}
        for i in range(B):
            j = jobs[int(b["job"][i])]
            for s in range(S):
                n = i * S + s
                row = {"q": j["q"], "robot": j["robot"], "bucket": j["bucket"], "frame": j["frames"][s],
                       "pool": j["pool"], "source": j["source"],
                       "matched": float(err["matched"][n]), "blank": float(err["blank"][n])}
                for k in COND:
                    if k in err and bool(b[f"has.{k}"][i]):
                        row[k] = float(err[k][n])
                if "left" in j:
                    arms = ds.repos[ds.ep[j["q"]][0]].emb.arms
                    row["arm_left_demo"] = active_arm(preds["left"][n : n + 1], arms)[0]
                    row["arm_right_demo"] = active_arm(preds["right"][n : n + 1], arms)[0]
                    row["arm_matched"] = active_arm(preds["matched"][n : n + 1], arms)[0]
                    row["q_arm"] = j.get("q_arm")
                rows.append(row)
        if bi % 10 == 0:
            logging.info("batch %d / %d", bi + 1, len(loader))

    groups = np.array([r["q"] for r in rows])
    m = np.array([r["matched"] for r in rows])
    res = {"ckpt": str(ckpt), "queries": len(jobs), "samples": len(rows), "mean_error": {"matched": float(m.mean())},
           "ratios": {}, "by_robot_blank": {}, "cross_arm": None}
    for k in ("blank",) + COND[:2]:
        sel = np.array([k in r for r in rows])
        if sel.any():
            num = np.array([r[k] for r in rows if k in r])
            res["mean_error"][k] = float(num.mean())
            res["ratios"][k] = paired_bootstrap_ratio(num, m[sel], groups[sel])
    for robot in sorted({r["robot"] for r in rows}):
        sel = np.array([r["robot"] == robot for r in rows])
        bl = np.array([r["blank"] for r in rows])[sel]
        res["by_robot_blank"][robot] = paired_bootstrap_ratio(bl, m[sel], groups[sel])
    # per data source: blank / alternate / wrong; alternate also by the size of the query's same-text held-out pool
    res["by_source"], res["alt_by_pool"] = {}, {}
    for src in sorted({r["source"] for r in rows}):
        sel = [r for r in rows if r["source"] == src]
        entry = {"samples": len(sel), "pool_median": float(np.median([r["pool"] for r in sel]))}
        for k in ("blank", "alt", "wrong"):
            sk = [r for r in sel if k in r]
            if sk:
                entry[k] = paired_bootstrap_ratio(np.array([r[k] for r in sk]), np.array([r["matched"] for r in sk]),
                                                  np.array([r["q"] for r in sk]))
        res["by_source"][src] = entry
    for name, lo, hi in (("2-3", 2, 3), ("4-6", 4, 6), ("7-15", 7, 15), ("16+", 16, 10**9)):
        sk = [r for r in rows if "alt" in r and lo <= r["pool"] <= hi]
        if sk:
            res["alt_by_pool"][name] = paired_bootstrap_ratio(np.array([r["alt"] for r in sk]),
                                                              np.array([r["matched"] for r in sk]), np.array([r["q"] for r in sk]))
    xr = [r for r in rows if "arm_left_demo" in r]
    if xr:
        follow = np.array([[r["arm_left_demo"] == "L", r["arm_right_demo"] == "R"] for r in xr], dtype=float)
        per_q = {}
        for r, f in zip(xr, follow):
            per_q.setdefault(r["q"], []).append(f.mean())
        vals = np.array([np.mean(v) for v in per_q.values()])
        rng = np.random.default_rng(0)
        boots = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(2000)]
        res["cross_arm"] = {"follow_rate": float(follow.mean()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                            "queries": len(per_q), "samples": len(xr),
                            "switch_rate": float(np.mean([r["arm_left_demo"] != r["arm_right_demo"] for r in xr])),
                            "note": "random = 0.5; arm = which arm moves more in the predicted chunk"}
    rb = res["ratios"].get("blank")
    res["G9_blank_gain"] = bool(rb and rb["ratio"] >= 1.10 and rb["ci95"][0] > 1.0)
    path = write_result(args.out, args.name, res)
    logging.info("written %s", path)
    print({k: (v["ratio"], v["ci95"]) for k, v in res["ratios"].items()}, "cross_arm", res["cross_arm"] and
          (res["cross_arm"]["follow_rate"], res["cross_arm"]["ci95"], res["cross_arm"]["queries"]))


if __name__ == "__main__":
    main()
