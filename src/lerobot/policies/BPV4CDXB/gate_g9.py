"""G9 (offline, `bpv4cdx Stage B 开发文档 v1.md` 4.1 / 4.2 / 4.4) on the main-path same-task index.

Fixed held-out queries (AE-3 held-out episodes; demo pools never cross the split), round robin over buckets; per
query fixed time steps (linspace) and fixed noise; only the demo changes:
  matched     another held-out episode of the same bucket (identical description), same arm when possible
  blank       the matched batch with Z_demo zeroed
  cross_arm   same bucket, the other arm (only buckets that have both arms)
  wrong_task  a held-out episode of another bucket of the same repo (else of the same robot type)
Errors: masked MSE over valid steps and real action dims in the normalized delta space (dataset targets).
Ratios E_x / E_matched with a trajectory-cluster bootstrap 95% CI (BPV4CDX.common). Cross-arm follow (dual-arm
6+1 robots): the arm with the larger predicted joint motion equals the demo's arm; 0.5 = chance.

In training: dataset.make_heldout_monitor() -> G9Monitor, called by lerobot_train at every checkpoint save (main
process evaluates, records go to <output_dir>/heldout_monitor.jsonl).
Offline: python -m lerobot.policies.BPV4CDXB.gate_g9 --ckpt <run>/checkpoints/<step>/pretrained_model \
           [--queries 512] [--out <dir>]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.common import bootstrap_mean, collate, masked_error, paired_bootstrap_ratio, to_device
from lerobot.policies.BPV4CDX.embodiment import Embodiment
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

CONDITIONS = ("matched", "blank", "cross_arm", "wrong_task")


def make_plan(ds, n_queries: int, seed: int = 0) -> list[dict]:
    """Fixed held-out evaluation plan (same for every call / checkpoint)."""
    rng = np.random.default_rng(seed)
    by_repo: dict[int, list[int]] = {}
    by_robot: dict[str, list[int]] = {}
    for bi, b in enumerate(ds.buckets):
        if len(b["pool"]["val"]) >= 2:
            by_repo.setdefault(b["repo"], []).append(bi)
            by_robot.setdefault(ds.repos[b["repo"]].robot_type, []).append(bi)
    order = [bi for bi in rng.permutation(sorted(bi for v in by_repo.values() for bi in v))]
    queues = {bi: list(rng.permutation(ds.buckets[bi]["pool"]["val"])) for bi in order}
    plan = []
    while len(plan) < n_queries and any(queues.values()):
        for bi in order:
            if not queues[bi] or len(plan) >= n_queries:
                continue
            ep = int(queues[bi].pop())
            b = ds.buckets[bi]
            qa = b["arm"].get(ep)
            same = ds.demo_pool(bi, ep, "val", arm=qa) or ds.demo_pool(bi, ep, "val")
            cross = ds.demo_pool(bi, ep, "val", other_arm=True)
            others = [x for x in by_repo[b["repo"]] if x != bi] or \
                     [x for x in by_robot[ds.repos[b["repo"]].robot_type] if x != bi]
            wrong = None
            if others:
                bj = int(rng.choice(others))
                wrong = (bj, int(rng.choice(ds.buckets[bj]["pool"]["val"])))
            plan.append({"bi": int(bi), "ep": ep, "frames": ds.query_frames(bi, ep, fixed=True),
                         "matched": (int(bi), int(rng.choice(same))),
                         "cross_arm": (int(bi), int(rng.choice(cross))) if cross else None, "wrong_task": wrong})
    return plan


class _EvalSet(torch.utils.data.Dataset):
    def __init__(self, ds, plan: list[dict]) -> None:
        self.ds, self.plan = ds, plan

    def __len__(self) -> int:
        return len(self.plan)

    def __getitem__(self, i: int) -> dict:
        p = self.plan[i]
        batch = self.ds.build(p["bi"], p["ep"], p["matched"], p["frames"])
        for c in ("cross_arm", "wrong_task"):
            batch[f"z.{c}"] = self.ds.zdemo(*p[c]) if p[c] else torch.zeros_like(batch["demo.z"])
            batch[f"has.{c}"] = torch.tensor(p[c] is not None)
        batch["plan.index"] = torch.tensor(i)
        return batch


def _arm_energy(pred: torch.Tensor, valid: torch.Tensor, emb: Embodiment) -> dict[str, float]:
    w = valid[..., None].float()
    return {arm: float((pred[..., sl].abs() * w).sum()) for arm, (sl, _) in emb.arms.items()}


@torch.no_grad()
def evaluate(policy, ds, plan: list[dict], *, batch_queries: int = 4, num_workers: int = 8) -> dict:
    """-> per-query errors (mean over its time steps) per condition and follow records, aggregated."""
    model = policy.model if hasattr(policy, "model") else policy
    was_training = model.training
    model.eval()
    loader = torch.utils.data.DataLoader(_EvalSet(ds, plan), batch_size=batch_queries, num_workers=num_workers,
                                         collate_fn=collate, multiprocessing_context="spawn" if num_workers else None)
    cfg = model.config
    horizon, dim = int(cfg.chunk_size), int(cfg.max_action_dim)
    err = {c: np.full(len(plan), np.nan) for c in CONDITIONS}
    follow, follow_ref = [], []
    t0 = time.time()
    for batch in loader:
        idx = batch.pop("plan.index").tolist()
        batch = to_device(batch)
        b, s = batch["obs.input_ids"].shape[:2]
        noise = torch.stack([torch.randn((s, horizon, dim), generator=torch.Generator().manual_seed(i)) for i in idx])
        noise = noise.reshape(b * s, horizon, dim).cuda()
        target = batch[ACTION].reshape(b * s, horizon, -1)
        valid_steps = batch[ACTION_VALID_MASK].reshape(b * s, -1).bool()
        base_z = batch["demo.z"]
        preds = {}
        for c in CONDITIONS:
            blank = None
            if c == "blank":
                blank = torch.ones(b, dtype=torch.bool, device=base_z.device)
            batch["demo.z"] = batch[f"z.{c}"] if c in ("cross_arm", "wrong_task") else base_z
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = model.sample_actions(batch, noise=noise, blank_demo=blank).float()
            preds[c] = pred
            e = masked_error(pred, target, valid_steps).reshape(b, s).mean(1).cpu().numpy()
            for j, i in enumerate(idx):
                if c in ("cross_arm", "wrong_task") and plan[i][c] is None:
                    continue
                err[c][i] = e[j]
        batch["demo.z"] = base_z
        for j, i in enumerate(idx):
            p = plan[i]
            bucket = ds.buckets[p["bi"]]
            emb = ds.repos[bucket["repo"]].emb
            if not emb.dual_arm_61:
                continue
            sl = slice(j * s, (j + 1) * s)
            qa = bucket["arm"].get(p["ep"])
            if qa in ("L", "R"):
                en = _arm_energy(preds["matched"][sl], valid_steps[sl], emb)
                follow_ref.append(float(max(en, key=en.get) == qa))
            if p["cross_arm"] is not None:
                da = bucket["arm"].get(p["cross_arm"][1])
                en = _arm_energy(preds["cross_arm"][sl], valid_steps[sl], emb)
                follow.append(float(max(en, key=en.get) == da))
    if was_training:
        model.train()
    return summarize(ds, plan, err, follow, follow_ref, time.time() - t0)


def summarize(ds, plan, err, follow, follow_ref, seconds: float) -> dict:
    groups = np.array([p["bi"] for p in plan])
    m = err["matched"]
    out = {"queries": len(plan), "segments": int(len(plan) * len(plan[0]["frames"])) if plan else 0,
           "seconds": round(seconds, 1), "mse_matched": float(np.nanmean(m)) if len(m) else None}
    for c in ("blank", "cross_arm", "wrong_task"):
        ok = ~np.isnan(err[c]) & ~np.isnan(m)
        out[f"mse_{c}"] = float(err[c][ok].mean()) if ok.any() else None
        out[f"{c}_over_match"] = paired_bootstrap_ratio(err[c][ok], m[ok], groups[ok]) if ok.sum() >= 2 else None
    out["cross_arm_follow"] = bootstrap_mean(np.array(follow)) if follow else None
    out["matched_arm_follow_ref"] = bootstrap_mean(np.array(follow_ref)) if follow_ref else None
    by_rt: dict[str, list[float]] = {}
    for p, e in zip(plan, m):
        by_rt.setdefault(ds.repos[ds.buckets[p["bi"]]["repo"]].robot_type, []).append(float(e))
    out["mse_matched_by_robot"] = {k: float(np.mean(v)) for k, v in sorted(by_rt.items())}
    bl = out.get("blank_over_match") or {}
    out["g9_gain_pass"] = bool(bl and bl["ratio"] >= 1.10 and bl["ci95"][0] > 1.0)
    return out


def line(r: dict) -> str:
    def ratio(k):
        v = r.get(k)
        return "-" if not v else f"{v['ratio']:.3f} [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]"

    f = r.get("cross_arm_follow")
    follow = "-" if not f else "{:.3f} (n={})".format(f["mean"], f["n"])
    return (f"E_matched={r['mse_matched']:.4f} blank/matched={ratio('blank_over_match')} "
            f"wrong/matched={ratio('wrong_task_over_match')} cross/matched={ratio('cross_arm_over_match')} "
            f"cross_follow={follow} queries={r['queries']} ({r['seconds']} s)")


class G9Monitor:
    """Training-time G9 on a fixed held-out subset (dataset.monitor_queries); main process only."""

    def __init__(self, ds) -> None:
        self.ds = ds
        self.plan = make_plan(ds, int(ds.cfg.monitor_queries))
        logging.info("G9 monitor: %d held-out queries, %d with a cross-arm demo", len(self.plan),
                     sum(p["cross_arm"] is not None for p in self.plan))

    def __call__(self, policy, accelerator, step: int) -> dict:
        if accelerator is not None and not accelerator.is_main_process:
            return {}
        if not self.plan:
            return {"step": step, "mse_matched": None, "segments": 0}
        r = evaluate(policy, self.ds, self.plan, num_workers=4)
        logging.info("G9 @ step %d: %s", step, line(r))
        return {"step": step, **r}


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig

    from .dataset_pretrain import BPV4CDXBPretrainDataset
    from .modeling_bpv4cdxb import BPV4CDXBPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="<run>/checkpoints/<step>/pretrained_model")
    ap.add_argument("--queries", type=int, default=512)
    ap.add_argument("--batch_queries", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--out", default=None, help="default: <ckpt>/../g9")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ck = Path(args.ckpt)
    cfg = TrainPipelineConfig.from_pretrained(str(ck / "train_config.json"))
    cfg.dataset.split = "val"
    ds = BPV4CDXBPretrainDataset(cfg.dataset, cfg, cfg.policy)
    policy = BPV4CDXBPolicy.from_pretrained(str(ck), config=cfg.policy).to("cuda").eval()
    plan = make_plan(ds, args.queries)
    r = evaluate(policy, ds, plan, batch_queries=args.batch_queries, num_workers=args.num_workers)
    r.update({"ckpt": str(ck.resolve()), "index": cfg.dataset.index, "zdemo_dir": cfg.dataset.zdemo_dir,
              "frozen_sha256": getattr(policy.model, "frozen_sha256", None)})
    out = Path(args.out) if args.out else ck.parent / "g9"
    out.mkdir(parents=True, exist_ok=True)
    (out / "g9_result.json").write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(line(r))
    print(json.dumps({k: r[k] for k in ("blank_over_match", "wrong_task_over_match", "cross_arm_follow",
                                        "g9_gain_pass")}, indent=1))
    print("wrote", out / "g9_result.json")


if __name__ == "__main__":
    main()
