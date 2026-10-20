"""Offline validation of bpv4_bpen checkpoints on the held-out episodes (same queries, masks, noise and flow times
for every checkpoint, so numbers are directly comparable between the main run and the no-mask control).

Conditions (flow loss averaged over t in {0.1, 0.5, 0.9}):
  nomask        all keyframes visible
  event         event-segment mask around p (falls back to the ratio window when the segment mask is invalid)
  window        ratio-window mask around p
  p_drop        no mask, progress token replaced by the null token
  lang_drop     no mask, language removed
  swap_same     no mask, BP replaced by another held-out episode of the same repo (same task)
  swap_wrong    no mask, BP replaced by a held-out episode of a different task
  event_swap    event mask, BP replaced by another episode of the same task (mask follows the original keyframes)

Usage: python -m lerobot.policies.BPV4BPEn.offline_eval --ckpt <pretrained_model dir> --out <json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import default_collate

from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4BPEnDatasetConfig
from lerobot.policies.BPV4BPEn.dataset import BPV4BPEnDataset
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import BPV4BPEnPolicy
from lerobot.policies.q35.modeling_q35 import masked_flow_loss
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

CONDITIONS = ("nomask", "event", "window", "p_drop", "lang_drop", "swap_same", "swap_wrong", "event_swap")
TIMES = (0.1, 0.5, 0.9)
BP_KEYS = ("bp.feats", "bp.time", "bp.kf_valid")


def build_val_dataset(ckpt: Path, repo_file: str | None) -> BPV4BPEnDataset:
    train_cfg = json.loads((ckpt / "train_config.json").read_text())
    d = dict(train_cfg["dataset"])
    known = set(BPV4BPEnDatasetConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in d.items() if k in known and k not in ("type", "image_transforms")}
    kwargs["split"] = "val"
    if isinstance(kwargs.get("mask_window_ratio"), list):
        kwargs["mask_window_ratio"] = tuple(kwargs["mask_window_ratio"])
    cfg = BPV4BPEnDatasetConfig(**kwargs)
    repos = [l.strip() for l in Path(repo_file or d["repo_id_file"]).read_text().splitlines() if l.strip()]
    return BPV4BPEnDataset(cfg, SimpleNamespace(dataset=cfg), repos, chunk_size=int(train_cfg["policy"]["chunk_size"]))


def item(ds: BPV4BPEnDataset, i: int) -> dict:
    torch.manual_seed(10_000 + i)  # same queries / masks / drops for every checkpoint
    return ds[i]


def partners(ds: BPV4BPEnDataset) -> tuple[list[int], list[int]]:
    by_repo: dict[int, list[int]] = {}
    for idx, (ri, _) in enumerate(ds.items):
        by_repo.setdefault(ri, []).append(idx)
    task_of = {ri: Path(r.repo).parent.name for ri, r in enumerate(ds.repos)}
    same, wrong = [], []
    for idx, (ri, _) in enumerate(ds.items):
        group = by_repo[ri]
        same.append(group[(group.index(idx) + 1) % len(group)])
        others = [j for j, (rj, _) in enumerate(ds.items) if task_of[rj] != task_of[ri]]
        wrong.append(others[(idx * 7919) % len(others)])
    return same, wrong


@torch.no_grad()
def evaluate(policy: BPV4BPEnPolicy, ds: BPV4BPEnDataset, batch_size: int, device: torch.device) -> dict:
    model = policy.model
    same, wrong = partners(ds)
    sums = {c: 0.0 for c in CONDITIONS}
    split_sums = {c: {"clean": [0.0, 0], "rand": [0.0, 0]} for c in CONDITIONS}
    count = 0
    for start in range(0, len(ds), batch_size):
        idx = list(range(start, min(len(ds), start + batch_size)))
        base = default_collate([item(ds, i) for i in idx])
        bp_same = default_collate([{k: item(ds, same[i])[k] for k in BP_KEYS} for i in idx])
        bp_wrong = default_collate([{k: item(ds, wrong[i])[k] for k in BP_KEYS} for i in idx])
        base = {k: v.to(device) if torch.is_tensor(v) else v for k, v in base.items()}
        bp_same = {k: v.to(device) for k, v in bp_same.items()}
        bp_wrong = {k: v.to(device) for k, v in bp_wrong.items()}
        b, s = base["query.progress"].shape
        n = b * s
        gen = torch.Generator(device="cpu").manual_seed(start)
        noise = torch.randn((n, base[ACTION].shape[2], policy.config.max_action_dim), generator=gen).to(device)
        valid = base[ACTION_VALID_MASK].reshape(n, -1)
        kf = base["bp.kf_valid"][:, None].expand(-1, s, -1)
        event_vis = torch.where(base["bp.event_ok"][..., None], base["bp.vis_event"], base["bp.vis_window"])
        is_rand = torch.tensor(["randomized" in ds.repos[ds.items[i][0]].repo for i in idx], device=device)
        is_rand = is_rand[:, None].expand(-1, s).reshape(-1)
        for cond in CONDITIONS:
            batch = dict(base)
            visible = kf
            if cond == "event":
                visible = event_vis
            elif cond == "window":
                visible = base["bp.vis_window"]
            elif cond == "p_drop":
                batch["query.progress_drop"] = torch.ones_like(base["query.progress_drop"])
            elif cond == "lang_drop":
                batch["observation.attention_mask"] = torch.zeros_like(base["observation.attention_mask"])
            elif cond in ("swap_same", "swap_wrong", "event_swap"):
                batch.update(bp_wrong if cond == "swap_wrong" else bp_same)
                visible = batch["bp.kf_valid"][:, None].expand(-1, s, -1)
                if cond == "event_swap":
                    visible = visible & event_vis
            if cond != "p_drop":
                batch["query.progress_drop"] = torch.zeros_like(base["query.progress_drop"])
            per_query = torch.zeros(n, device=device)
            for t in TIMES:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    flow = model.forward_queries(batch, visible, noise=noise, time=torch.full((n,), t, device=device))["flow"]
                flow = flow[..., : base[ACTION].shape[-1]].float()
                w = valid[..., None].float()
                per_query += (flow * w).sum((1, 2)) / (w.sum((1, 2)) * flow.shape[-1]).clamp_min(1) / len(TIMES)
            sums[cond] += float(per_query.sum())
            for name, sel in (("clean", ~is_rand), ("rand", is_rand)):
                split_sums[cond][name][0] += float(per_query[sel].sum())
                split_sums[cond][name][1] += int(sel.sum())
        count += n
    return {
        "num_queries": count,
        "loss": {c: sums[c] / count for c in CONDITIONS},
        "loss_by_split": {c: {k: v[0] / max(1, v[1]) for k, v in split_sums[c].items()} for c in CONDITIONS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo_file", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ckpt = Path(args.ckpt)
    device = torch.device("cuda")
    policy = BPV4BPEnPolicy.from_pretrained(str(ckpt)).to(device).eval()
    ds = build_val_dataset(ckpt, args.repo_file)
    if args.limit:
        ds.items = ds.items[: args.limit]
    result = evaluate(policy, ds, args.batch_size, device)
    result["ckpt"] = str(ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result["loss"], indent=1))


if __name__ == "__main__":
    main()
