"""Offline evaluation of bpv4_xdemo checkpoints on held-out episodes (both the current trajectory and every
demonstration come from the validation split). Queries, demonstrations, noise and flow times are fixed, so numbers are
comparable across checkpoints.

Conditions (flow loss averaged over t in {0.1, 0.5, 0.9}):
  normal      demonstration = another validation episode of the same task (and category)
  swap_same   demonstration = a third validation episode of the same task          -> should stay close to normal
  swap_wrong  demonstration from a different task                                  -> should rise clearly
  no_bp       BP tokens hidden from the prefix                                     -> should rise clearly
  no_image    current-image tokens hidden from the prefix                          -> should rise clearly
  scene_swap  only for same-scene task pairs (shake_bottle <-> shake_bottle_horizontally): demonstration from the
              paired task in the identical scene                                   -> should rise (must follow the demo)
normal is also split by whether the demonstration's grasp arm equals the current one (same_arm / diff_arm).

Usage: python -m lerobot.policies.BPV4XDemo.offline_eval --ckpt <pretrained_model> --out <json> [--limit N]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import default_collate

from lerobot.policies.BPV4XDemo.dataset import eval_dataset
from lerobot.policies.BPV4XDemo.modeling_bpv4_xdemo import BPV4XDemoPolicy
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

TIMES = (0.1, 0.5, 0.9)
SCENE_PAIRS = {"shake_bottle": "shake_bottle_horizontally", "shake_bottle_horizontally": "shake_bottle"}
BP_KEYS = ("bp.feats", "bp.time", "bp.kf_valid", "bp.visible")
CONDITIONS = ("normal", "swap_same", "swap_wrong", "no_bp", "no_image", "scene_swap")


class EvalPairs(torch.utils.data.Dataset):
    """Item i: query i with fixed frames, its current inputs, and the BP of each condition's demonstration."""

    def __init__(self, ds, limit: int = 0):
        self.ds = ds
        self.items = ds.items[:limit] if limit else ds.items
        groups = sorted(ds.pools)
        self.by_task = {}
        for g in groups:
            self.by_task.setdefault(g[0], []).extend(sorted(ds.pools[g]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ds, query = self.ds, self.items[i]
        group = ds.group_of[query]
        pool = [d for d in sorted(ds.pools[group]) if d != query]
        j = sorted(ds.pools[group]).index(query)
        demos = {"normal": pool[j % len(pool)], "swap_same": pool[(j + 1) % len(pool)]}
        others = [t for t in self.by_task if t != group[0]]
        wrong_task = others[(i * 7919) % len(others)]
        demos["swap_wrong"] = self.by_task[wrong_task][i % len(self.by_task[wrong_task])]
        pair = SCENE_PAIRS.get(group[0])
        if pair in self.by_task:
            demos["scene_swap"] = self.by_task[pair][i % len(self.by_task[pair])]
        torch.manual_seed(10_000 + i)
        frames = ds.query_frames(*query)
        out = ds.build(query, demos["normal"], frames, drop=False)
        for cond in ("swap_same", "swap_wrong", "scene_swap"):
            if cond in demos:
                bp = ds.demo_features(*demos[cond], drop=False)
                out.update({f"{cond}/{k}": v for k, v in bp.items()})
        out["has_scene_swap"] = torch.tensor("scene_swap" in demos)
        out["is_rand"] = torch.tensor("randomized" in ds.repos[query[0]].repo)
        out["task_index"] = torch.tensor(sorted(self.by_task).index(group[0]))
        return out


def collate(items):
    keys = set().union(*items)
    full = [k for k in keys if all(k in it for it in items)]
    out = default_collate([{k: it[k] for k in full} for it in items])
    # scene_swap BP exists only for some items: fill the others with their own normal BP (masked out later)
    for k in BP_KEYS:
        key = f"scene_swap/{k}"
        if key not in out and any(key in it for it in items):
            out[key] = torch.stack([it.get(key, it[k]) for it in items])
    return out


@torch.no_grad()
def evaluate(policy: BPV4XDemoPolicy, loader, task_names: list[str], device: torch.device) -> dict:
    model = policy.model
    sums = {c: [0.0, 0] for c in CONDITIONS}
    split = {c: {"clean": [0.0, 0], "rand": [0.0, 0]} for c in CONDITIONS}
    arm = {"same_arm": [0.0, 0], "diff_arm": [0.0, 0]}
    scene_normal = [0.0, 0]
    per_task = {}
    for bi, base in enumerate(loader):
        base = {k: v.to(device) for k, v in base.items()}
        b, s = base["query.progress"].shape
        n = b * s
        gen = torch.Generator().manual_seed(bi)
        noise = torch.randn((n, base[ACTION].shape[2], policy.config.max_action_dim), generator=gen).to(device)
        valid = base[ACTION_VALID_MASK].reshape(n, -1)
        rep = lambda x: x.repeat_interleave(s)  # noqa: E731
        is_rand, same_arm, has_scene = rep(base["is_rand"]), rep(base["pair.same_arm"]), rep(base["has_scene_swap"])
        task = rep(base["task_index"])
        for cond in CONDITIONS:
            if cond == "scene_swap" and not bool(has_scene.any()):
                continue
            batch, kwargs = dict(base), {}
            if cond in ("swap_same", "swap_wrong", "scene_swap"):
                batch.update({k: base[f"{cond}/{k}"] for k in BP_KEYS})
            elif cond == "no_bp":
                kwargs["bp_valid"] = torch.zeros(n, dtype=torch.bool, device=device)
            elif cond == "no_image":
                kwargs["image_valid"] = torch.zeros(n, dtype=torch.bool, device=device)
            per_query = torch.zeros(n, device=device)
            for t in TIMES:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    flow = model.forward_queries(batch, noise=noise, time=torch.full((n,), t, device=device),
                                                 **kwargs)["flow"]
                flow = flow[..., : base[ACTION].shape[-1]].float()
                w = valid[..., None].float()
                per_query += (flow * w).sum((1, 2)) / (w.sum((1, 2)) * flow.shape[-1]).clamp_min(1) / len(TIMES)
            sel = has_scene if cond == "scene_swap" else torch.ones_like(has_scene)
            sums[cond][0] += float(per_query[sel].sum())
            sums[cond][1] += int(sel.sum())
            for name, m in (("clean", ~is_rand & sel), ("rand", is_rand & sel)):
                split[cond][name][0] += float(per_query[m].sum())
                split[cond][name][1] += int(m.sum())
            if cond == "normal":
                for name, m in (("same_arm", same_arm), ("diff_arm", ~same_arm)):
                    arm[name][0] += float(per_query[m].sum())
                    arm[name][1] += int(m.sum())
                scene_normal[0] += float(per_query[has_scene].sum())
                scene_normal[1] += int(has_scene.sum())
            for ti in task.unique().tolist():
                m = task == ti
                rec = per_task.setdefault(task_names[ti], {}).setdefault(cond, [0.0, 0])
                rec[0] += float(per_query[m].sum())
                rec[1] += int(m.sum())
        print(f"batch {bi + 1}/{len(loader)}", flush=True)
    mean = lambda v: v[0] / max(1, v[1])  # noqa: E731
    loss = {c: mean(v) for c, v in sums.items()}
    return {
        "num_queries": sums["normal"][1],
        "loss": loss,
        "ratio_to_normal": {c: loss[c] / loss["normal"] for c in CONDITIONS if c != "scene_swap"}
        | {"scene_swap": mean(sums["scene_swap"]) / max(1e-12, mean(scene_normal))},
        "scene_pair_normal": mean(scene_normal),
        "loss_by_split": {c: {k: mean(v) for k, v in d.items()} for c, d in split.items()},
        "loss_by_arm": {k: mean(v) for k, v in arm.items()},
        "count_by_arm": {k: v[1] for k, v in arm.items()},
        "loss_by_task": {t: {c: mean(v) for c, v in d.items()} for t, d in sorted(per_task.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repo_file", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ckpt = Path(args.ckpt)
    train_cfg = json.loads((ckpt / "train_config.json").read_text())
    repo_file = args.repo_file or train_cfg["dataset"]["repo_id_file"]
    repos = [l.strip() for l in Path(repo_file).read_text().splitlines() if l.strip()]
    ds = eval_dataset(train_cfg, repos, split="val")
    pairs = EvalPairs(ds, args.limit)
    loader = torch.utils.data.DataLoader(pairs, batch_size=args.batch_size, num_workers=args.num_workers,
                                         collate_fn=collate)
    device = torch.device("cuda")
    policy = BPV4XDemoPolicy.from_pretrained(str(ckpt)).to(device).eval()
    result = evaluate(policy, loader, sorted(pairs.by_task), device)
    result["ckpt"] = str(ckpt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in ("loss", "ratio_to_normal", "loss_by_arm")}, indent=1))


if __name__ == "__main__":
    main()
