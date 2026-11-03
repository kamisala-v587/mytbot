"""B-2 pre-training checks (`bpv4cdx Stage B B200 checklist.md` section 0), on this machine.

  frozen       75w delivery: checksums (load_frozen), provisional false, sampler S-U v2
  paths        index <-> manifest sha256, every index repo resolves under the data roots, array store / ViT cache /
               Z_demo coverage
  pairing      every bucket > 3 trajectories; N random (query, demo) draws of the training dataset: identical
               description text (re-read from the repo's meta/episodes), i != j, same split
  forward      1 batch through bpv4cdx_b (forward + backward), peak memory; G9 monitor on a few held-out queries
Writes outputs/BPv4CDX/stage_b/b2_verify/verify_b2.{json,md}.

usage: source tools/bpv4cdxb/env.local.sh; CUDA_VISIBLE_DEVICES=0 python tools/bpv4cdxb/verify_b2.py \
         [--config configs/Pro6k/bpv4cdxb_plan/b2_main.jsonc] [--draws 2000] [--max_buckets 0]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

OUT = ROOT / "outputs/BPv4CDX/stage_b/b2_verify"


def check_frozen(policy_cfg) -> dict:
    from lerobot.policies.BPV4CDXB.frozen import describe, load_frozen

    f = load_frozen(policy_cfg.frozen_encoder)
    d = describe(f)
    ok = (not f.manifest.get("provisional")) and (f.manifest["sampler"]["version"] == "su-uniform_active_v2_embodiment")
    return {"pass": bool(ok), "tag": f.tag, **d}


def check_paths(cfg, index: dict, vit_cache: Path, tag: str) -> dict:
    from lerobot.policies.BPV4CDX.paths import resolve_rel
    from lerobot.policies.BPV4CDX.repo_data import store_dir

    import hashlib

    manifest_ok = hashlib.sha256(Path(index["manifest"]).read_bytes()).hexdigest() == index["manifest_sha256"]
    roots = cfg.data_root.split(":")
    unresolved, no_store, vit, zd, total = [], [], 0, 0, 0
    zroot = Path(cfg.zdemo_dir)
    for b_repo in index["repos"]:
        p = resolve_rel(b_repo["rel"], roots)
        if p is None:
            unresolved.append(b_repo["rel"])
            continue
        if not (store_dir(cfg.array_store, p) / "index.json").is_file():
            no_store.append(b_repo["rel"])
    for b in index["buckets"]:
        rel = index["repos"][b["repo"]]["rel"]
        eps = b["train"] + b["heldout"]
        total += len(eps)
        vdir, zdir = vit_cache / rel, zroot / rel
        vset = set(os.listdir(vdir)) if vdir.is_dir() else set()
        zset = set(os.listdir(zdir)) if zdir.is_dir() else set()
        vit += sum(f"ep{e:05d}.pt" in vset for e in eps)
        zd += sum(f"ep{e:05d}.pt" in zset for e in eps)
    zman = zroot / "manifest.json"
    zman_tag = json.loads(zman.read_text())["frozen"]["ckpt_sha256"][:12] if zman.is_file() else None
    return {"pass": bool(manifest_ok and not unresolved and zman_tag in (None, tag)),
            "manifest_sha256_match": manifest_ok, "repos": len(index["repos"]), "unresolved": unresolved[:10],
            "repos_without_array_store": len(no_store), "episodes": total,
            "vit_cache_coverage": round(vit / max(1, total), 4), "zdemo_coverage": round(zd / max(1, total), 4),
            "zdemo_manifest_tag": zman_tag}


def raw_text(repo_path: str, cache: dict) -> dict[int, str]:
    if repo_path not in cache:
        files = sorted(Path(repo_path, "meta", "episodes").rglob("*.parquet"))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "tasks"]) for f in files)
        cache[repo_path] = {int(e): (t if isinstance(t, str) else str(list(t)[0])) for e, t in zip(df.episode_index, df.tasks)}
    return cache[repo_path]


def check_pairing(ds, index: dict, draws: int) -> dict:
    sizes = [len(b["train"]) + len(b["heldout"]) for b in index["buckets"]]
    rng = np.random.default_rng(0)
    bad_text, same_ep, cross_split, cross = 0, 0, 0, 0
    cache: dict = {}
    torch.manual_seed(0)
    picks = rng.choice(len(ds.items), size=min(draws, len(ds.items)), replace=False)
    for i in picks:
        bi, ep, split = ds.items[int(i)]
        d, c = ds.pick_demo(bi, ep, split)
        cross += c
        b = ds.buckets[bi]
        same_ep += int(d == ep)
        cross_split += int(d not in b["eps"][split])
        texts = raw_text(ds.repos[b["repo"]].repo, cache)
        bad_text += int(texts[ep] != texts[d] or texts[ep] != b["text"])
    n = len(picks)
    return {"pass": bool(min(sizes) > 3 and bad_text == 0 and same_ep == 0 and cross_split == 0),
            "bucket_min_size": int(min(sizes)), "draws": int(n), "text_mismatch": bad_text, "same_episode": same_ep,
            "cross_split": cross_split, "cross_arm_draw_frac": round(cross / max(1, n), 4)}


def check_forward(ds, train_cfg) -> dict:
    from lerobot.policies.BPV4CDX.common import collate, to_device
    from lerobot.policies.BPV4CDXB.gate_g9 import evaluate, line, make_plan
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    train_cfg.policy.preflight_memory_check = False
    policy = BPV4CDXBPolicy(train_cfg.policy).to("cuda")
    policy.train()
    torch.manual_seed(0)
    idx = np.random.default_rng(1).choice(len(ds), size=int(train_cfg.batch_size), replace=False)
    t0 = time.time()
    batch = to_device(collate([ds[int(i)] for i in idx]))
    t_data = time.time() - t0
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, logs = policy(batch)
    loss.backward()
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    grads = sum(p.grad is not None for p in policy.parameters() if p.requires_grad)
    frozen_grad = sum(p.grad is not None for p in policy.model.core.backbone.vlm.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**30
    robots = sorted({ds.repos[ds.buckets[ds.items[int(i)][0]]["repo"]].robot_type for i in idx})
    plan = make_plan(ds, 8)
    g9 = evaluate(policy, ds, plan, batch_queries=4, num_workers=0)
    ok = bool(np.isfinite(float(loss)) and grads > 0 and frozen_grad == 0)
    return {"pass": ok, "loss": float(loss), "batch_pairs": len(idx), "robot_types_in_batch": robots,
            "data_seconds": round(t_data, 1), "peak_mem_gib": round(peak, 1), "trainable_params": int(trainable),
            "vlm_params_with_grad": int(frozen_grad), "g9_smoke": line(g9), "g9_cross_queries":
            sum(p["cross_arm"] is not None for p in plan)}


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV4CDXB.dataset_pretrain import BPV4CDXBPretrainDataset
    from lerobot.utils.jsonc import load_jsonc

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/Pro6k/bpv4cdxb_plan/b2_main.jsonc"))
    ap.add_argument("--vit_cache", default="/vla/workspace/cache/bpv4cdx_ae3_su")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--max_buckets", type=int, default=0)
    ap.add_argument("--skip_forward", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_dict = load_jsonc(args.config)
    cfg_dict["dataset"]["max_buckets"] = args.max_buckets
    cfg_dict["output_dir"] = str(OUT / "_unused")
    cfg_path = OUT / "_cfg_verify.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=1))
    cfg = TrainPipelineConfig.from_pretrained(str(cfg_path))
    index = json.loads(Path(cfg.dataset.index).read_text())

    result = {"config": args.config, "index": cfg.dataset.index, "time": time.strftime("%Y-%m-%d %H:%M")}
    result["frozen"] = check_frozen(cfg.policy)
    print("frozen", result["frozen"]["pass"], result["frozen"]["tag"], flush=True)
    result["paths"] = check_paths(cfg.dataset, index, Path(args.vit_cache), result["frozen"]["tag"])
    print("paths", json.dumps(result["paths"]), flush=True)
    t0 = time.time()
    ds = BPV4CDXBPretrainDataset(cfg.dataset, cfg, cfg.policy)
    result["dataset"] = {"query_episodes": len(ds), "buckets": len(ds.buckets), "repos": len(ds.repos),
                         "init_seconds": round(time.time() - t0, 1)}
    print("dataset", result["dataset"], flush=True)
    if len(ds):
        result["pairing"] = check_pairing(ds, index, args.draws)
        print("pairing", result["pairing"], flush=True)
        if not args.skip_forward:
            result["forward"] = check_forward(ds, cfg)
            print("forward", result["forward"], flush=True)
    gates = [result[k]["pass"] for k in ("frozen", "paths", "pairing", "forward") if k in result]
    result["pass"] = bool(gates and all(gates) and "forward" in result)
    (OUT / "verify_b2.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))
    md = ["## bpv4cdx Stage B — B-2 开训前核验", "", f"时间：{result['time']}；配置：`{args.config}`", "",
          "| 项 | 结果 | 关键数 |", "| --- | --- | --- |"]
    for k in ("frozen", "paths", "pairing", "forward"):
        if k in result:
            r = result[k]
            keys = [x for x in r if x != "pass"][:6]
            md.append(f"| {k} | {'PASS' if r['pass'] else 'FAIL'} | " + "；".join(f"{x}={r[x]}" for x in keys) + " |")
    md += ["", f"**总判定**：{'PASS' if result['pass'] else 'FAIL'}", "", f"原始 json：`{OUT / 'verify_b2.json'}`"]
    (OUT / "verify_b2.md").write_text("\n".join(md) + "\n")
    print("B-2 verify", "PASS" if result["pass"] else "FAIL", OUT / "verify_b2.md")


if __name__ == "__main__":
    main()
