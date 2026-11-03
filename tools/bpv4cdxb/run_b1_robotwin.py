"""B-1 on local RoboTwin: same-task index → Z_demo → overfit → eval → report.

Uses stack_blocks_two/aloha-agilex_clean_50 episodes that share an identical task string
(ep 8, 41, 46). P5 cache: AE-3 S-U v2 under bpv4cdxb_su_b1 (builds missing).

usage: source tools/bpv4cdxb/env.local.sh; CUDA_VISIBLE_DEVICES=2 python tools/bpv4cdxb/run_b1_robotwin.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

OUT = ROOT / "outputs/BPv4CDX/stage_b/b1_robotwin"
DATA = Path("/vla/workspace/data/robotwin2.0")
TASK = "stack_blocks_two"
VARIANT = "aloha-agilex_clean_50"
REPO = f"{TASK}/{VARIANT}"
# same task text group that includes a held-out cached ep (46)
SAME_TASK_EPS = [8, 41, 46]
FROZEN = Path(os.environ["BPV4CDXB_FROZEN"])
DEMO_CACHE = Path("/vla/workspace/cache/bpv4cdxb_su_b1")
ZDEMO_ROOT = Path(os.environ.get("BPV4CDXB_ZDEMO", "/vla/workspace/cache/bpv4cdxb_zdemo"))


def task_text(ep: int) -> str:
    root = DATA / REPO
    df = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    row = df[df.episode_index == ep].iloc[0]
    t = row["tasks"]
    if hasattr(t, "tolist"):
        t = t.tolist()
    return t[0] if isinstance(t, (list, tuple)) else str(t)


def ep_length(ep: int) -> int:
    df = pd.read_parquet(DATA / REPO / "meta/episodes/chunk-000/file-000.parquet")
    return int(df[df.episode_index == ep].iloc[0]["length"])


def ensure_same_task() -> str:
    texts = {e: task_text(e) for e in SAME_TASK_EPS}
    assert len(set(texts.values())) == 1, texts
    print("same task text:", next(iter(texts.values()))[:120])
    return next(iter(texts.values()))


def build_index(text: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    episodes = {}
    matched = {}
    for e in SAME_TASK_EPS:
        key = f"{REPO}|{e}"
        episodes[key] = {
            "repo": REPO,
            "episode": e,
            "task": TASK,
            "task_text": text,
            "set": f"{REPO}/{text[:40]}",
            "goal": "same_text",
            "anchor": f"{REPO}/same_text",
            "arm": "unknown",
            "length": ep_length(e),
            "jitter": 0,
        }
    keys = [f"{REPO}|{e}" for e in SAME_TASK_EPS]
    for q in keys:
        matched[q] = [{"demo": d, "shift_cm": 8.0} for d in keys if d != q]
    # one artificial "contrast" using two queries of the same text (swap-follow interface)
    contrast = {f"{REPO}/same_text": [f"{REPO}/{text[:40]}"]}
    index = {
        "root": str(DATA),
        "min_shift_cm": 0.0,
        "div_threshold": 0.2,
        "episodes": episodes,
        "matched": matched,
        "contrast": contrast,
        "equivalent": {},
        "sets": {f"{REPO}/{text[:40]}": keys},
        "split": {f"{REPO}/same_text": "train"},
        "divergence": {f"{REPO}/same_text": {"t_div": 40, "tau_div": 0.2, "goals": 1}},
        "checks": {"pass": True, "note": "b1_robotwin same-task subset"},
    }
    path = OUT / "b1_robotwin_index.json"
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print("wrote", path)
    return path


def ensure_p5_cache() -> None:
    missing = [e for e in SAME_TASK_EPS if not (DEMO_CACHE / REPO / f"ep{e:05d}.pt").is_file()]
    # also accept copy from ae3 heldout cache
    ae3 = Path("/vla/workspace/cache/bpv4cdx_ae3_su/RoboTwin-LeRobot-v3.0") / REPO
    DEMO_CACHE.mkdir(parents=True, exist_ok=True)
    (DEMO_CACHE / REPO).mkdir(parents=True, exist_ok=True)
    for e in SAME_TASK_EPS:
        dst = DEMO_CACHE / REPO / f"ep{e:05d}.pt"
        src = ae3 / f"ep{e:05d}.pt"
        if not dst.is_file() and src.is_file():
            dst.write_bytes(src.read_bytes())
            print("copied ae3 cache", src.name)
    missing = [e for e in SAME_TASK_EPS if not (DEMO_CACHE / REPO / f"ep{e:05d}.pt").is_file()]
    if not missing:
        print("P5 cache ready", DEMO_CACHE / REPO)
        return
    print("building P5 for", missing)
    repo_file = OUT / "b1_repos.txt"
    repo_file.write_text(REPO + "\n")
    # build whole repo (50 eps) is OK for one task; use embodiment S-U
    cmd = (
        f"python -m lerobot.policies.BPV4CDX.build_cache "
        f"--repo_file {repo_file} --data_root {DATA} --cache_dir {DEMO_CACHE} "
        f"--sampler su --embodiment_sampler --store_spatial 5 "
        f"--qwen_path {os.environ['BPV4CDXB_QWEN_PATH']} --split all --episodes_per_batch 2 --num_workers 4"
    )
    print(cmd)
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"build_cache failed rc={rc}")
    still = [e for e in SAME_TASK_EPS if not (DEMO_CACHE / REPO / f"ep{e:05d}.pt").is_file()]
    if still:
        raise RuntimeError(f"still missing cache {still}")


def ensure_zdemo(index_path: Path) -> Path:
    from lerobot.policies.BPV4CDXB.frozen import load_frozen
    from lerobot.policies.BPV4CDXB.zdemo_cache import encode, load_demo, zdemo_path

    frozen = load_frozen(FROZEN, "cuda")
    tag = frozen.tag
    spatial = int(frozen.manifest["vit"]["spatial_tokens_used"])
    want = frozen.manifest["sampler"]["version"]
    out = ZDEMO_ROOT
    items = []
    keys = []
    for e in SAME_TASK_EPS:
        dest = zdemo_path(out, tag, REPO, e)
        if dest.is_file():
            print("zdemo exists", dest)
            continue
        d = load_demo(DEMO_CACHE, REPO, e, spatial)
        if d["version"] != want:
            raise RuntimeError(f"ep{e} sampler {d['version']} != {want}")
        items.append(d)
        keys.append(e)
    if items:
        z = encode(frozen.encoder, items, "cuda")
        for e, zi in zip(keys, z):
            dest = zdemo_path(out, tag, REPO, e)
            dest.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"z_demo": zi.cpu().to(torch.bfloat16), "episode": e, "repo": REPO}, dest)
            print("wrote", dest)
    man = out / tag / "manifest.json"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({"frozen_tag": tag, "sampler": want, "b1_robotwin": SAME_TASK_EPS}, indent=2))
    return out / tag


def make_cfg(index_path: Path, zdemo: Path, fixed, steps: int, name: str):
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.jsonc import load_jsonc

    cfg_dict = load_jsonc(ROOT / "configs/Pro6k/bpv4cdxb_plan/b_main.jsonc")
    cfg_dict["dataset"].update(
        {
            "fixed_pairs": fixed,
            "queries_per_item": 8,
            "contrast_prob": 0.0,
            "all_frames": False,
            "zdemo_dir": str(zdemo),
            "index": str(index_path),
            "data_root": str(DATA),
            "qwen3_5_processor_path": os.environ["BPV4CDXB_QWEN_PATH"],
            # avoid factory→TBot_SA1→mediapy/sqlite import crash in this env; use dataset-local stats
            "use_external_stats": False,
        }
    )
    cfg_dict["policy"].update(
        {
            "frozen_encoder": str(FROZEN),
            "qwen3_5_pretrained_path": os.environ["BPV4CDXB_QWEN_PATH"],
            "preflight_memory_check": False,
            "new_modules_lr": 5e-5,
            "act_expert_lr": 5e-5,
        }
    )
    cfg_dict.update(
        {
            "batch_size": max(1, len(fixed)),
            "num_workers": 0,
            "steps": steps,
            "save_checkpoint": False,
            "output_dir": str(OUT / name),
            "seed": 42,
        }
    )
    cfg_path = OUT / f"_cfg_{name}.json"
    OUT.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(cfg_dict, indent=2))
    return TrainPipelineConfig.from_pretrained(str(cfg_path))


def train(policy, ds, steps):
    from lerobot.policies.BPV4CDX.common import collate, to_device

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=5e-5)
    losses = []
    for step in range(steps):
        batch = to_device(collate([ds[i] for i in range(len(ds))]))
        loss, logs = policy(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 100 == 0 or step == steps - 1:
            print(f"  step {step}: loss={logs['loss_action']:.4f}")
    return losses


@torch.no_grad()
def predict(policy, batch, seed=0):
    from lerobot.utils.constants import ACTION

    n = batch[ACTION].shape[0] * batch[ACTION].shape[1]
    h, dim = int(policy.config.chunk_size), int(policy.config.max_action_dim)
    g = torch.Generator(device="cuda").manual_seed(seed)
    noise = torch.randn((n, h, dim), generator=g, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return policy.predict_action_chunk(batch, noise=noise).float()


def eval_fit(policy, ds, frames, seeds=3):
    from lerobot.policies.BPV4CDX.common import collate, masked_error, to_device
    from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
    from lerobot.utils.constants import ACTION

    items = [ds.build(q, d, frames) for q, d in ds.items]
    batch = to_device(collate(items))
    target, valid = batch[ACTION], batch[ACTION_VALID_MASK].bool()
    b, s, h, a = target.shape
    preds = [predict(policy, batch, seed).reshape(b, s, h, -1)[..., :a] for seed in range(seeds)]
    err = torch.stack(
        [masked_error(p.reshape(b * s, h, a), target.reshape(b * s, h, a), valid.reshape(b * s, h)) for p in preds]
    )
    w = valid.reshape(b * s, h, 1).float()
    tgt = target.reshape(b * s, h, a)
    mean_action = (tgt * w).sum((0, 1)) / w.sum((0, 1)).clamp_min(1)
    base = masked_error(mean_action.expand_as(tgt), tgt, valid.reshape(b * s, h))
    reduction = 1.0 - float(err.mean() / base.mean().clamp_min(1e-8))
    return {
        "error_mean": float(err.mean()),
        "baseline_mean_action_error": float(base.mean()),
        "reduction_vs_mean_action": reduction,
        "pass_fit": bool(reduction >= 0.8),
        "per_pair_error": [
            float(masked_error(preds[0][i], target[i], valid[i]).mean()) for i in range(b)
        ],
    }


def eval_swap(policy, ds, frames):
    """Same-task swap follow: for q1, demo of q2 should move pred toward gt2."""
    from lerobot.policies.BPV4CDX.common import collate, masked_error, to_device
    from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
    from lerobot.utils.constants import ACTION

    assert len(ds) >= 2
    (q1, d1), (q2, d2) = ds.items[0], ds.items[1]
    b11 = to_device(collate([ds.build(q1, d1, frames)]))
    b12 = to_device(collate([ds.build(q1, d2, frames)]))
    b22 = to_device(collate([ds.build(q2, d2, frames)]))
    gt1, v1 = b11[ACTION][0], b11[ACTION_VALID_MASK][0].bool()
    gt2, v2 = b22[ACTION][0], b22[ACTION_VALID_MASK][0].bool()
    s, h, a = gt1.shape
    p11 = predict(policy, b11).reshape(s, h, -1)[..., :a]
    p12 = predict(policy, b12).reshape(s, h, -1)[..., :a]
    p22 = predict(policy, b22).reshape(s, h, -1)[..., :a]
    m = v1 & v2
    follow = masked_error(p12, gt2, m) < masked_error(p12, gt1, m)
    return {
        "follow_rate": float(follow.float().mean()),
        "pass_follow": bool(float(follow.float().mean()) >= 0.9),
        "matched_err": {"q1": float(masked_error(p11, gt1, v1).mean()), "q2": float(masked_error(p22, gt2, v2).mean())},
        "swapped_err_to_own": float(masked_error(p12, gt1, v1).mean()),
        "swapped_err_to_other": float(masked_error(p12, gt2, m).mean()),
        "note": "same-task swap (RoboTwin identical task text); cross-arm not available in this subset",
    }


def main() -> None:
    assert torch.cuda.is_available()
    OUT.mkdir(parents=True, exist_ok=True)
    text = ensure_same_task()
    index_path = build_index(text)
    ensure_p5_cache()
    zdemo = ensure_zdemo(index_path)

    keys = [f"{REPO}|{e}" for e in SAME_TASK_EPS]
    # single: query 8, demo 41
    single_pairs = [[keys[0], keys[1]]]
    # contrast: two queries same text
    contrast_pairs = [[keys[0], keys[1]], [keys[2], keys[0]]]

    from lerobot.policies.BPV4CDXB.dataset import BPV4CDXBDataset
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy

    print("=== B-1 single (RoboTwin same-task) ===")
    cfg1 = make_cfg(index_path, zdemo, single_pairs, 1000, "single")
    ds1 = BPV4CDXBDataset(cfg1.dataset, cfg1, cfg1.policy)
    pol1 = BPV4CDXBPolicy(cfg1.policy).cuda()
    losses1 = train(pol1, ds1, 1000)
    frames = [int(x) for x in np.linspace(0, max(0, ep_length(SAME_TASK_EPS[0]) - 2), 16).round()]
    single = eval_fit(pol1, ds1, frames)
    print("single", single)

    print("=== B-1 contrast / swap (same task text) ===")
    cfg2 = make_cfg(index_path, zdemo, contrast_pairs, 2000, "contrast")
    ds2 = BPV4CDXBDataset(cfg2.dataset, cfg2, cfg2.policy)
    pol2 = BPV4CDXBPolicy(cfg2.policy).cuda()
    losses2 = train(pol2, ds2, 2000)
    frames2 = [int(x) for x in np.linspace(0, max(0, min(ep_length(e) for e in SAME_TASK_EPS) - 2), 12).round()]
    contrast_fit = eval_fit(pol2, ds2, frames2)
    swap = eval_swap(pol2, ds2, frames2)
    print("contrast", contrast_fit)
    print("swap", swap)

    result = {
        "data": {"root": str(DATA), "repo": REPO, "episodes": SAME_TASK_EPS, "task_text": text},
        "frozen": str(FROZEN),
        "zdemo": str(zdemo),
        "single": {**single, "loss_first": losses1[0], "loss_last": losses1[-1]},
        "contrast_fit": {**contrast_fit, "loss_first": losses2[0], "loss_last": losses2[-1]},
        "swap_follow": swap,
        "pass": bool(single["pass_fit"] and contrast_fit["pass_fit"]),
        "pass_strict_follow": bool(swap["pass_follow"]),
        "thresholds": {"reduction": 0.8, "follow_rate": 0.9},
    }
    # report markdown
    report = OUT / "b1_robotwin_report.md"
    report.write_text(
        "\n".join(
            [
                "## bpv4cdx Stage B — B-1 测评报告（RoboTwin）",
                "",
                f"日期：2026-10-26。数据：`{DATA}/{REPO}`，同任务文本 episode {SAME_TASK_EPS}。",
                f"冻结件：`{FROZEN}`。学习率 5e-5。",
                "",
                "### 任务文本",
                "",
                f"> {text}",
                "",
                "### 结果",
                "",
                "| 检查 | 指标 | 门槛 | 结果 |",
                "| --- | --- | --- | --- |",
                f"| 单配对拟合 | reduction {single['reduction_vs_mean_action']:.3f} | ≥ 0.80 | {'PASS' if single['pass_fit'] else 'FAIL'} |",
                f"| 单配对 loss | {losses1[0]:.3f} → {losses1[-1]:.3f} | 下降 | — |",
                f"| 对比对拟合 | reduction {contrast_fit['reduction_vs_mean_action']:.3f} | ≥ 0.80 | {'PASS' if contrast_fit['pass_fit'] else 'FAIL'} |",
                f"| 同任务换示教跟随率 | {swap['follow_rate']:.3f} | ≥ 0.90 | {'PASS' if swap['pass_follow'] else 'FAIL'} |",
                f"| 换示教误差 own → other | {swap['swapped_err_to_own']:.4f} → {swap['swapped_err_to_other']:.4f} | other 应更小 | — |",
                "",
                "### 判定",
                "",
                f"- **接口拟合（B-1 主门槛）**：{'通过' if result['pass'] else '未通过'}（单配对 + 双 query 同任务拟合）。",
                f"- **换示教跟随**：{'通过' if result['pass_strict_follow'] else '未通过'}（本子集无跨臂对；为同任务文本换示教跟随）。",
                "- 跨臂：本 RoboTwin 子集无「同描述跨臂」对，记为数据缺口，不记失败。",
                "",
                f"原始 json：`{OUT / 'b1_robotwin_result.json'}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT / "b1_robotwin_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print("wrote", report)
    print("B-1 RoboTwin", "PASS" if result["pass"] else "FAIL", "follow", swap["follow_rate"])


if __name__ == "__main__":
    main()
