"""B-1 smoke: one (query, demo) pair and one contrast pair (same anchor, two goals) overfit for a few hundred steps.

usage: source tools/bpv4cdxb/env.local.sh; CUDA_VISIBLE_DEVICES=2 python tools/bpv4cdxb/smoke_b1.py
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

import torch

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.BPV4CDX.common import collate, to_device
from lerobot.policies.BPV4CDXB.dataset import BPV4CDXBDataset
from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy, prefix_spans
from lerobot.utils.jsonc import load_jsonc


def main() -> None:
    index = json.loads(Path(os.environ["BPV4CDXB_INDEX"]).read_text())
    # one train-anchor pair for place_shoe_bp, and one contrast pair (two goals) of the same anchor
    train_eps = [
        e
        for e, m in index["episodes"].items()
        if index["split"][m["anchor"]] == "train" and m["task"] == "place_shoe_bp"
    ]
    # pick an anchor that has ≥2 candidate sets with train episodes
    anchors = sorted({index["episodes"][e]["anchor"] for e in train_eps})
    fixed = None
    for anchor in anchors:
        by_set: dict[str, list[str]] = {}
        for e in train_eps:
            if index["episodes"][e]["anchor"] != anchor:
                continue
            by_set.setdefault(index["episodes"][e]["set"], []).append(e)
        sets = [s for s in index["contrast"].get(anchor, []) if by_set.get(s)]
        if len(sets) < 2:
            continue
        q_a, q_b = by_set[sets[0]][0], by_set[sets[1]][0]
        # 2 contrast goals + 1 extra pair from the first set (or same query with another demo)
        fixed = [[q_a, index["matched"][q_a][0]["demo"]], [q_b, index["matched"][q_b][0]["demo"]]]
        extras = [e for e in by_set[sets[0]] if e != q_a]
        if extras:
            q_c = extras[0]
            fixed.append([q_c, index["matched"][q_c][0]["demo"]])
        elif len(index["matched"][q_a]) > 1:
            fixed.append([q_a, index["matched"][q_a][1]["demo"]])
        else:
            fixed.append([q_a, index["matched"][q_a][0]["demo"]])
        break
    assert fixed is not None, "no place_shoe_bp train anchor with ≥2 contrast sets"
    print("pairs", fixed)

    frozen = Path(os.environ["BPV4CDXB_FROZEN"])
    ztag = hashlib.sha256((frozen / "StageA_AE_frozen.ckpt").read_bytes()).hexdigest()[:12]
    zdemo = f"{os.environ['BPV4CDXB_ZDEMO']}/{ztag}"
    assert Path(zdemo, "manifest.json").is_file(), zdemo

    cfg_dict = load_jsonc(ROOT / "configs/Pro6k/bpv4cdxb_plan/b_main.jsonc")
    cfg_dict["dataset"].update(
        {
            "fixed_pairs": fixed,
            "all_frames": True,
            "contrast_prob": 0.0,
            "zdemo_dir": zdemo,
            "index": os.environ["BPV4CDXB_INDEX"],
            "data_root": os.environ["BPV4CDXB_DATA_ROOT"],
            "qwen3_5_processor_path": os.environ["BPV4CDXB_QWEN_PATH"],
        }
    )
    cfg_dict["policy"].update(
        {
            "frozen_encoder": str(frozen),
            "qwen3_5_pretrained_path": os.environ["BPV4CDXB_QWEN_PATH"],
            "preflight_memory_check": False,
        }
    )
    cfg_dict.update(
        {
            "batch_size": 3,
            "num_workers": 0,
            "steps": 400,
            "save_checkpoint": False,
            "log_freq": 50,
            "output_dir": str(ROOT / "outputs/BPv4CDX/stage_b/b1_smoke"),
        }
    )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg_dict, f)
        cfg_path = f.name
    cfg = TrainPipelineConfig.from_pretrained(cfg_path)

    ds = BPV4CDXBDataset(cfg.dataset, cfg, cfg.policy)
    policy = BPV4CDXBPolicy(cfg.policy).cuda()
    frozen_n = sum(p.numel() for n, p in policy.named_parameters() if not p.requires_grad and "vlm" in n)
    train = [(n, p.numel()) for n, p in policy.named_parameters() if p.requires_grad]
    print("trainable", [(n, c) for n, c in train[:8]], f"... ({len(train)} groups)", "frozen vlm params", frozen_n)
    assert abs(policy.model.demo_start).sum() > 0 and abs(policy.model.demo_end).sum() > 0

    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)
    losses = []
    for step in range(400):
        batch = to_device(collate([ds[i] for i in range(len(ds))]))
        if step == 0:
            pref, ids, valid = policy.model.build_prefix(batch)
            print(
                "prefix",
                tuple(pref.shape),
                "ids",
                tuple(ids.shape),
                "valid",
                tuple(valid.shape),
                "demo.z",
                tuple(batch["demo.z"].shape),
                "state",
                tuple(batch["observation.state"].shape),
                "action",
                tuple(batch["action"].shape),
            )
            assert pref.shape == ids.shape[:2] + (pref.shape[-1],)
            # obs_only mask: only image tokens visible to the expert
            policy.config.expert_visibility = "obs_only"
            m = policy.model._mask(valid, 50)
            sp = prefix_spans(policy.config, valid.shape[1])
            assert not m[0, 0, valid.shape[1], : sp["images"].start].any()
            assert m[0, 0, valid.shape[1], sp["images"]].all()
            policy.config.expert_visibility = "full"
        loss, logs = policy(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == 399:
            print(f"step {step}: loss={logs['loss_action']:.4f}")
    print(f"first {losses[0]:.4f} -> last {losses[-1]:.4f} ({100 * (1 - losses[-1] / losses[0]):.1f}% drop)")
    assert losses[-1] < 0.4 * losses[0], "B-1 smoke: loss did not drop enough"
    print("B-1 smoke PASS")


if __name__ == "__main__":
    main()
