"""Single-GPU smoke for BPv5 Stage A: load wall VLM + Q-Former, one real P5 cache, forward + a few train steps.

usage:
  cd /vla/workspace/my_tbot && source tools/bpv4cdxb/env.local.sh
  export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:$LD_LIBRARY_PATH
  CUDA_VISIBLE_DEVICES=3 python tools/bpv5/smoke_stage_a.py [--steps 3] [--cache_pt PATH]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerConfig, WALL_VLM
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import ACTION_DIM_MASK, BPV5QFormerPolicy
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

GENERIC = "reconstruct the demonstrated action segment"
DEFAULT_CACHE = ""


def make_batch(cache_pt: Path, tokenizer, *, segments: int, horizon: int, text_len: int, device: str) -> dict:
    rec = torch.load(cache_pt, map_location="cpu", weights_only=True)
    feats = rec["feats"]  # [32,3,5,2560] bf16
    time_ratio = rec["time_ratio"].float()
    camera_valid = rec["camera_valid"].bool()
    length = int(rec["length"])
    a = 80
    # synthetic normalized delta actions (full M3 data pipe comes later)
    action = torch.randn(1, segments, horizon, a)
    step_ok = torch.ones(1, segments, horizon, dtype=torch.bool)
    # leave a few trailing dims invalid to exercise dim mask
    dim_mask = torch.ones(a, dtype=torch.bool)
    dim_mask[-4:] = False
    state = torch.zeros(1, segments, a)
    starts = torch.linspace(0, max(0, length - horizon), segments).long()
    tau = torch.stack(
        [
            starts.float() / max(1, length - 1),
            (starts + horizon - 1).float() / max(1, length - 1),
        ],
        dim=-1,
    )[None]
    steps = torch.ones(1, segments)
    tok = tokenizer(
        [GENERIC] * segments,
        padding="max_length",
        truncation=True,
        max_length=text_len,
        return_tensors="pt",
    )
    batch = {
        "demo.feats": feats[None].to(device),
        "demo.time": time_ratio[None].to(device),
        "demo.camera_valid": camera_valid[None].to(device),
        ACTION: action.to(device),
        ACTION_VALID_MASK: step_ok.to(device),
        ACTION_DIM_MASK: dim_mask.to(device),
        OBS_STATE: state.to(device),
        "seg.tau": tau.to(device),
        "seg.steps": steps.to(device),
        "seg.start": starts[None].to(device),
        "text.input_ids": tok["input_ids"][None].to(device),
        "text.attention_mask": tok["attention_mask"][None].to(device),
        "text.is_generic": torch.ones(1, segments, dtype=torch.bool, device=device),
        "text.is_dropped": torch.zeros(1, segments, dtype=torch.bool, device=device),
    }
    return batch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default=WALL_VLM)
    ap.add_argument("--cache_pt", default=DEFAULT_CACHE)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--segments", type=int, default=2, help="fewer segments than 4 to save memory in smoke")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not args.cache_pt:
        raise SystemExit("pass --cache_pt pointing at a .pt under /vla/workspace/cache/bpv5_cleaned_p5")

    t0 = time.time()
    print(f"loading tokenizer + policy from {args.vlm}")
    tok = AutoTokenizer.from_pretrained(args.vlm)
    cfg = BPV5QFormerConfig(
        qwen3_5_pretrained_path=args.vlm,
        qwen3_5_processor_path=args.vlm,
        device=args.device,
        # smoke: keep memory down
        gradient_checkpointing=True,
        share_prefix=False,
    )
    policy = BPV5QFormerPolicy(cfg)
    policy.train()
    policy.to(args.device)
    print(
        f"ready in {time.time() - t0:.1f}s; fingerprint={cfg.qwen3_5_fingerprint}; "
        f"expert_hidden={policy.model.core.backbone.expert.config.hidden_size}"
    )

    batch = make_batch(
        Path(args.cache_pt),
        tok,
        segments=args.segments,
        horizon=int(cfg.chunk_size),
        text_len=48,
        device=args.device,
    )
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)

    for i in range(args.steps):
        opt.zero_grad(set_to_none=True)
        t1 = time.time()
        loss, logs = policy(batch)
        loss.backward()
        opt.step()
        print(f"step {i}: loss={float(loss.detach()):.4f} logs={logs} time={time.time() - t1:.1f}s")

    # prefix layout check: bos + 130 demo + 48 text + 1 sq = 180 when demo_input=video and prefix_bos
    with torch.no_grad():
        z = policy.model.encode_demo(batch)
        prefix, valid = policy.model.build_prefix(batch, z)
    expect = 1 + 1 + 128 + 1 + 48 + 1  # bos, demo_start, Z, demo_end, text, sq
    print(f"prefix shape={tuple(prefix.shape)} valid_sum={int(valid.sum())} expect_len={expect}")
    assert prefix.shape[1] == expect, f"prefix length {prefix.shape[1]} != {expect}"
    print(f"smoke ok in {time.time() - t0:.1f}s total")


if __name__ == "__main__":
    main()
