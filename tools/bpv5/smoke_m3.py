"""M3 smoke: real 80-D delta Dataset + BPv5QFormer train step(s). Supports 1–2 GPUs via torchrun.

examples:
  # dataset-only check (CPU)
  python tools/bpv5/smoke_m3.py --dataset_only --max_trajectories 4

  # single GPU
  CUDA_VISIBLE_DEVICES=0 python tools/bpv5/smoke_m3.py --steps 2 --batch 2 --max_trajectories 8

  # dual GPU
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 tools/bpv5/smoke_m3.py \\
    --steps 2 --batch 2 --max_trajectories 16
"""

from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from lerobot.policies.BPV5QFormer.configuration_bpv5qformer import BPV5QFormerConfig, BPV5QFormerDatasetConfig, WALL_VLM
from lerobot.policies.BPV5QFormer.dataset import BPV5QFormerDataset, collate_bpv5
from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import BPV5QFormerPolicy
from lerobot.utils.constants import ACTION


def setup_dist():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        return rank, world, local
    return 0, 1, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", default=WALL_VLM)
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv5_cleaned_p5")
    ap.add_argument("--sources", nargs="*", default=["robotwin"])
    ap.add_argument("--repo_ids", nargs="*", default=[])
    ap.add_argument("--max_trajectories", type=int, default=16)
    ap.add_argument("--batch", type=int, default=2, help="trajectories per GPU")
    ap.add_argument("--segments", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--dataset_only", action="store_true")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    rank, world, local = setup_dist()
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    is_main = rank == 0

    dcfg = BPV5QFormerDatasetConfig(
        repo_id="bpv5_80d",  # required by DatasetConfig; real repos come from sources / repo_ids
        qwen3_5_processor_path=args.vlm,
        cache_dir=args.cache_dir,
        sources=list(args.sources),
        repo_ids=list(args.repo_ids),
        max_trajectories=int(args.max_trajectories),
        segments_per_trajectory=int(args.segments),
        segment_mode="fixed",
        delta_mode="bpv5_80d",
        generic_text_prob=0.5,
        length_weighted_sampling=False,
    )
    ds = BPV5QFormerDataset(dcfg, chunk_size=50)
    if is_main:
        print(f"dataset: {len(ds)} trajectories, sources={args.sources or args.repo_ids}")
        # text / delta spot-check
        for i in range(min(2, len(ds))):
            item = ds[i]
            rel, ep_idx, _ = ds.items[i]
            print(
                f"  [{i}] rel={rel} ep={ep_idx} "
                f"action={tuple(item[ACTION].shape)} dim_mask={int(item['action.dim_mask'].sum())} "
                f"generic={item['text.is_generic'].tolist()}"
            )

    if args.dataset_only:
        if world > 1:
            dist.destroy_process_group()
        return

    sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True) if world > 1 else None
    loader = DataLoader(
        ds,
        batch_size=int(args.batch),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(args.workers),
        collate_fn=collate_bpv5,
        drop_last=True,
    )
    if len(loader) == 0:
        raise RuntimeError("empty loader: lower --batch or raise --max_trajectories")

    cfg = BPV5QFormerConfig(
        qwen3_5_pretrained_path=args.vlm,
        qwen3_5_processor_path=args.vlm,
        device=device,
        gradient_checkpointing=True,
    )
    t0 = time.time()
    policy = BPV5QFormerPolicy(cfg).to(device).train()
    if is_main:
        print(f"policy ready in {time.time() - t0:.1f}s fingerprint={cfg.qwen3_5_fingerprint}")
    model = DDP(policy, device_ids=[local]) if world > 1 and device.startswith("cuda") else policy
    opt = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=1e-4)

    it = iter(loader)
    for step in range(int(args.steps)):
        if sampler is not None:
            sampler.set_epoch(step)
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        t1 = time.time()
        loss, logs = model(batch)
        loss.backward()
        opt.step()
        if is_main:
            print(
                f"step {step}: loss={float(loss.detach()):.4f} "
                f"B={args.batch}x{world} S={args.segments} N={args.batch * world * args.segments} "
                f"time={time.time() - t1:.1f}s logs={logs}"
            )

    if world > 1:
        dist.destroy_process_group()
    if is_main:
        print("smoke_m3 ok")


if __name__ == "__main__":
    main()
