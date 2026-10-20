"""Build the 32-slot ViT feature cache for one sampler (S-A / S-U / S-V); multi-GPU shards, resumable.

Same file layout and record format as the bpv4_bpen cache (`<cache>/<task>/<variant>/ep{ep:05d}.pt`, feats
[32, 3, 64, 1024] bf16 from the Qwen3.5-0.8B base ViT), plus `sampler` and `repeats`. A bpv4_bpen cache (S-A, K=32,
alpha=0.5) is therefore directly usable as an S-A cache.

S-A / S-U: DataLoader workers decode the slot frames and run the Qwen image processor; the main process runs the ViT.
S-V: every frame goes through the ViT first (per-frame proxy), then the 32 slots are chosen; slow, main process only.

Usage (8 GPUs):
  for r in 0 1 2 3 4 5 6 7; do CUDA_VISIBLE_DEVICES=$r python -m lerobot.policies.BPV4CDX.build_cache \
      --repo_file <txt> --data_root <dir> --cache_dir <dir> --sampler sa --qwen_path <base_800m> \
      --rank $r --world 8 --num_workers 12 & done; wait
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import CAMERAS, cache_path, load_states, load_vit, visual_tokens
from lerobot.transforms.utils import resize_with_pad

from .paths import read_repo_list
from .sampling import sample


class SlotFrames(torch.utils.data.Dataset):
    """One item per episode: processor outputs for its 32 slot frames x 3 cameras (S-A / S-U)."""

    def __init__(self, items, sampler: str, num_slots: int, alpha: float, size: int, qwen_path: str) -> None:
        self.items, self.sampler, self.k, self.alpha, self.size, self.qwen_path = items, sampler, num_slots, alpha, size, qwen_path
        self._ds: dict[str, object] = {}
        self._states: dict[str, dict[int, np.ndarray]] = {}
        self._processor = None

    def __len__(self) -> int:
        return len(self.items)

    def _repo(self, repo: str):
        if repo not in self._ds:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._ds = {repo: LeRobotDataset(repo, video_backend="torchcodec")}  # one open repo per worker
            self._states = {repo: load_states(repo)}
        return self._ds[repo], self._states[repo]

    def __getitem__(self, i: int):
        repo, ep = self.items[i]
        ds, states = self._repo(repo)
        plan = sample(self.sampler, states[ep], self.k, self.alpha)
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        images = []
        for f in plan.frames:
            item = ds[start + int(f)]
            images.extend(resize_with_pad(item[c][None], self.size, self.size)[0] for c in CAMERAS)
        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            torch.set_num_threads(2)
            self._processor, _ = load_qwen35_vision(self.qwen_path)
        out = self._processor(images, do_rescale=False)
        return {"repo": repo, "episode": ep, "plan": plan, "pixels": torch.as_tensor(out.pixel_values),
                "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long)}


def record(plan, feats: torch.Tensor, qwen_path: str) -> dict:
    k, c = len(plan.frames), len(CAMERAS)
    return {
        "frames": torch.as_tensor(plan.frames, dtype=torch.long),
        "forced": torch.as_tensor(np.isin(plan.frames, plan.flips), dtype=torch.bool),
        "flips": torch.as_tensor(plan.flips, dtype=torch.long),
        "active": tuple(int(x) for x in plan.active),
        "length": int(plan.length),
        "time_ratio": torch.as_tensor(plan.frames / max(1, plan.length - 1), dtype=torch.float32),
        "cameras": CAMERAS,
        "vit_source": qwen_path,
        "sampler": plan.sampler,
        "repeats": int(plan.repeats),
        "feats": feats.reshape(k, c, feats.shape[-2], feats.shape[-1]).to(torch.bfloat16).cpu(),
    }


def save(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(rec, tmp)
    os.replace(tmp, path)


def build_sv(items, args, vlm, device) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

    processor, _ = load_qwen35_vision(args.qwen_path)
    opened: dict[str, tuple] = {}
    for n, (repo, ep) in enumerate(items, 1):
        if repo not in opened:
            opened = {repo: (LeRobotDataset(repo, video_backend="torchcodec"), load_states(repo))}
        ds, states = opened[repo]
        start, end = int(ds.meta.episodes[ep]["dataset_from_index"]), int(ds.meta.episodes[ep]["dataset_to_index"])
        tokens = []
        for s in range(start, end, 16):
            images = []
            for i in range(s, min(end, s + 16)):
                item = ds[i]
                images.extend(resize_with_pad(item[c][None], args.image_size, args.image_size)[0] for c in CAMERAS)
            proc = processor(images, do_rescale=False)
            with torch.no_grad():
                t = visual_tokens(vlm, torch.as_tensor(proc.pixel_values).to(device, torch.bfloat16),
                                  torch.as_tensor(proc.image_grid_thw, dtype=torch.long).to(device))
            tokens.append(t.reshape(-1, len(CAMERAS), t.shape[-2], t.shape[-1]).cpu())
        tokens = torch.cat(tokens)  # [T, 3, 64, D]
        proxy = tokens.float().mean(2).reshape(len(tokens), -1).numpy()
        plan = sample("sv", states[ep], args.num_slots, args.alpha, frame_features=proxy)
        feats = tokens[torch.as_tensor(plan.frames)].reshape(-1, tokens.shape[-2], tokens.shape[-1])
        save(cache_path(args.cache_dir, repo, ep), record(plan, feats, args.qwen_path))
        if n % 20 == 0:
            logging.info("%d / %d", n, len(items))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--sampler", choices=["sa", "su", "sv"], required=True)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--num_slots", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending episodes (testing)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"[rank {args.rank}] %(asctime)s %(message)s")

    items = []
    for repo in read_repo_list(args.repo_file, args.data_root):
        n = int(pd.read_json(Path(repo, "meta", "info.json"), typ="series")["total_episodes"])
        items += [(repo, ep) for ep in range(n) if not cache_path(args.cache_dir, repo, ep).is_file()]
    items = items[args.rank :: args.world]
    if args.limit:
        items = items[: args.limit]
    logging.info("sampler %s, pending episodes: %d", args.sampler, len(items))
    if not items:
        return
    device = torch.device("cuda")
    vlm = load_vit(None, args.qwen_path, device)
    if args.sampler == "sv":
        build_sv(items, args, vlm, device)
        logging.info("done")
        return

    # contiguous block per worker, so each worker opens few repos
    workers = max(1, args.num_workers)
    blocks = [items[i * len(items) // workers : (i + 1) * len(items) // workers] for i in range(workers)]
    items = [blocks[i % workers][i // workers] for i in range(workers * max(map(len, blocks))) if i // workers < len(blocks[i % workers])]
    loader = torch.utils.data.DataLoader(
        SlotFrames(items, args.sampler, args.num_slots, args.alpha, args.image_size, args.qwen_path),
        batch_size=None, num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None,
        # the ViT is already on CUDA in this process; forked workers running the image processor segfault
        multiprocessing_context="spawn" if args.num_workers else None,
    )
    for n, item in enumerate(loader, 1):
        with torch.no_grad():
            feats = visual_tokens(vlm, item["pixels"].to(device, torch.bfloat16), item["grids"].to(device))
        save(cache_path(args.cache_dir, item["repo"], item["episode"]), record(item["plan"], feats, args.qwen_path))
        if n % 100 == 0:
            logging.info("%d / %d", n, len(items))
    logging.info("done")


if __name__ == "__main__":
    main()
