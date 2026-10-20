"""Pre-compute frozen ViT features for the adaptive keyframes of every episode.

One file per episode: <cache_dir>/<task>/<variant>/ep{index:05d}.pt with
  frames [K_eff] int, forced [K_eff] bool, flips [n] int, active (lo, hi), length,
  time_ratio [K_eff] float (frame / (length - 1)), cameras list,
  feats [K_eff, C, T, D] bfloat16 (T visual tokens per image after the Qwen3.5 merger).
The ViT is the original Qwen3.5 base ViT (no VLA weights). Pass --init_ckpt only to override it with a
policy checkpoint's ViT. Each file records `vit_source`.

Usage (one process per GPU, resumable):
  python -m lerobot.policies.BPV4BPEn.build_feature_cache --repo_file <txt> --cache_dir <dir> \
      --rank 0 --world 4
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4BPEn.keyframes import plan_keyframes
from lerobot.transforms.utils import resize_with_pad

CAMERAS = ["observation.images.cam_high", "observation.images.cam_left_wrist", "observation.images.cam_right_wrist"]


def cache_path(cache_dir: str | Path, repo: str | Path, episode: int) -> Path:
    repo = Path(repo)
    return Path(cache_dir) / repo.parent.name / repo.name / f"ep{int(episode):05d}.pt"


def load_states(repo: str | Path) -> dict[int, np.ndarray]:
    files = sorted(Path(repo, "data").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "observation.state"]) for f in files)
    return {int(ep): np.stack(g["observation.state"].to_numpy()).astype(np.float32) for ep, g in df.groupby("episode_index")}


class EpisodeFrames(torch.utils.data.Dataset):
    """Decodes the keyframe images of one episode per item (all cameras, resized with pad) and runs the Qwen3.5
    image processor, so the main process only runs the ViT."""

    def __init__(self, items: list[tuple[str, int]], num_keyframes: int, alpha: float, size: int, qwen_path: str) -> None:
        self.items, self.k, self.alpha, self.size, self.qwen_path = items, num_keyframes, alpha, size, qwen_path
        self._ds: dict[str, object] = {}
        self._states: dict[str, dict[int, np.ndarray]] = {}
        self._processor = None

    def __len__(self) -> int:
        return len(self.items)

    def _repo(self, repo: str):
        if repo not in self._ds:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            self._ds = {repo: LeRobotDataset(repo, video_backend="torchcodec")}  # keep one repo open per worker
            self._states = {repo: load_states(repo)}
        return self._ds[repo], self._states[repo]

    def __getitem__(self, i: int):
        repo, ep = self.items[i]
        ds, states = self._repo(repo)
        plan = plan_keyframes(states[ep], self.k, self.alpha)
        start = int(ds.meta.episodes[ep]["dataset_from_index"])
        images = []
        for f in plan.frames:
            sample = ds[start + int(f)]
            images.extend(resize_with_pad(sample[c][None], self.size, self.size)[0] for c in CAMERAS)
        if self._processor is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            torch.set_num_threads(2)
            self._processor, _ = load_qwen35_vision(self.qwen_path)
        out = self._processor(images, do_rescale=False)
        return {
            "repo": repo,
            "episode": ep,
            "pixels": torch.as_tensor(out.pixel_values),
            "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long),
            "num_frames": len(plan.frames),
            "plan": plan,
        }


def load_vit(init_ckpt: str | None, base_path: str, device: torch.device):
    from safetensors import safe_open
    from transformers import Qwen3_5ForConditionalGeneration

    vlm = Qwen3_5ForConditionalGeneration.from_pretrained(base_path, dtype=torch.bfloat16)
    if init_ckpt:
        prefix = "model.core.backbone.vlm.model.visual."
        with safe_open(str(Path(init_ckpt) / "model.safetensors"), "pt") as f:
            state = {key[len(prefix):]: f.get_tensor(key) for key in f.keys() if key.startswith(prefix)}
        vlm.model.visual.load_state_dict(state, strict=True)
    vlm.model.visual.to(device).eval()
    return vlm


def visual_tokens(vlm, pixels: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """[N images] -> [N, T, D]; same call path as BPVAv31Model._visual_features."""
    chunks = vlm.model.get_image_features(pixels, grids).pooler_output
    if not isinstance(chunks, (tuple, list)):
        merge = int(vlm.model.visual.spatial_merge_size)
        lengths = [int(v) for v in (grids.prod(-1) // merge**2).tolist()]
        chunks = torch.split(chunks, lengths)
    return torch.stack(list(chunks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_file", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--init_ckpt", default=None, help="optional policy checkpoint whose ViT overrides the base ViT")
    parser.add_argument("--qwen_path", default="/vla/workspace/models/q35/base_800m")
    parser.add_argument("--num_keyframes", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="only process the first N pending episodes (testing)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"[rank {args.rank}] %(asctime)s %(message)s")

    repos = [l.strip() for l in Path(args.repo_file).read_text().splitlines() if l.strip()]
    items = []
    for repo in repos:
        n = int(pd.read_json(Path(repo, "meta", "info.json"), typ="series")["total_episodes"])
        items += [(repo, ep) for ep in range(n) if not cache_path(args.cache_dir, repo, ep).is_file()]
    items = items[args.rank :: args.world]
    if args.limit:
        items = items[: args.limit]
    logging.info("pending episodes: %d", len(items))
    if not items:
        return

    # Worker i of W receives positions i, i+W, ...; order items so that each worker gets one contiguous block
    # (episodes of the same repo stay on the same worker, so each worker opens few datasets).
    workers = max(1, args.num_workers)
    blocks = [items[i * len(items) // workers : (i + 1) * len(items) // workers] for i in range(workers)]
    items = [blocks[i % workers][i // workers] for i in range(workers * max(map(len, blocks))) if i // workers < len(blocks[i % workers])]

    device = torch.device("cuda")
    vlm = load_vit(args.init_ckpt, args.qwen_path, device)
    loader = torch.utils.data.DataLoader(
        EpisodeFrames(items, args.num_keyframes, args.alpha, args.image_size, args.qwen_path),
        batch_size=None, num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None,
        # The ViT is already on CUDA in this process; forked workers running the image processor segfault.
        multiprocessing_context="spawn" if args.num_workers else None,
    )
    for n, item in enumerate(loader, 1):
        k, c = item["num_frames"], len(CAMERAS)
        pixels = item["pixels"].to(device, torch.bfloat16)
        grids = item["grids"].to(device)
        with torch.no_grad():
            feats = visual_tokens(vlm, pixels, grids)
        plan = item["plan"]
        record = {
            "frames": torch.as_tensor(plan.frames, dtype=torch.long),
            "forced": torch.as_tensor(plan.forced, dtype=torch.bool),
            "flips": torch.as_tensor(plan.flips, dtype=torch.long),
            "active": tuple(int(x) for x in plan.active),
            "length": int(plan.length),
            "time_ratio": torch.as_tensor(plan.frames / max(1, plan.length - 1), dtype=torch.float32),
            "cameras": CAMERAS,
            "vit_source": args.init_ckpt or args.qwen_path,
            "feats": feats.reshape(k, c, feats.shape[-2], feats.shape[-1]).to(torch.bfloat16).cpu(),
        }
        path = cache_path(args.cache_dir, item["repo"], item["episode"])
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save(record, tmp)
        os.replace(tmp, path)
        if n % 100 == 0:
            logging.info("%d / %d", n, len(items))
    logging.info("done")


if __name__ == "__main__":
    main()
