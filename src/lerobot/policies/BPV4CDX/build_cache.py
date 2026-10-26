"""Build the 32-slot ViT feature cache for one sampler; multi-GPU / multi-process shards, resumable.

Two modes:
  repo list (AE-1 / AE-2): --repo_file [--data_root]; files <cache>/<task>/<variant>/epXXXXX.pt, feats
      [32, 3, 64, 1024] bf16, aloha cameras, the original aloha samplers (bit-identical to the earlier caches)
  split manifest (AE-3): --manifest --data_roots; files <cache>/<relative repo id>/epXXXXX.pt; any embodiment
      (cameras via IMAGE_MAPPING into slots image0..image2, a missing camera is zero with camera_valid False);
      S-U / S-UF on the MASK_MAPPING joint / gripper dims; --store_spatial 5 stores the P5-pooled tokens only
      ([32, 3, 5, 1024], 13x smaller); --order heldout_first builds the held-out episodes and RoboTwin first
Record fields: feats, frames, time_ratio, camera_valid, cameras, forced, flips, active, length, repeats, sampler,
sampler_version, vit_source, spatial (64 or 5), rel.

Concurrency: shard r of w (--rank / --world) gets every w-th episode of a repo-contiguous order; run several
processes per GPU (tools/bpv4cdx/build_cache_ae3.sh) so ViT batches of different processes overlap with decoding;
each process decodes with --num_workers spawn workers and runs the ViT on --episodes_per_batch episodes at once.
Existing files are skipped; a failing episode is logged to <cache>/_failures/rank<r>.jsonl and skipped.

Usage (AE-3, one process; the launcher starts gpus x procs of these):
  python -m lerobot.policies.BPV4CDX.build_cache --manifest <ae3_split.json> --data_roots /share --cache_dir <dir> \
      --sampler suf --store_spatial 5 --array_store <store> --qwen_path <base_800m> --rank 0 --world 16 --num_workers 8
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens
from lerobot.transforms.utils import resize_with_pad

from .dataset import demo_cache_path, pool_spatial
from .embodiment import embodiment_from_meta
from .paths import load_manifest, manifest_repos, read_repo_list, rel_id, split_episodes
from .repo_data import EpView, read_arrays, store_dir
from .sampling import sample


def repo_states(repo: str, emb, array_store: str | None) -> EpView:
    sd = store_dir(array_store, repo) if array_store else None
    if sd is not None and (sd / "index.json").is_file():
        index = json.loads((sd / "index.json").read_text())
        return EpView(np.load(sd / "state.npy", mmap_mode="r"), {int(e): (int(s), int(t)) for e, s, t in index["episodes"]})
    state, _, spans = read_arrays(repo, emb)
    return EpView(state, spans)


class SlotFrames(torch.utils.data.Dataset):
    """One item per episode: processor outputs for its 32 slots x the valid cameras."""

    def __init__(self, items, sampler: str, num_slots: int, alpha: float, size: int, qwen_path: str,
                 use_embodiment: bool, array_store: str | None) -> None:
        self.items, self.sampler, self.k, self.alpha, self.size = items, sampler, num_slots, alpha, size
        self.qwen_path, self.use_emb, self.array_store = qwen_path, use_embodiment, array_store
        self._open: dict[str, tuple] = {}
        self._processor = None

    def __len__(self) -> int:
        return len(self.items)

    def _repo(self, repo: str):
        if repo not in self._open:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            ds = LeRobotDataset(repo, video_backend="torchcodec")
            emb = embodiment_from_meta(ds.meta.robot_type, ds.meta.features)
            self._open = {repo: (ds, emb, repo_states(repo, emb, self.array_store))}  # one open repo per worker
        return self._open[repo]

    def __getitem__(self, i: int):
        repo, ep = self.items[i]
        try:
            ds, emb, states = self._repo(repo)
            plan = sample(self.sampler, states[ep], self.k, self.alpha, emb=emb if self.use_emb else None)
            start = int(ds.meta.episodes[ep]["dataset_from_index"])
            valid = [c is not None for c in emb.cameras]
            images = []
            for f in plan.frames:
                item = ds[start + int(f)]
                images.extend(resize_with_pad(item[c][None], self.size, self.size)[0] for c in emb.cameras if c is not None)
            if self._processor is None:
                from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

                torch.set_num_threads(2)
                self._processor, _ = load_qwen35_vision(self.qwen_path)
            out = self._processor(images, do_rescale=False)
            return {"repo": repo, "episode": ep, "plan": plan, "pixels": torch.as_tensor(out.pixel_values),
                    "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long), "valid": valid,
                    "cameras": list(emb.cameras)}
        except Exception as e:  # noqa: BLE001 - one broken episode must not stop the build
            return {"repo": repo, "episode": ep, "error": f"{type(e).__name__}: {e}"}


def record(plan, feats: torch.Tensor, qwen_path: str, valid: list[bool], cameras: list, repo: str,
           store_spatial: int) -> dict:
    """feats: [K * n_valid_cameras, 64, D] in slot-major order -> [K, 3, P, D] with zeros for missing cameras."""
    k, c = len(plan.frames), len(valid)
    nv = sum(valid)
    f = feats.reshape(k, nv, feats.shape[-2], feats.shape[-1])
    full = torch.zeros(k, c, feats.shape[-2], feats.shape[-1], dtype=feats.dtype, device=feats.device)
    full[:, [j for j in range(c) if valid[j]]] = f
    if store_spatial == 5:
        full = pool_spatial(full.float(), 5)
    return {
        "frames": torch.as_tensor(plan.frames, dtype=torch.long),
        "forced": torch.as_tensor(plan.forced, dtype=torch.bool),
        "flips": torch.as_tensor(plan.flips, dtype=torch.long),
        "active": tuple(int(x) for x in plan.active),
        "length": int(plan.length),
        "time_ratio": torch.as_tensor(plan.frames / max(1, plan.length - 1), dtype=torch.float32),
        "camera_valid": torch.tensor(valid, dtype=torch.bool)[None].expand(k, c).clone(),
        "cameras": cameras,
        "vit_source": qwen_path,
        "sampler": plan.sampler,
        "sampler_version": plan.version,
        "repeats": int(plan.repeats),
        "spatial": int(full.shape[2]),
        "rel": rel_id(repo),
        "feats": full.to(torch.bfloat16).cpu(),
    }


def save(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(rec, tmp)
    os.replace(tmp, path)


def build_sv(items, args, vlm, device) -> None:
    """S-V (aloha repo lists only): every frame through the ViT first, then the 32 slots."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.BPV4BPEn.build_feature_cache import CAMERAS, load_states
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
        save(demo_cache_path(args.cache_dir, repo, ep), record(plan, feats, args.qwen_path, [True] * 3, CAMERAS, repo, 64))
        if n % 20 == 0:
            logging.info("%d / %d", n, len(items))


def pending_items(args) -> list[tuple[str, int]]:
    """All (repo, episode) of the selected split, in build order (existing cache files: `drop_existing`)."""
    layout = "rel" if args.manifest else "task_variant"
    items = []
    if args.manifest:
        manifest = load_manifest(args.manifest)
        pairs = manifest_repos(manifest, args.data_roots.split(":"), args.sources.split(",") if args.sources else None)
        for repo, entry in pairs:
            eps = split_episodes(repo, int(entry["total_episodes"]), args.split, {}, entry["heldout"])
            held = set(entry["heldout"])
            if args.limit_per_repo:
                # smoke tests: every held-out episode (HOM) and the first N training episodes per repo
                eps = sorted(set(e for e in eps if e in held) | set([e for e in eps if e not in held][: args.limit_per_repo]))
            for ep in eps:
                # build order key: held-out first, then RoboTwin, then the rest (stable inside)
                prio = 0 if ep in held else (1 if entry["rel"].startswith("RoboTwin") else 2)
                items.append((repo, ep, prio if args.order == "heldout_first" else 0))
        items.sort(key=lambda x: x[2])
        items = [(r, e) for r, e, _ in items]
    else:
        rules = {"clean_50": 5, "randomized_100": 10}
        for repo in read_repo_list(args.repo_file, args.data_root):
            n = int(pd.read_json(Path(repo, "meta", "info.json"), typ="series")["total_episodes"])
            items += [(repo, ep) for ep in split_episodes(repo, n, args.split, rules)]
    return items


def drop_existing(items: list[tuple[str, int]], cache_dir: str, layout: str) -> list[tuple[str, int]]:
    # one directory listing per repo instead of one stat per episode (network file system)
    listed: dict[Path, set[str]] = {}
    out = []
    for r, e in items:
        p = demo_cache_path(cache_dir, r, e, layout)
        if p.parent not in listed:
            listed[p.parent] = set(os.listdir(p.parent)) if p.parent.is_dir() else set()
        if p.name not in listed[p.parent]:
            out.append((r, e))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", default=None, help="repo list (AE-1 / AE-2)")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--manifest", default=None, help="AE-3 split manifest (replaces --repo_file)")
    ap.add_argument("--data_roots", default="", help="':'-separated local roots for the manifest relative ids")
    ap.add_argument("--sources", default="", help="','-separated top-level sources of the manifest (default all)")
    ap.add_argument("--array_store", default=None, help="per-repo state arrays (ae3_data prepare-arrays) for sampling")
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--sampler", choices=["sa", "su", "sv", "suf"], required=True)
    ap.add_argument("--qwen_path", required=True)
    ap.add_argument("--num_slots", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--store_spatial", type=int, choices=[5, 64], default=64)
    ap.add_argument("--episodes_per_batch", type=int, default=1, help="episodes per ViT forward")
    ap.add_argument("--order", choices=["heldout_first", "manifest"], default="heldout_first")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending episodes of this shard (testing)")
    ap.add_argument("--limit_per_repo", type=int, default=0, help="at most N episodes per repo (smoke tests)")
    ap.add_argument("--split", choices=["all", "train", "val"], default="all")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"[rank {args.rank}] %(asctime)s %(message)s")
    if not args.manifest and not args.repo_file:
        raise SystemExit("--manifest or --repo_file is required")
    layout = "rel" if args.manifest else "task_variant"
    use_emb = bool(args.manifest) or args.sampler == "suf"

    items = pending_items(args)
    # repo-contiguous blocks per shard (each shard opens few repos); held-out-first order is kept inside a shard
    items = drop_existing(items[args.rank :: args.world], args.cache_dir, layout)
    if args.limit:
        items = items[: args.limit]
    logging.info("sampler %s, layout %s, store_spatial %d, pending episodes: %d", args.sampler, layout,
                 args.store_spatial, len(items))
    if not items:
        return
    device = torch.device("cuda")
    vlm = load_vit(None, args.qwen_path, device)
    if args.sampler == "sv":
        build_sv(items, args, vlm, device)
        return

    workers = max(1, args.num_workers)
    blocks = [items[i * len(items) // workers : (i + 1) * len(items) // workers] for i in range(workers)]
    items = [blocks[i % workers][i // workers] for i in range(workers * max(map(len, blocks))) if i // workers < len(blocks[i % workers])]
    loader = torch.utils.data.DataLoader(
        SlotFrames(items, args.sampler, args.num_slots, args.alpha, args.image_size, args.qwen_path, use_emb,
                   args.array_store),
        batch_size=None, num_workers=args.num_workers, prefetch_factor=4 if args.num_workers else None,
        # the ViT is already on CUDA in this process; forked workers running the image processor segfault
        multiprocessing_context="spawn" if args.num_workers else None,
    )
    fail_dir = Path(args.cache_dir) / "_failures"
    fail_dir.mkdir(parents=True, exist_ok=True)
    fail_file = fail_dir / f"rank{args.rank}.jsonl"
    t0, done, failed, batch = time.time(), 0, 0, []

    def flush(batch):
        pixels = torch.cat([b["pixels"] for b in batch]).to(device, torch.bfloat16)
        grids = torch.cat([b["grids"] for b in batch]).to(device)
        with torch.no_grad():
            feats = visual_tokens(vlm, pixels, grids)
        sizes = [len(b["grids"]) for b in batch]
        for b, f in zip(batch, torch.split(feats, sizes)):
            save(demo_cache_path(args.cache_dir, b["repo"], b["episode"], layout),
                 record(b["plan"], f, args.qwen_path, b["valid"], b["cameras"], b["repo"], args.store_spatial))

    for item in loader:
        if "error" in item:
            failed += 1
            with fail_file.open("a") as fh:
                fh.write(json.dumps({"rel": rel_id(item["repo"]), "episode": int(item["episode"]), "error": item["error"]}) + "\n")
            continue
        batch.append(item)
        if len(batch) >= args.episodes_per_batch:
            flush(batch)
            done += len(batch)
            batch = []
            if done % 100 < args.episodes_per_batch:
                rate = done / max(1e-6, time.time() - t0)
                logging.info("%d / %d done, %d failed, %.2f ep/s, eta %.1f h", done, len(items), failed, rate,
                             (len(items) - done - failed) / max(1e-6, rate) / 3600)
    if batch:
        flush(batch)
        done += len(batch)
    logging.info("done: %d built, %d failed (%s)", done, failed, fail_file)


if __name__ == "__main__":
    main()
