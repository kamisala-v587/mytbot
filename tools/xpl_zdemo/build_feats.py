"""F1 producer — real Z_demo FEATURE cache for XR1 over the converted RoboDojo root.

Per episode (upstream-pinned path, everything imported, nothing re-implemented):

    episode json -> state [N,14] -> sample_su_emb (32 slots, version-checked)
    -> mp4 frames at plan.frames via torchcodec (slot-major, cameras
       ego/wrist_left/wrist_right inner — the order record() reshapes)
    -> float32/255 [K,3,H,W] -> resize_with_pad(., 224, 224)
    -> Qwen3.5 processor(images, do_rescale=False) -> Qwen3.5-0.8B ViT (visual_tokens)
    -> BPV4CDX.build_cache.record(plan, feats, qwen_path, [True]*3, CAMERAS, key,
       store_spatial=5)   # P5 pooling happens inside record()
    -> REDUCED consumer record {feats, time, camera_valid, sampler_version, spatial}
       saved to <out>/<key>.pt (tmp + os.replace, upstream build_cache.save).

Record layout/dtypes are exactly what XPolicyLab.utils.bp.zdemo.ZdemoCache reads:
feats bf16 [32,3,5,1024]; time float32 [32] (renamed from upstream time_ratio);
camera_valid bool [32,3]; sampler_version the STRING (upstream stores a dict);
spatial 5.

Failures go to <out>/_failures/rank<r>.jsonl (upstream build_cache.py:267-300
ergonomics); built + failed must reconcile with the scanned shard. Resumable:
existing <out>/<key>.pt files are skipped. --finalize writes index.json +
manifest.json from the frozen delivery describe() (real sha256, synthetic: false).

usage (q35va env): python build_feats.py --root <data root> --out <cache root> \
    [--rank r --world w --episodes_per_batch 4 --limit N --dry_run --finalize]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from episode_source import CAMERAS, iter_episodes, load_episode
from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens
from lerobot.transforms.utils import resize_with_pad
from plan_slots import NUM_SLOTS, SAMPLER_VERSION, plan_for

QWEN_PATH = "/vla/workspace/models/q35/base_800m"
MYTBOT_ROOT = Path(__file__).resolve().parents[2]


def read_frames(video_path: str, frames: np.ndarray, num_frames: int, cam: str, key: str) -> torch.Tensor:
    """Slot-ordered RGB frames from one mp4: [K,3,H,W] float32 in [0,1].

    torchcodec VideoDecoder (the same backend as upstream LeRobotDataset(
    video_backend="torchcodec")); indices deduplicated for the read, then mapped
    back so repeated slots (very short episodes) resolve to the same frame.
    """
    from torchcodec.decoders import VideoDecoder

    dec = VideoDecoder(video_path)
    nf = int(dec.metadata.num_frames)
    if nf != int(num_frames):
        raise ValueError(f"{key}: {cam} video has {nf} frames, episode num_frames={num_frames}")
    uniq, inverse = np.unique(np.asarray(frames, dtype=np.int64), return_inverse=True)
    batch = dec.get_frames_at([int(f) for f in uniq]).data  # [U,3,H,W] uint8 RGB
    if tuple(batch.shape)[0] != len(uniq):
        raise ValueError(f"{key}: {cam} decoder returned {tuple(batch.shape)[0]} frames for {len(uniq)} indices")
    sel = batch[torch.as_tensor(inverse)]  # [K,3,H,W] slot order
    return sel.to(torch.float32).div_(255.0)


class EpisodeFrames(torch.utils.data.Dataset):
    """One item per episode: processor pixel outputs for its 32 slots x 3 cameras."""

    def __init__(self, episodes, size: int, qwen_path: str) -> None:
        self.episodes, self.size, self.qwen_path = episodes, size, qwen_path
        self._processor = None

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, i: int):
        key, json_path = self.episodes[i]
        try:
            ep = load_episode(json_path)
            plan = plan_for(ep)
            per_cam = [read_frames(ep["videos"][cam], np.asarray(plan.frames), ep["num_frames"], cam, key)
                       for cam in CAMERAS]
            per_cam = [resize_with_pad(frames, self.size, self.size) for frames in per_cam]  # [K,3,224,224]
            k = len(plan.frames)
            images = [per_cam[c][s] for s in range(k) for c in range(len(CAMERAS))]  # slot-major, cams inner
            if self._processor is None:
                from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

                torch.set_num_threads(2)
                self._processor, _ = load_qwen35_vision(self.qwen_path)
            out = self._processor(images, do_rescale=False)
            return {"key": key, "plan": plan, "pixels": torch.as_tensor(out.pixel_values),
                    "grids": torch.as_tensor(out.image_grid_thw, dtype=torch.long)}
        except Exception as e:  # noqa: BLE001 - one broken episode must not stop the build
            return {"key": key, "error": f"{type(e).__name__}: {e}"}


def reduce_record(rec: dict, plan) -> dict:
    """Upstream build_cache.record dict -> the consumer's five fields (exact set)."""
    feats = rec["feats"]
    if feats.dtype != torch.bfloat16 or tuple(feats.shape) != (NUM_SLOTS, 3, 5, 1024):
        raise ValueError(f"record() produced feats {tuple(feats.shape)}/{feats.dtype}, "
                         f"expected [32, 3, 5, 1024] bf16")
    time_ratio = torch.as_tensor(rec["time_ratio"], dtype=torch.float32)
    if tuple(time_ratio.shape) != (NUM_SLOTS,):
        raise ValueError(f"record() produced time_ratio {tuple(time_ratio.shape)}")
    if float(time_ratio.min()) < 0.0 or float(time_ratio.max()) > 1.0:
        raise ValueError(f"time_ratio outside [0,1]: [{time_ratio.min()}, {time_ratio.max()}]")
    valid = rec["camera_valid"].to(torch.bool)
    if tuple(valid.shape) != (NUM_SLOTS, 3):
        raise ValueError(f"record() produced camera_valid {tuple(valid.shape)}")
    return {"feats": feats, "time": time_ratio, "camera_valid": valid,
            "sampler_version": plan.version["version"], "spatial": 5}


def save(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    torch.save(rec, tmp)
    os.replace(tmp, path)  # upstream build_cache.save


def drop_existing(episodes, out: Path) -> list:
    """Skip episodes whose record already exists (resumable; one listing per chunk dir)."""
    listed: dict[Path, set] = {}
    pending = []
    for key, path in episodes:
        d = out / key.split("/", 1)[0]
        if d not in listed:
            listed[d] = set(os.listdir(d)) if d.is_dir() else set()
        if f"{Path(key).stem}.pt" not in listed[d]:
            pending.append((key, path))
    return pending


def git_head() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                          cwd=str(MYTBOT_ROOT)).stdout.strip()
    if not head.startswith("94859f7"):
        raise RuntimeError(f"my_tbot HEAD moved: {head!r} (F1 was pinned to 94859f7) — STOP")
    return head[:7]


def build(args) -> None:
    episodes = iter_episodes(args.root)
    shard = drop_existing(episodes[args.rank::args.world], Path(args.out))
    if args.limit:
        shard = shard[:args.limit]
    logging.info("root %s: %d episodes scanned, shard %d/%d -> %d pending (skip-existing applied)",
                 args.root, len(episodes), args.rank, args.world, len(shard))
    if not shard or args.dry_run:
        return
    device = torch.device("cuda")
    vlm = load_vit(None, args.qwen_path, device)
    from lerobot.policies.BPV4CDX.build_cache import record

    workers = max(1, args.num_workers)
    loader = torch.utils.data.DataLoader(
        EpisodeFrames(shard, args.image_size, args.qwen_path), batch_size=None,
        num_workers=workers, prefetch_factor=4 if workers else None,
        multiprocessing_context="spawn" if workers else None,
    )
    out = Path(args.out)
    fail_dir = out / "_failures"
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
            rec = record(b["plan"], f, args.qwen_path, [True] * len(CAMERAS), list(CAMERAS),
                         b["key"], store_spatial=5)
            save(out / f"{b['key']}.pt", reduce_record(rec, b["plan"]))

    for item in loader:
        if "error" in item:
            failed += 1
            with fail_file.open("a") as fh:
                fh.write(json.dumps({"key": item["key"], "error": item["error"]}) + "\n")
            continue
        batch.append(item)
        if len(batch) >= args.episodes_per_batch:
            flush(batch)
            done += len(batch)
            batch = []
            if done % 100 < args.episodes_per_batch:
                rate = done / max(1e-6, time.time() - t0)
                logging.info("%d / %d done, %d failed, %.2f ep/s, eta %.1f h", done, len(shard), failed,
                             rate, (len(shard) - done - failed) / max(1e-6, rate) / 3600)
    if batch:
        flush(batch)
        done += len(batch)
    if done + failed != len(shard):
        raise RuntimeError(f"rank {args.rank} reconcile FAILED: built {done} + failed {failed} != {len(shard)}")
    logging.info("rank %d done: %d built, %d failed (%s)", args.rank, done, failed, fail_file)


def finalize(args) -> None:
    """T4: index.json + manifest.json over the finished cache root."""
    from lerobot.policies.BPV4CDXB.frozen import describe, load_frozen

    out = Path(args.out)
    written = sorted(f"{p.parent.name}/{p.stem}" for p in out.glob("*/episode_*.pt"))
    scanned = iter_episodes(args.root)
    fail_files = sorted((out / "_failures").glob("rank*.jsonl")) if (out / "_failures").is_dir() else []
    failed_keys = set()
    for f in fail_files:
        for line in f.open():
            line = line.strip()
            if line:
                failed_keys.add(json.loads(line)["key"])
    print(f"records on disk: {len(written)}; failed keys: {len(failed_keys)} "
          f"({[f.name for f in fail_files]}); scanned episodes: {len(scanned)}")
    if len(written) + len(failed_keys) != len(scanned) or len(failed_keys & set(written)):
        raise RuntimeError(f"reconcile FAILED: {len(written)} built + {len(failed_keys)} failed "
                           f"(overlap {len(failed_keys & set(written))}) != {len(scanned)} scanned")
    missing = sorted(k for k, _ in scanned if k not in set(written) and k not in failed_keys)
    if missing:
        raise RuntimeError(f"{len(missing)} scanned episodes have neither a record nor a failure line "
                           f"(first: {missing[:5]}) — rerun their shards")
    if failed_keys:
        print(f"WARNING: {len(failed_keys)} failed episodes are NOT in the index; the XR1 dataset "
              f"coverage init will fail until they are resolved (first: {sorted(failed_keys)[:5]})")
    index = {k: f"{k}.pt" for k in written}


    frozen_dir = args.frozen or os.environ.get("BPV4CDXB_FROZEN")
    if not frozen_dir:
        raise SystemExit("--frozen or $BPV4CDXB_FROZEN (the StageA_AE_frozen/0750000 dir) is required")
    desc = describe(load_frozen(frozen_dir))
    if desc.get("spatial_tokens_used") != 5:
        raise RuntimeError(f"frozen delivery spatial_tokens_used={desc.get('spatial_tokens_used')!r} != 5")
    sampler = desc.get("sampler") or {}
    if sampler.get("version") != SAMPLER_VERSION or sampler.get("num_slots") != NUM_SLOTS:
        raise RuntimeError(f"frozen delivery sampler block drifted: {sampler}")
    manifest = {
        "frozen": desc,
        "synthetic": False,
        "producer": {
            "tool": "tools/xpl_zdemo/build_feats.py",
            "root": str(Path(args.root).resolve()),
            "qwen_path": args.qwen_path,
            "episodes": len(written),
            "failures": len(failed_keys),
            "date": time.strftime("%Y-%m-%d"),
            "git_head_my_tbot": git_head(),
        },
    }
    (out / "index.json").write_text(json.dumps(index))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"finalized {out}: {len(written)} records, ckpt_sha256 {desc['ckpt_sha256'][:12]}, synthetic false")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="converted dataset root (chunk-*/episode_*.json)")
    ap.add_argument("--out", required=True, help="feature cache root to write")
    ap.add_argument("--qwen_path", default=QWEN_PATH)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--episodes_per_batch", type=int, default=4, help="episodes per ViT forward")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--limit", type=int, default=0, help="only the first N pending episodes of this shard")
    ap.add_argument("--dry_run", action="store_true", help="scan + shard sizes only; nothing runs")
    ap.add_argument("--finalize", action="store_true", help="T4: write index.json + manifest.json and exit")
    ap.add_argument("--frozen", default=None, help="StageA_AE_frozen dir for --finalize (default $BPV4CDXB_FROZEN)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"[rank {args.rank}] %(asctime)s %(message)s")
    if args.finalize:
        finalize(args)
    else:
        build(args)


if __name__ == "__main__":
    main()
