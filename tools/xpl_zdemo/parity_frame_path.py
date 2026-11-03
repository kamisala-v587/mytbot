"""PARITY A (F1 T5) — frame path vs an upstream cached ViT record (strongest check).

For episodes that have records in an upstream ViT cache (e.g. /vla/workspace/cache/
bpv4cdxb_su_p1, built by BPV4CDX/build_cache.py with the same pinned Qwen3.5 ViT)
whose repo is locally openable as a LeRobot dataset:

  - re-extract the RECORDED frames (the record supplies `frames`/`cameras`; the
    sampler is NOT under test here) through the producer's exact frame path —
    my torchcodec reader -> float32/255 -> resize_with_pad 224 -> Qwen3.5
    processor(do_rescale=False) -> ViT visual_tokens -> (pool_spatial to 5 if
    the record is P64)
  - compare against the stored `feats`: per-episode cosine similarity + max abs
    diff; PASS = cosine >= 0.99 (upstream's own cached-vs-fresh gate,
    BPV4CDX/gate_g1_g2.py G2).

Also cross-checks the direct torchcodec reader against LeRobotDataset(
video_backend="torchcodec") pixel-for-pixel on the same mp4s, so the producer's
reader is validated, not just the resize/processor/ViT/pooling chain.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")

from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens
from lerobot.policies.BPV4CDX.dataset import pool_spatial
from lerobot.transforms.utils import resize_with_pad

SAMPLER_VERSION = "su-uniform_active_v2_embodiment"


def pick_records(cache, repo_root, count):
    """(repo_path, episode, record_path) for the first `count` records whose repo exists."""
    cache, repo_root = __import__("pathlib").Path(cache), __import__("pathlib").Path(repo_root)
    out = []
    for rec_path in sorted(cache.glob("*/*/ep*.pt")):
        ep_dir = rec_path.parent
        repo = repo_root / ep_dir.relative_to(cache)
        if repo.is_dir():
            out.append((repo, int(rec_path.stem[2:]), rec_path))
        if len(out) >= count:
            break
    return out


def file_start_index(ds, cam, file_index, chunk_index):
    """Global index of the first frame inside one shared mp4 (episodes concatenate)."""
    best = None
    for m in ds.meta.episodes:
        if m.get(f"videos/{cam}/file_index") == file_index and \
                m.get(f"videos/{cam}/chunk_index", m.get("data/chunk_index")) == chunk_index:
            v = int(m["dataset_from_index"])
            best = v if best is None else min(best, v)
    if best is None:
        raise ValueError(f"no episode row for videos/{cam} file {file_index} chunk {chunk_index}")
    return best


def episode_pixels(repo, episode, frames, cams, size=224):
    """(my_images, ds_images, max_reader_diff): the 96 slot-major frames both ways."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from torchcodec.decoders import VideoDecoder

    ds = LeRobotDataset(str(repo), video_backend="torchcodec")
    start = int(ds.meta.episodes[episode]["dataset_from_index"])
    mine, theirs, diff = [], [], 0.0
    decoders = {}
    for f in frames:
        item = ds[start + int(f)]
        for cam in cams:
            t_ds = item[cam]
            meta = ds.meta.episodes[episode]
            fi, ci = meta[f"videos/{cam}/file_index"], meta.get(f"videos/{cam}/chunk_index",
                                                                meta.get("data/chunk_index"))
            if cam not in decoders:
                decoders[cam] = (VideoDecoder(str(repo / ds.meta.get_video_file_path(episode, cam))),
                                 file_start_index(ds, cam, fi, ci))
            dec, base = decoders[cam]
            t_mine = dec[int(start - base + int(f))].data.to(torch.float32).div_(255.0)
            diff = max(diff, float((t_mine - t_ds).abs().max()))
            mine.append(resize_with_pad(t_mine[None], size, size)[0])
            theirs.append(resize_with_pad(t_ds[None], size, size)[0])
    return mine, theirs, diff


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/vla/workspace/cache/bpv4cdxb_su_p1")
    ap.add_argument("--repo-root", default="/vla/workspace/my_robotwin/data_bp/p1_n10_lerobot")
    ap.add_argument("--qwen_path", default="/vla/workspace/models/q35/base_800m")
    ap.add_argument("--count", type=int, default=3)
    args = ap.parse_args()

    device = torch.device("cuda")
    vlm = load_vit(None, args.qwen_path, device)
    from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

    processor, _ = load_qwen35_vision(args.qwen_path)

    records = pick_records(args.cache, args.repo_root, args.count)
    if len(records) < args.count:
        raise SystemExit(f"only {len(records)} usable records under {args.cache} (repo root {args.repo_root})")
    all_pass = True
    for repo, episode, rec_path in records:
        rec = torch.load(rec_path, map_location="cpu", weights_only=False)
        version = (rec.get("sampler_version") or {}).get("version")
        if version != SAMPLER_VERSION:
            raise SystemExit(f"{rec_path}: sampler version {version!r} != pinned — wrong cache lineage")
        frames = [int(f) for f in rec["frames"]]
        cams = list(rec["cameras"])
        mine, theirs, rdiff = episode_pixels(repo, episode, frames, cams)
        out = processor(mine, do_rescale=False)
        with torch.no_grad():
            feats = visual_tokens(vlm, torch.as_tensor(out.pixel_values).to(device, torch.bfloat16),
                                  torch.as_tensor(out.image_grid_thw, dtype=torch.long).to(device))
        k = len(frames)
        mine64 = feats.reshape(k, len(cams), feats.shape[-2], feats.shape[-1]).float().cpu()  # [K,C,64,D]
        stored = rec["feats"].float()  # [K,C,P,D] with P = rec["spatial"]

        def cos(a, b):
            return float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0))

        lines = [f"{repo.name}/ep{episode:05d} ({rec_path.parent.relative_to(args.cache)}): "
                 f"reader-vs-LeRobotDataset max abs diff {rdiff:.2e}"]
        if int(rec["spatial"]) == 64:
            pooled_mine, pooled_ref = pool_spatial(mine64, 5), pool_spatial(stored, 5)
            lines.append(f"  P64: cosine {cos(mine64, stored):.6f}, max abs diff "
                         f"{float((mine64 - stored).abs().max()):.4f}")
            lines.append(f"  P5 : cosine {cos(pooled_mine, pooled_ref):.6f}, max abs diff "
                         f"{float((pooled_mine - pooled_ref).abs().max()):.4f}")
            ok = cos(pooled_mine, pooled_ref) >= 0.99 and cos(mine64, stored) >= 0.99
        else:
            lines.append(f"  P5 : cosine {cos(mine64, stored):.6f}, max abs diff "
                         f"{float((mine64 - stored).abs().max()):.4f}")
            ok = cos(mine64, stored) >= 0.99
        ok = ok and rdiff <= 1.0 / 255.0
        all_pass &= ok
        print("\n".join(lines) + f"  -> {'PASS' if ok else 'FAIL'}")
    print("PARITY A " + ("PASS" if all_pass else "FAIL (STOP)") + f" — {len(records)} episodes, "
          f"gate cosine >= 0.99")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
