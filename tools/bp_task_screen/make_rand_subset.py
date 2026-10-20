"""Create <task>/aloha-agilex_randomized_<N> from the first N episodes of aloha-agilex_randomized_500.

Video files are hard-linked (no re-encoding) and episodes keep their original from/to timestamps, so decoded frames
are identical to the source dataset; data parquet and metadata are rewritten for the kept episodes.

Usage: python make_rand_subset.py N task [task ...]   (task "all" = every task with a randomized_500 dataset)
"""
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np

import lerobot.datasets.dataset_tools as dt
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROOT = Path("/vla/workspace/data/robotwin2.0")


def _link_videos(src_dataset, dst_meta, episode_mapping, **_):
    """Replacement for dt._copy_and_reindex_videos: hard-link whole video files, keep original timestamps."""
    meta = {new: {} for new in episode_mapping.values()}
    for key in src_dataset.meta.video_keys:
        files = set()
        for old, new in episode_mapping.items():
            ep = src_dataset.meta.episodes[old]
            chunk, file = ep[f"videos/{key}/chunk_index"], ep[f"videos/{key}/file_index"]
            files.add((chunk, file))
            meta[new].update({
                f"videos/{key}/chunk_index": chunk,
                f"videos/{key}/file_index": file,
                f"videos/{key}/from_timestamp": ep[f"videos/{key}/from_timestamp"],
                f"videos/{key}/to_timestamp": ep[f"videos/{key}/to_timestamp"],
            })
        for chunk, file in files:
            rel = src_dataset.meta.video_path.format(video_key=key, chunk_index=chunk, file_index=file)
            dst = dst_meta.root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src_dataset.root / rel, dst)
            except OSError:
                shutil.copy(src_dataset.root / rel, dst)
    return meta


dt._copy_and_reindex_videos = _link_videos


def make(task: str, n: int) -> None:
    src_root = ROOT / task / "aloha-agilex_randomized_500"
    name = f"aloha-agilex_randomized_{n}"
    dst_root = ROOT / task / name
    if (dst_root / "meta" / "info.json").is_file():
        print(f"[skip] {dst_root}", flush=True)
        return
    if dst_root.exists():
        shutil.rmtree(dst_root)
    src = LeRobotDataset(str(src_root), video_backend="torchcodec")
    keep = list(range(min(n, src.meta.total_episodes)))
    dt.split_dataset(src, {name: keep}, output_dir=ROOT / task)
    check(src, LeRobotDataset(str(dst_root), video_backend="torchcodec"), keep)
    print(f"[done] {dst_root}  episodes={len(keep)}", flush=True)


def check(src, dst, keep) -> None:
    if dst.meta.total_episodes != len(keep):
        raise RuntimeError(f"episode count {dst.meta.total_episodes} != {len(keep)}")
    expected = sum(int(src.meta.episodes[i]["length"]) for i in keep)
    if dst.meta.total_frames != expected:
        raise RuntimeError(f"frame count {dst.meta.total_frames} != {expected}")
    for ep in (0, len(keep) - 1):
        s0, d0 = int(src.meta.episodes[ep]["dataset_from_index"]), int(dst.meta.episodes[ep]["dataset_from_index"])
        length = int(src.meta.episodes[ep]["length"])
        for off in (0, length - 1):
            a, b = src[s0 + off], dst[d0 + off]
            for key in src.meta.video_keys:
                if not np.array_equal(a[key].numpy(), b[key].numpy()):
                    raise RuntimeError(f"frame mismatch ep={ep} off={off} key={key}")
            for key in ("observation.state", "action"):
                if not np.allclose(a[key].numpy(), b[key].numpy()):
                    raise RuntimeError(f"{key} mismatch ep={ep} off={off}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    n = int(sys.argv[1])
    tasks = sys.argv[2:]
    if tasks == ["all"]:
        tasks = sorted(p.name for p in ROOT.iterdir() if (p / "aloha-agilex_randomized_500" / "meta" / "info.json").is_file())
    failed = []
    for t in tasks:
        try:
            make(t, n)
        except Exception as exc:  # keep going; report at the end
            failed.append(t)
            print(f"[fail] {t}: {exc!r}", flush=True)
    print("failed:", failed)
