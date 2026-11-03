"""Episode scanning/loading for the XR1 F1 Z_demo feature cache (RoboDojo converted root).

The converted root (``<OUTPUT_DIR>/data`` of the XR1 conversion) holds one json per
episode: ``<root>/<chunk>/episode_XXXXXXX.json`` with ``num_frames``, raw proprios and
absolute mp4 paths per camera. This module turns that into what the pinned upstream
sampler consumes:

    state  np.float32 [N, 14] — np.concatenate(left_arm_joint, left_gripper_pos,
          right_arm_joint, right_gripper_pos) in THAT order (must match the
          embodiment mask: 6 joints + 1 gripper per arm, grippers at dims 6/13)
    videos {ego, wrist_left, wrist_right: absolute mp4 path} (slot order
          image0/1/2 = head/left-wrist/right-wrist, upstream embodiment.py:19-20)

Every anomaly is LOUD (ValueError naming the episode key): a missing field, a
proprios length != num_frames, a non-zero video start, a video path not on disk.
Callers route these to the failures jsonl — never silently skip or repair.

Key rule is identical to XPolicyLab
``bp/tools/make_synthetic_zdemo_feats.py:keys_from_dataset_root``:
``<task_dir>/<episode_stem>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CAMERAS = ("ego", "wrist_left", "wrist_right")
STATE_KEYS = ("left_arm_joint", "left_gripper_pos", "right_arm_joint", "right_gripper_pos")


def iter_episodes(root) -> list:
    """Sorted ``[(key, json_path)]`` for every ``<root>/<task_dir>/episode_*.json``."""
    root = Path(root)
    return sorted((f"{p.parent.name}/{p.stem}", p) for p in root.glob("*/episode_*.json"))


def load_episode(json_path) -> dict:
    """One episode json -> {"key", "num_frames", "state" [N,14] f32, "videos" {cam: path}}."""
    path = Path(json_path)
    key = f"{path.parent.name}/{path.stem}"
    data = json.loads(path.read_text())
    for field in ("num_frames", "proprios", "observations"):
        if field not in data:
            raise ValueError(f"{key}: episode json lacks {field!r}")
    n = int(data["num_frames"])
    parts = []
    for k in STATE_KEYS:
        try:
            arr = np.asarray(data["proprios"][k], dtype=np.float32)
        except KeyError as err:
            raise ValueError(f"{key}: proprios lacks {k!r}") from err
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"{key}: proprios.{k} shape {arr.shape}, expected [{n}, d] (num_frames)")
        parts.append(arr)
    videos = {}
    for cam in CAMERAS:
        try:
            entry = data["observations"][cam][0]
        except (KeyError, IndexError, TypeError) as err:
            raise ValueError(f"{key}: observations.{cam}[0] missing") from err
        start = int(entry.get("start", 0))
        if start != 0:
            raise ValueError(f"{key}: observations.{cam}[0].start={start} != 0 — unsupported layout")
        try:
            vpath = Path(entry["path"])
        except KeyError as err:
            raise ValueError(f"{key}: observations.{cam}[0] lacks 'path'") from err
        if not vpath.is_file():
            raise ValueError(f"{key}: video for {cam} not on disk: {vpath}")
        videos[cam] = str(vpath)
    return {"key": key, "num_frames": n, "state": np.concatenate(parts, axis=1), "videos": videos}


def audit(root, check_videos: bool = True) -> int:
    """Preflight audit: every episode loads; optionally every video frame-count matches.

    Returns 0 when every episode passes; prints the per-chunk breakdown. Any anomaly
    prints one line per bad episode (and the count) — the caller decides to STOP.
    """
    from collections import Counter

    episodes = iter_episodes(root)
    per_chunk: Counter = Counter(key.split("/", 1)[0] for key, _ in episodes)
    print(f"scanned {len(episodes)} episode jsons under {root}")
    for chunk in sorted(per_chunk):
        print(f"  {chunk}: {per_chunk[chunk]}")
    bad = 0
    decoders = {}
    for i, (key, path) in enumerate(episodes, 1):
        try:
            ep = load_episode(path)
            if check_videos:
                from torchcodec.decoders import VideoDecoder

                for cam, vp in ep["videos"].items():
                    dec = decoders.get(vp)
                    if dec is None:
                        dec = decoders[vp] = VideoDecoder(vp)
                    nf = int(dec.metadata.num_frames)
                    if nf != ep["num_frames"]:
                        raise ValueError(f"{cam} video has {nf} frames, num_frames={ep['num_frames']}")
        except Exception as e:  # noqa: BLE001 - audit reports, never repairs
            bad += 1
            print(f"BAD {key}: {type(e).__name__}: {e}")
        if i % 500 == 0:
            print(f"  ... {i}/{len(episodes)} audited")
        if len(decoders) > 64:
            decoders.clear()
    print(f"audit: {len(episodes) - bad} OK, {bad} bad")
    return 1 if bad or len(episodes) != 3500 else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="converted dataset root (holds chunk-*/episode_*.json)")
    ap.add_argument("--check-videos", action="store_true", help="also open every mp4 and compare frame counts")
    ap.add_argument("--load-one", help="debug: load this episode json and print a summary")
    args = ap.parse_args()
    if args.load_one:
        ep = load_episode(args.load_one)
        print({k: (v if k != "state" else f"{v.shape} {v.dtype}") for k, v in ep.items()})
        return
    raise SystemExit(audit(args.root, check_videos=args.check_videos))


if __name__ == "__main__":
    main()
