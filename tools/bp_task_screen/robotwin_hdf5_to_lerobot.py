"""Convert RoboTwin HDF5 episodes (script/my_collect_data.py output) into a LeRobot dataset laid out like
/vla/workspace/data/robotwin2.0/<task>/aloha-agilex_clean_50:
  fps 30, observation.state = joint vector[t], action = joint vector[t + 1] (last frame dropped),
  cam_high / cam_left_wrist / cam_right_wrist upscaled 320x240 -> 640x480 (the released data is upscaled the same way).
scene_info.json (object ids / poses) is copied to <out>/meta/scene_info.json.

Usage: python robotwin_hdf5_to_lerobot.py <robotwin_task_dir> <out_dir> [--instruction_split seen]
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

REFERENCE = Path("/vla/workspace/data/robotwin2.0/place_shoe/aloha-agilex_clean_50/meta/info.json")
CAMERAS = {"head_camera": "observation.images.cam_high", "left_camera": "observation.images.cam_left_wrist",
           "right_camera": "observation.images.cam_right_wrist"}


def features() -> dict:
    ref = json.loads(REFERENCE.read_text())["features"]
    out = {}
    for key in ("observation.state", "action", *CAMERAS.values()):
        f = dict(ref[key])
        f.pop("info", None)
        f["shape"] = tuple(f["shape"])
        out[key] = f
    return out


def decode(buf) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(bytes(buf), np.uint8), cv2.IMREAD_COLOR)  # stored from RGB arrays
    return cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)


def instruction(src: Path, ep: int, split: str, task: str) -> str:
    f = src / "instructions" / f"episode{ep}.json"
    if f.exists():
        d = json.loads(f.read_text())
        items = d.get(split) or d.get("seen") or next(iter(d.values()), [])
        if items:
            return items[0]
    return task.replace("_", " ")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--instruction_split", default="seen")
    ap.add_argument("--episodes", default=None, help="start:end slice of the sorted source episodes (one shard)")
    args = ap.parse_args()
    src, out = Path(args.src), Path(args.out)
    task = src.parent.name
    eps = sorted(int(p.stem.replace("episode", "")) for p in (src / "data").glob("episode*.hdf5"))
    if args.episodes:
        lo, hi = (int(v) for v in args.episodes.split(":"))
        eps = eps[lo:hi]
    if out.exists():
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=f"bp_probe/{task}", fps=30, features=features(), root=out, robot_type="aloha",
                               use_videos=True, image_writer_threads=8)
    for ep in eps:
        with h5py.File(src / "data" / f"episode{ep}.hdf5", "r") as f:
            vec = f["joint_action/vector"][()].astype(np.float32)
            imgs = {cam: f[f"observation/{cam}/rgb"][()] for cam in CAMERAS}
        text = instruction(src, ep, args.instruction_split, task)
        for t in range(len(vec) - 1):
            frame = {"observation.state": vec[t], "action": vec[t + 1], "task": text}
            for cam, key in CAMERAS.items():
                frame[key] = decode(imgs[cam][t])
            ds.add_frame(frame)
        ds.save_episode()
        print(task, ep, len(vec) - 1, flush=True)
    ds.finalize()
    shutil.copy(src / "scene_info.json", out / "meta" / "scene_info.json")
    # LeRobot episode i of this repo = source episode source_episodes[i] (scene_info / HDF5 numbering)
    (out / "meta" / "source_episodes.json").write_text(json.dumps({"source_dir": str(src), "source_episodes": eps}))


if __name__ == "__main__":
    main()
