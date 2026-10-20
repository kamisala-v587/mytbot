"""Grasp / place point labels by forward kinematics.

For every episode: grasp = first gripper close (either arm); place = the next opening of the same gripper.
The point is the arm's link6 origin in the robot base frame, so left- and right-arm episodes share one frame.
yourdfpy is borrowed from the `myrobotwin` env (appended to sys.path, so this env's packages take priority).

Usage: python probe_labels.py <repo_file> <out.json>
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append("/vla/.conda/miniconda3/envs/myrobotwin/lib/python3.10/site-packages")
import yourdfpy  # noqa: E402

URDF = "/vla/workspace/my_robotwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
ARMS = {
    "L": ([f"fl_joint{i}" for i in range(1, 7)], "fl_link6", slice(0, 6), 6),
    "R": ([f"fr_joint{i}" for i in range(1, 7)], "fr_link6", slice(7, 13), 13),
}


def first_close(state: np.ndarray):
    best = None
    for arm, (_, _, _, g) in ARMS.items():
        closed = state[:, g] < 0.5
        idx = np.nonzero(closed[1:] & ~closed[:-1])[0]
        if len(idx) and (best is None or idx[0] + 1 < best[1]):
            best = (arm, int(idx[0] + 1))
    return best


def main() -> None:
    repos = [l.strip() for l in Path(sys.argv[1]).read_text().splitlines() if l.strip()]
    robot = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)
    base = robot.base_link
    out = {}
    for repo in repos:
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", "observation.state"])
                       for f in sorted(Path(repo, "data").rglob("*.parquet")))
        for ep, g in df.groupby("episode_index"):
            s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)
            ev = first_close(s)
            rec = {"arm": None, "grasp": None, "place": None, "grasp_frame": None, "place_frame": None}
            if ev is not None:
                arm, t = ev
                joints, link, sl, gdim = ARMS[arm]

                def fk(frame: int) -> list[float]:
                    robot.update_cfg(dict(zip(joints, s[frame, sl])))
                    return robot.get_transform(link, base)[:3, 3].tolist()

                rec.update(arm=arm, grasp_frame=t, grasp=fk(t))
                closed = s[:, gdim] < 0.5
                opens = np.nonzero(~closed[t + 1 :] & closed[t:-1])[0]
                if len(opens):
                    tp = int(t + 1 + opens[0])
                    rec.update(place_frame=tp, place=fk(tp))
            out[f"{repo}|{int(ep)}"] = rec
        print(repo, flush=True)
    Path(sys.argv[2]).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
