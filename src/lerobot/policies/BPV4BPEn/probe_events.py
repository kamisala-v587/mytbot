"""Position probes over ALL gripper interaction events, per task, with / without removing the grasp-arm mean.

Events of a trajectory: every gripper close (either arm) in time order -> event k = 1, 2, 3, ...; its "grasp" point is
the closing arm's link6 position (FK, robot base frame), its "place" point the same arm's position at the next opening.
Press-type tasks (the gripper closes to press, not to grasp) are reported separately as "contact".

Targets: event-k position minus the train mean of (task, k, arm) ["arm" residual, as in probe_eval] or of (task, k)
["task" residual: left / right side information kept]. Linear-kernel ridge on all BP tokens (see probe_eval.ridge_r2).
  pooled    one probe for all tasks (as in probe_eval); R² over all validation episodes and split per task
  per_task  a separate probe per task (train / validation episodes of that task only)
Features: raw ViT head-camera tokens of keyframe 0, BP tokens with all keyframes / only keyframe 0, for a random
encoder and the given checkpoints.

Usage: python -m lerobot.policies.BPV4BPEn.probe_events --out out.json --ckpt main10k=<...> --ckpt nomask10k=<...>
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import sinusoidal_encoding
from lerobot.policies.BPV4BPEn.probe_eval import episodes, gram, load_modules, ridge_r2
from lerobot.policies.BPV4BPEn.probe_shortcut import Trajectories

URDF = "/vla/workspace/my_robotwin/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf"
ARMS = {"L": ([f"fl_joint{i}" for i in range(1, 7)], "fl_link6", slice(0, 6), 6),
        "R": ([f"fr_joint{i}" for i in range(1, 7)], "fr_link6", slice(7, 13), 13)}
PRESS_TASKS = {"click_bell", "click_alarmclock", "press_stapler", "turn_switch"}
MAX_EVENTS = 3


def repo_events(repo: str):
    sys.path.append("/vla/.conda/miniconda3/envs/myrobotwin/lib/python3.10/site-packages")
    import yourdfpy

    robot = yourdfpy.URDF.load(URDF, load_meshes=False, build_scene_graph=True)
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "observation.state"])
                   for f in sorted(Path(repo, "data").rglob("*.parquet")))
    out = {}
    for ep, g in df.groupby("episode_index"):
        s = np.stack(g["observation.state"].to_numpy()).astype(np.float64)

        def fk(arm, frame):
            joints, link, sl, _ = ARMS[arm]
            robot.update_cfg(dict(zip(joints, s[frame, sl])))
            return robot.get_transform(link, robot.base_link)[:3, 3].tolist()

        events = []
        for arm, (_, _, _, gd) in ARMS.items():
            closed = s[:, gd] < 0.5
            for c in np.nonzero(closed[1:] & ~closed[:-1])[0] + 1:
                opens = np.nonzero(~closed[c + 1:] & closed[c:-1])[0]
                o = int(c + 1 + opens[0]) if len(opens) else None
                events.append({"arm": arm, "close": int(c), "open": o, "grasp": fk(arm, int(c)),
                               "place": fk(arm, o) if o is not None else None})
        out[int(ep)] = sorted(events, key=lambda e: e["close"])
    return repo, out


def r2_by_task(res: dict, task: np.ndarray) -> dict:
    idx, pred, true = np.array(res["val_index"]), np.array(res["pred"]), np.array(res["true"])
    out = {}
    for t in np.unique(task[idx]):
        m = task[idx] == t
        if m.sum() >= 5:
            sst = ((true[m] - true[m].mean(0)) ** 2).sum(0)
            out[t] = float(np.mean(1 - ((pred[m] - true[m]) ** 2).sum(0)[:2] / np.clip(sst[:2], 1e-12, None)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_file", default="/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_clean50_rand100.txt")
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2")
    ap.add_argument("--ckpt", action="append", default=[])
    args = ap.parse_args()

    items = episodes(args.repo_file)
    repos = sorted({r for r, _, _ in items})
    with ProcessPoolExecutor(32) as ex:
        ev_by_repo = dict(ex.map(repo_events, repos))
    events = [ev_by_repo[r].get(e, []) for r, e, _ in items]
    print("events done", flush=True)

    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}
    first = [ev[0]["close"] if ev else 0 for ev in events]
    loader = torch.utils.data.DataLoader(Trajectories(items, args.cache_dir, first), batch_size=32, num_workers=16)
    feats = {"vit_head_kf0": []}
    for m in models:
        feats[f"{m}.all"], feats[f"{m}.kf0"] = [], []
    with torch.no_grad():
        for batch in loader:
            x, t, vis = batch["feats"].cuda(), batch["time"].cuda(), batch["vis"].cuda()
            b = x.shape[0]
            feats["vit_head_kf0"].append(x[:, 0, 0].reshape(b, -1).half().cpu())
            for name, (enc, res) in models.items():
                content = enc(x)
                content = content + sinusoidal_encoding(t, content.shape[-1])[:, :, None]
                q = content.shape[2]
                tokens = content.reshape(b, -1, content.shape[-1])
                for ci, c in ((0, "all"), (1, "kf0")):
                    key_vis = vis[:, ci, :, None].expand(-1, -1, q).reshape(b, -1)
                    feats[f"{name}.{c}"].append(res(tokens, key_vis).reshape(b, -1).half().cpu())
    feats = {k: torch.cat(v).numpy() for k, v in feats.items()}
    print("features done", flush=True)

    task = np.array([Path(r).parent.name for r, _, _ in items])
    is_val = np.array([v for _, _, v in items])
    kind = np.where(np.isin(task, list(PRESS_TASKS)), "contact", "grasp")

    # targets: (event k, point, residual mode) -> (y [N,3], ok [N])
    targets = {}
    for k in range(1, MAX_EVENTS + 1):
        for point in ("grasp", "place"):
            y, ok = np.zeros((len(items), 3)), np.zeros(len(items), bool)
            arm = np.array([""] * len(items), dtype=object)
            for i, ev in enumerate(events):
                if len(ev) >= k and ev[k - 1][point] is not None:
                    y[i], ok[i], arm[i] = ev[k - 1][point], True, ev[k - 1]["arm"]
            for mode in ("arm", "task"):
                group = np.array([f"{t}|{a}" if mode == "arm" else t for t, a in zip(task, arm)])
                r = y.copy()
                good = ok.copy()
                for g in np.unique(group[ok]):
                    tr_g = ok & ~is_val & (group == g)
                    if tr_g.sum() < 3:
                        good &= group != g
                        continue
                    r[group == g] -= y[tr_g].mean(0)
                targets[(k, point, mode)] = (r, good)

    result = {"counts": {}, "pooled": {}, "per_task": {}}
    for (k, point, mode), (_, ok) in targets.items():
        if mode == "arm":
            result["counts"][f"{point}{k}"] = {kd: {"train": int((ok & ~is_val & (kind == kd)).sum()),
                                                    "val": int((ok & is_val & (kind == kd)).sum()),
                                                    "tasks": int(len(np.unique(task[ok & (kind == kd)])))}
                                               for kd in ("grasp", "contact")}
    for name, x in feats.items():
        g = gram(x, ~is_val)
        for (k, point, mode), (y, ok) in targets.items():
            for kd in ("grasp", "contact"):
                if kd == "contact" and (point == "place" or k > 1):
                    continue
                sel = ok & (kind == kd)
                if (sel & is_val).sum() < 20:
                    continue
                res = ridge_r2(g, sel & ~is_val, sel & is_val, y)
                key = f"{kd}|{point}{k}|{mode}"
                result["pooled"].setdefault(key, {})[name] = {
                    "r2_xyz": res["r2_xyz"], "r2_xy": float(np.mean(res["r2_xyz"][:2])), "by_task": r2_by_task(res, task)}
        # separate probe per task for the first grasp / place (arm residual)
        for point in ("grasp", "place"):
            y, ok = targets[(1, point, "arm")]
            per = {}
            for t in np.unique(task[ok]):
                sel = ok & (task == t)
                if (sel & is_val).sum() >= 5 and (sel & ~is_val).sum() >= 30:
                    res = ridge_r2(g, sel & ~is_val, sel & is_val, y)
                    per[t] = float(np.mean(res["r2_xyz"][:2]))
            result["per_task"].setdefault(f"{point}1|arm", {})[name] = per
        del g
        torch.cuda.empty_cache()
        print(name, {k: round(v[name]["r2_xy"], 3) for k, v in result["pooled"].items() if name in v}, flush=True)
    Path(args.out).write_text(json.dumps(result))


if __name__ == "__main__":
    main()
