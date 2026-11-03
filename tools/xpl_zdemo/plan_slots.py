"""Pinned S-U embodiment slot planning for the XR1 F1 Z_demo feature cache.

The sampler is IMPORTED from upstream my_tbot (never re-implemented):
``BPV4CDX.sampling.sample_su_emb`` with a directly-constructed ``Embodiment``
describing the converted RoboDojo arx_x5-ee layout:

    state_keys [left_arm_joint, left_gripper_pos, right_arm_joint, right_gripper_pos]
    state_sizes [6, 1, 6, 1]  -> state [N, 14]
    mask True*6 + False + True*6 + False  (True = joint, False = gripper; dims 6/13)
    cameras [ego, wrist_left, wrist_right]  (slot order image0/1/2)

This IS the dual-arm 6+1 layout the embodiment helpers were written for
(upstream embodiment.py:44-50). ``plan_for`` asserts the pinned sampler version —
a drift means the my_tbot checkout moved (STOP, report HEAD).
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np

from lerobot.policies.BPV4CDX.embodiment import Embodiment
from lerobot.policies.BPV4CDX.sampling import sample_su_emb

SAMPLER_VERSION = "su-uniform_active_v2_embodiment"
NUM_SLOTS = 32
MYTBOT_ROOT = Path(__file__).resolve().parents[2]

EMBODIMENT = Embodiment(
    robot_type="xpl_arx_x5",
    resolved="xpl_arx_x5_dual_arm_61",
    state_keys=["left_arm_joint", "left_gripper_pos", "right_arm_joint", "right_gripper_pos"],
    action_keys=[],  # unused by the sampler
    state_sizes=[6, 1, 6, 1],
    action_sizes=[],
    mask=np.array([True] * 6 + [False] + [True] * 6 + [False]),
    cameras=["ego", "wrist_left", "wrist_right"],
)


def plan_for(episode: dict):
    """episode (episode_source.load_episode) -> upstream SlotPlan, version/range-checked."""
    plan = sample_su_emb(episode["state"], EMBODIMENT, NUM_SLOTS)
    version = plan.version["version"]
    if version != SAMPLER_VERSION:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(MYTBOT_ROOT)).stdout.strip()
        raise RuntimeError(
            f"sampler version drift: plan.version={version!r}, pinned {SAMPLER_VERSION!r}; "
            f"my_tbot HEAD is {head!r} (expected 94859f7...) — STOP, do not adapt")
    frames = np.asarray(plan.frames)
    n = int(episode["num_frames"])
    if len(frames) != NUM_SLOTS or frames.min() < 0 or frames.max() >= n:
        raise ValueError(
            f"{episode['key']}: plan.frames out of range [{frames.min()}, {frames.max()}] "
            f"for num_frames={n}, len={len(frames)}")
    return plan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=200, help="episodes scanned for the idle-prefix check")
    ap.add_argument("--root", help="converted dataset root (for the default 3-episode report)")
    ap.add_argument("--episode", action="append", default=[], help="episode json to plan (repeatable)")
    args = ap.parse_args()

    def report(ep):
        plan = plan_for(ep)
        idle = plan.active[0]
        print(f"{ep['key']}: num_frames={ep['num_frames']} active={plan.active} "
              f"slots={len(plan.frames)} repeats={plan.repeats} version={plan.version['version']}")
        print(f"  idle prefix excluded by active span: {idle} frames "
              f"({'yes' if idle > 0 else 'no idle prefix on this episode'})")

    if args.episode:
        from episode_source import load_episode

        for p in args.episode:
            report(load_episode(p))
        return
    if not args.root:
        ap.error("--root or --episode is required")
    from episode_source import iter_episodes, load_episode

    episodes = iter_episodes(args.root)
    for key, path in episodes[:3]:
        report(load_episode(path))
    # self-check on one episode with an obvious idle prefix (max lo over a sample)
    best, best_lo = None, -1
    for key, path in episodes[: args.sample]:
        ep = load_episode(path)
        lo = plan_for(ep).active[0]
        if lo > best_lo:
            best, best_lo = (key, path), lo
    print(f"largest idle prefix in the first {args.sample} episodes: {best[0]} (lo={best_lo})")
    report(load_episode(best[1]))


if __name__ == "__main__":
    main()
