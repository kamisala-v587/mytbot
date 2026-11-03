"""Checkpoint-free baseline: replay a demonstration of the requested candidate set open-loop.

For each episode (episode_uid) one demonstration of the candidate set's LeRobot repo is picked with a seeded RNG; the
response at step t (bp_step from the client) is that demonstration's actions [t, t + horizon), padded with its last
action. It ignores the current observation, so it measures how far copying the demonstration gets in a jittered
scene, and it tests the whole client / server / demonstration plumbing without a model.

replay_self=True: on training scenes (episode_uid '<set>|train<k>|r<rep>', client scene_source: train) replay the
scene's own recorded episode (training scene k of a set = LeRobot episode k of its repo). A recorded trajectory
replayed in its own scene should succeed; failures point at data / execution mismatches, not at the policy.
"""

from __future__ import annotations

import random
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class DemoReplayPolicy:
    name = "demo_replay"
    family = "bp"

    def __init__(self, index: dict[str, dict], horizon: int = 30, seed: int = 0, replay_self: bool = False) -> None:
        self.index = index
        self.horizon = int(horizon)
        self.seed = int(seed)
        self.replay_self = bool(replay_self)
        self._repos: dict[str, dict[int, np.ndarray]] = {}
        self._lock = threading.Lock()
        self._metadata = {
            "model_type": "demo_replay", "adapter": "demo_replay", "family": "bp",
            "deployment": "bprobotwin_demo_replay", "infer_horizon": self.horizon, "chunk_size": self.horizon,
            "action_dim": 14, "target_action_dim": 14, "action_mode": "abs",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def on_request(self, connection_id: str, obs: dict[str, Any]) -> None:
        return None

    def _episodes(self, repo: str) -> dict[int, np.ndarray]:
        with self._lock:
            if repo not in self._repos:
                files = sorted(Path(repo, "data").rglob("*.parquet"))
                df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "action"]) for f in files)
                df = df.sort_values(["episode_index", "frame_index"])
                self._repos[repo] = {int(ep): np.stack(g["action"].to_numpy()).astype(np.float32)
                                     for ep, g in df.groupby("episode_index")}
            return self._repos[repo]

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for obs in observations:
            cs = obs["bp_candidate_set"]
            repo = self.index[cs]["repo"]
            episodes = self._episodes(repo)
            keys = sorted(episodes)
            own = re.search(r"\|train(\d+)\|", str(obs.get("episode_uid") or "")) if self.replay_self else None
            if self.replay_self and own is None:
                raise ValueError("replay_self needs training scenes (client scene_source: train)")
            if own is not None:
                ep = keys[int(own.group(1))]
            else:
                ep = keys[random.Random(f"{self.seed}:{obs.get('episode_uid') or cs}").randrange(len(keys))]
            actions = episodes[ep]
            t = int(obs.get("bp_step", 0))
            chunk = actions[t: t + self.horizon]
            if len(chunk) < self.horizon:
                chunk = np.concatenate([chunk, np.repeat(actions[-1:], self.horizon - len(chunk), axis=0)])
            out.append({"actions": chunk, "action": chunk[0], "bp_demo_repo": repo, "bp_demo_episode": int(ep)})
        return out
