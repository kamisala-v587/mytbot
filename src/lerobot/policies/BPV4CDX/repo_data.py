"""One repo's state / action arrays and the q35 target transforms, for any embodiment (replaces bpv4 `_Repo`, which
reads only the aloha columns `observation.state` / `action`).

Arrays: the FEATURE_MAPPING columns of every episode, concatenated in mapping order (`states[ep]` [L, Ds],
`actions[ep]` [L, Da], raw units). They come from an array store when one exists (AE-3: one `state.npy` /
`action.npy` / `index.json` per repo, memory-mapped, so 8 training ranks do not each hold ~450 M frames), otherwise
from the parquet files (small datasets, AE-1 / AE-2).
Targets (`query`): exactly the q35 / bpv4 transform chain (delta w.r.t. the query-frame state on the MASK_MAPPING
dims -> normalization with the external stats per source key -> composition -> padding to max dims).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK, Q35ComposeFieldsTransformFn, Q35DeltaActionTransformFn
from lerobot.transforms.core import (NormalizeTransformFn, PadStateAndActionTransformFn, hydrate_compose_field_transform,
                                     hydrate_delta_action_transform, hydrate_normalize_transform)
from lerobot.utils.constants import ACTION, OBS_STATE

from .embodiment import Embodiment, embodiment_from_meta
from .paths import canonical_text, is_robotwin_like, rel_id, task_name


class EpView:
    """episode -> rows of a (memory-mapped) frame array."""

    def __init__(self, array: np.ndarray, spans: dict[int, tuple[int, int]]) -> None:
        self.array, self.spans = array, spans

    def __getitem__(self, ep: int) -> np.ndarray:
        s, e = self.spans[int(ep)]
        return np.array(self.array[s:e], dtype=np.float32)  # a writable copy (a memmap view is read-only)

    def __len__(self) -> int:
        return len(self.spans)

    def __contains__(self, ep) -> bool:
        return int(ep) in self.spans

    def keys(self):
        return self.spans.keys()

    def items(self):
        return ((ep, self[ep]) for ep in self.spans)

    def values(self):
        return (self[ep] for ep in self.spans)


def store_dir(store_root: str | Path, repo: str | Path) -> Path:
    return Path(store_root) / rel_id(repo)


def read_arrays(repo: str | Path, emb: Embodiment) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
    """All episodes of a repo from its parquet files -> (state [N, Ds], action [N, Da], {ep: (start, end)})."""
    cols = emb.state_keys + emb.action_keys
    files = sorted(Path(repo, "data").rglob("*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", *cols]) for f in files)
    df = df.sort_values(["episode_index", "frame_index"], kind="stable")

    def stack(keys):
        return np.concatenate([np.stack(df[k].to_numpy()).reshape(len(df), -1).astype(np.float32) for k in keys], 1)

    state, action = stack(emb.state_keys), stack(emb.action_keys)
    eps = df["episode_index"].to_numpy()
    bounds = np.flatnonzero(np.r_[True, eps[1:] != eps[:-1], True])
    spans = {int(eps[s]): (int(s), int(e)) for s, e in zip(bounds[:-1], bounds[1:])}
    return state, action, spans


def write_store(repo: str | Path, store_root: str | Path, emb: Embodiment | None = None) -> Path:
    """Write the array store of one repo (atomic per file); returns its directory."""
    if emb is None:
        meta = LeRobotDatasetMetadata(str(repo))
        emb = embodiment_from_meta(meta.robot_type, meta.features)
    out = store_dir(store_root, repo)
    out.mkdir(parents=True, exist_ok=True)
    state, action, spans = read_arrays(repo, emb)
    for name, arr in (("state", state), ("action", action)):
        tmp = out / f"{name}.tmp.npy"
        np.save(tmp, arr)
        os.replace(tmp, out / f"{name}.npy")
    index = {"episodes": [[ep, s, e] for ep, (s, e) in sorted(spans.items())], "state_keys": emb.state_keys,
             "action_keys": emb.action_keys, "state_sizes": emb.state_sizes, "action_sizes": emb.action_sizes}
    tmp = out / "index.tmp.json"
    tmp.write_text(json.dumps(index))
    os.replace(tmp, out / "index.json")
    return out


class RepoData:
    def __init__(self, repo: str, cfg, train_cfg, store_root: str | None = None, load_arrays: bool = True) -> None:
        """load_arrays=False: metadata and transforms only (`open_arrays` later), so the object can be built in a
        worker process and pickled back without copying the frame arrays."""
        from lerobot.datasets.factory import _load_external_stats_for_dataset

        self.repo = repo
        self.rel = rel_id(repo)
        meta = LeRobotDatasetMetadata(repo)
        holder = SimpleNamespace(meta=meta)
        _load_external_stats_for_dataset(train_cfg, holder)
        self.emb = embodiment_from_meta(meta.robot_type, meta.features)
        transforms = [
            Q35DeltaActionTransformFn(),
            NormalizeTransformFn(),
            Q35ComposeFieldsTransformFn(),
            PadStateAndActionTransformFn(max_state_dim=cfg.max_state_dim, max_action_dim=cfg.max_action_dim),
        ]
        for hydrate in (hydrate_delta_action_transform, hydrate_normalize_transform, hydrate_compose_field_transform):
            transforms = hydrate(transforms, holder)
        self.transforms = transforms
        self.stats = meta.stats
        self.robot_type = meta.robot_type
        self.total_episodes = int(meta.total_episodes)
        self.fps = int(meta.fps)
        # task identity: RoboTwin layout keeps the AE-1 / AE-2 task name; other datasets add the embodiment, since
        # the same task folder exists for several robots
        self.task = task_name(repo) if is_robotwin_like(repo) else f"{self.emb.resolved}/{task_name(repo)}"
        self.text = canonical_text(repo)
        self.states = self.actions = None
        if load_arrays:
            self.open_arrays(store_root)

    def has_store(self, store_root: str | None) -> bool:
        return bool(store_root) and (store_dir(store_root, self.repo) / "index.json").is_file()

    def open_arrays(self, store_root: str | None) -> None:
        if self.has_store(store_root):
            sd = store_dir(store_root, self.repo)
            index = json.loads((sd / "index.json").read_text())
            state = np.load(sd / "state.npy", mmap_mode="r")
            action = np.load(sd / "action.npy", mmap_mode="r")
            spans = {int(ep): (int(s), int(e)) for ep, s, e in index["episodes"]}
        else:
            state, action, spans = read_arrays(self.repo, self.emb)
        self.states, self.actions = EpView(state, spans), EpView(action, spans)

    def query(self, episode: int, frame: int, horizon: int) -> dict[str, torch.Tensor]:
        state, actions = self.states[episode], self.actions[episode]
        idx = frame + np.arange(horizon)
        pad = idx > len(actions) - 1
        act = actions[np.minimum(idx, len(actions) - 1)]
        # one padding flag per composed action field (Q35ComposeFieldsTransformFn requires all of them)
        data = {f"{k}_is_pad": torch.from_numpy(pad) for k in self.emb.action_keys}
        data["robot_type"] = self.robot_type
        for keys, sizes, row in ((self.emb.state_keys, self.emb.state_sizes, state[frame]),
                                 (self.emb.action_keys, self.emb.action_sizes, act)):
            start = 0
            for k, n in zip(keys, sizes):
                data[k] = torch.from_numpy(np.ascontiguousarray(row[..., start : start + n]))
                start += n
        for t in self.transforms:
            data = t(data)
        return {OBS_STATE: data[OBS_STATE].float(), ACTION: data[ACTION].float(), ACTION_VALID_MASK: data[ACTION_VALID_MASK]}
