"""bpv4_bpen dataset: one item = one trajectory (cached keyframe ViT features) + S queries on that trajectory.

Per query: progress p, the state at p, the 50-step action chunk after p (delta + normalized exactly like v3 / pt-15w),
language tokens, and two keyframe-visibility proposals (event segment / ratio window). Whether a query is masked and
which proposal is used is decided by the policy from its own training step (workers do not know the global step).
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK, Q35ComposeFieldsTransformFn, Q35DeltaActionTransformFn
from lerobot.transforms.core import (
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    hydrate_compose_field_transform,
    hydrate_delta_action_transform,
    hydrate_normalize_transform,
)
from lerobot.utils.constants import ACTION, OBS_STATE

from .build_feature_cache import cache_path
from .configuration_bpv4_bpen import BPV4BPEnDatasetConfig
from .keyframes import event_segment


def _holdout(repo: str, rules: dict[str, int]) -> int:
    for key, n in rules.items():
        if key in str(repo):
            return int(n)
    return 0


class _Repo:
    """State / action arrays, task strings and hydrated state-action transforms of one repo (no video decoding)."""

    def __init__(self, repo: str, cfg, train_cfg) -> None:
        from lerobot.datasets.factory import _load_external_stats_for_dataset

        self.repo = repo
        meta = LeRobotDatasetMetadata(repo)
        holder = SimpleNamespace(meta=meta)
        _load_external_stats_for_dataset(train_cfg, holder)
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
        files = sorted(Path(repo, "data").rglob("*.parquet"))
        df = pd.concat(pd.read_parquet(f, columns=["episode_index", OBS_STATE, ACTION]) for f in files)
        self.states, self.actions = {}, {}
        for ep, g in df.groupby("episode_index"):
            self.states[int(ep)] = np.stack(g[OBS_STATE].to_numpy()).astype(np.float32)
            self.actions[int(ep)] = np.stack(g[ACTION].to_numpy()).astype(np.float32)
        self.tasks = {int(e["episode_index"]): (e["tasks"][0] if len(e["tasks"]) else "") for e in meta.episodes}
        self.total_episodes = int(meta.total_episodes)

    def query(self, episode: int, frame: int, horizon: int) -> dict[str, torch.Tensor]:
        state, actions = self.states[episode], self.actions[episode]
        idx = frame + np.arange(horizon)
        pad = idx > len(actions) - 1
        data = {
            OBS_STATE: torch.from_numpy(state[frame].copy()),
            ACTION: torch.from_numpy(actions[np.minimum(idx, len(actions) - 1)].copy()),
            f"{ACTION}_is_pad": torch.from_numpy(pad),
            "robot_type": self.robot_type,
        }
        for t in self.transforms:
            data = t(data)
        return {OBS_STATE: data[OBS_STATE].float(), ACTION: data[ACTION].float(), ACTION_VALID_MASK: data[ACTION_VALID_MASK]}


class BPV4BPEnDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV4BPEnDatasetConfig, train_cfg, repo_ids: list[str], chunk_size: int) -> None:
        self.cfg, self.chunk_size = cfg, int(chunk_size)
        self.repos = [_Repo(r, cfg, train_cfg) for r in repo_ids]
        self.items: list[tuple[int, int]] = []
        for ri, repo in enumerate(self.repos):
            n = repo.total_episodes
            hold = _holdout(repo.repo, dict(cfg.val_holdout_last))
            eps = range(n) if cfg.split == "all" else range(n - hold, n) if cfg.split == "val" else range(n - hold)
            for ep in eps:
                if not cache_path(cfg.bp_feature_cache_dir, repo.repo, ep).is_file():
                    raise FileNotFoundError(f"missing feature cache for {repo.repo} episode {ep}; run build_feature_cache")
                self.items.append((ri, ep))
        self.num_episodes = len(self.items)
        self.num_frames = sum(len(self.repos[ri].states[ep]) for ri, ep in self.items)
        self._tokenizer = None
        self._token_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        logging.info("bpv4_bpen dataset split=%s: %d trajectories, %d frames, %d repos",
                     cfg.split, self.num_episodes, self.num_frames, len(self.repos))

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        return {self.repos[0].robot_type: self.repos[0].stats} if self.repos else {}

    def _tokens(self, ri: int, ep: int) -> tuple[torch.Tensor, torch.Tensor]:
        key = (ri, ep)
        if key not in self._token_cache:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.qwen3_5_processor_path)
            out = self._tokenizer(self.repos[ri].tasks.get(ep, ""), max_length=self.cfg.language_max_length,
                                  padding="max_length", truncation=True, return_tensors="pt")
            self._token_cache[key] = (out["input_ids"][0], out["attention_mask"][0].bool())
        return self._token_cache[key]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        ri, ep = self.items[i]
        repo = self.repos[ri]
        rec = torch.load(cache_path(cfg.bp_feature_cache_dir, repo.repo, ep), map_location="cpu", weights_only=False)
        k_max, k_eff = int(cfg.bp_num_keyframes), int(rec["frames"].shape[0])
        feats = rec["feats"]
        pad_feats = torch.zeros((k_max, *feats.shape[1:]), dtype=feats.dtype)
        pad_feats[:k_eff] = feats
        time = torch.zeros(k_max)
        time[:k_eff] = rec["time_ratio"]
        kf_valid = torch.zeros(k_max, dtype=torch.bool)
        kf_valid[:k_eff] = True
        frames = np.full(k_max, -10**9, dtype=np.int64)
        frames[:k_eff] = rec["frames"].numpy()
        flips, active, length = rec["flips"].numpy(), tuple(rec["active"]), int(rec["length"])
        lo, hi = active

        s = int(cfg.queries_per_trajectory)
        edges = np.linspace(0, length, s + 1).astype(int)
        q_frames = [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]
        ids, attn = self._tokens(ri, ep)
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK, "observation.input_ids", "observation.attention_mask",
                               "bp.vis_event", "bp.event_ok", "bp.vis_window")}
        valid_np = kf_valid.numpy()
        n_valid = int(valid_np.sum())
        for f in q_frames:
            q = repo.query(ep, f, self.chunk_size)
            for key in (OBS_STATE, ACTION, ACTION_VALID_MASK):
                out[key].append(q[key])
            drop_lang = bool(torch.rand(()) < cfg.language_drop_prob) or not cfg.use_language
            out["observation.input_ids"].append(ids)
            out["observation.attention_mask"].append(torch.zeros_like(attn) if drop_lang else attn)
            # event segment: hide (s, e] around f
            seg_s, seg_e = event_segment(flips, active, f)
            hide = (frames > seg_s) & (frames <= seg_e) & valid_np
            ok = 1 <= int(hide.sum()) and n_valid - int(hide.sum()) >= int(cfg.mask_min_visible_keyframes)
            out["bp.vis_event"].append(torch.from_numpy(valid_np & ~hide))
            out["bp.event_ok"].append(torch.tensor(ok))
            # ratio window around f
            r_lo, r_hi = cfg.mask_window_ratio
            span = hi - lo + 1
            win = max(1, int(round(float(torch.empty(()).uniform_(r_lo, r_hi)) * span)))
            start = float(torch.empty(()).uniform_(f - win, f))
            start = int(round(min(max(start, lo), max(lo, hi - win))))
            hide_w = (frames >= start) & (frames <= start + win) & valid_np
            if not hide_w.any() and n_valid > 1:
                hide_w[int(np.argmin(np.where(valid_np, np.abs(frames - f), 10**9)))] = True
            out["bp.vis_window"].append(torch.from_numpy(valid_np & ~hide_w))
        batch = {key: torch.stack(v) for key, v in out.items()}
        progress = torch.tensor([f / max(1, length - 1) for f in q_frames], dtype=torch.float32)
        batch.update({
            "bp.feats": pad_feats,
            "bp.time": time,
            "bp.kf_valid": kf_valid,
            "query.progress": progress,
            "query.progress_drop": torch.rand(s) < cfg.progress_drop_prob,
            "query.frame": torch.tensor(q_frames, dtype=torch.long),
            "query.u_mask": torch.rand(s),
            "query.u_type": torch.rand(s),
            "bp.traj_index": torch.tensor(i, dtype=torch.long),
        })
        return batch


def make_bpv4_bpen_dataset(train_cfg) -> tuple[BPV4BPEnDataset, dict]:
    from lerobot.datasets.factory import resolve_repo_ids

    repo_ids = resolve_repo_ids(train_cfg)
    ds = BPV4BPEnDataset(train_cfg.dataset, train_cfg, repo_ids, chunk_size=int(train_cfg.policy.chunk_size))
    return ds, ds.stats
