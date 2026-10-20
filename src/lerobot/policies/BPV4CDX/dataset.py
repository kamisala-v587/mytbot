"""bpv4cdx Stage A-AE dataset: one item = one demo trajectory + S target segments of that same trajectory.

Demo side: 32-slot cached ViT features (pooled to 5 tokens per image, or all 64), slot times, camera validity.
Target side per segment: tau_start / tau_end / valid length, the 50-step action chunk after the segment start
(delta w.r.t. the segment-start state, normalized with the external stats, exactly like bpv4_bpen / q35), the
segment-start state (used only by the AE-S0 branch) and the canonical or generic text tokens.
The demo action is only a label: nothing on the target side is fed to the video encoder.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
from lerobot.policies.BPV4BPEn.dataset import _Repo
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdx import BPV4CDXAEDatasetConfig
from .paths import GENERIC_TEXT, canonical_text, read_repo_list, split_episodes, task_name


def pool_spatial(feats: torch.Tensor, tokens: int) -> torch.Tensor:
    """[K, C, 64, D] (8x8 grid, row-major) -> [K, C, 5, D] (2x2 block means + global mean) or unchanged for 64."""
    if tokens == 64:
        return feats
    k, c, t, d = feats.shape
    grid = feats.float().reshape(k, c, 2, 4, 2, 4, d).mean(dim=(3, 5)).reshape(k, c, 4, d)
    return torch.cat([grid, feats.float().mean(2, keepdim=True)], dim=2).to(feats.dtype)


def relative_key(repo: str, ep: int) -> str:
    p = Path(repo)
    return f"{p.parent.name}/{p.name}|{int(ep)}"


class BPV4CDXAEDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV4CDXAEDatasetConfig, train_cfg, repo_ids: list[str], chunk_size: int) -> None:
        if int(chunk_size) != int(cfg.segment_len):
            raise ValueError("dataset.segment_len must equal policy.chunk_size")
        self.cfg, self.chunk_size = cfg, int(chunk_size)
        self.repos = [_Repo(r, cfg, train_cfg) for r in repo_ids]
        tasks = sorted({task_name(r) for r in repo_ids})
        self.task_index = {t: i for i, t in enumerate(tasks)}
        whitelist = set(cfg.episode_whitelist)
        self.items: list[tuple[int, int]] = []
        for ri, repo in enumerate(self.repos):
            for ep in split_episodes(repo.repo, repo.total_episodes, cfg.split, dict(cfg.val_holdout_last)):
                if whitelist and relative_key(repo.repo, ep) not in whitelist:
                    continue
                if not cache_path(cfg.cache_dir, repo.repo, ep).is_file():
                    raise FileNotFoundError(f"missing demo cache for {repo.repo} episode {ep}; run BPV4CDX.build_cache")
                self.items.append((ri, ep))
        if cfg.max_trajectories:
            self.items = self.items[: int(cfg.max_trajectories)]
        self.num_episodes = len(self.items)
        self.num_frames = sum(len(self.repos[ri].states[ep]) for ri, ep in self.items)
        self._tokenizer = None
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        logging.info("bpv4cdx_ae dataset split=%s: %d trajectories, %d frames, %d repos",
                     cfg.split, self.num_episodes, self.num_frames, len(self.repos))

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        return {self.repos[0].robot_type: self.repos[0].stats} if self.repos else {}

    def tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._text_cache:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.qwen3_5_processor_path)
            out = self._tokenizer(text, max_length=self.cfg.language_max_length, padding="max_length",
                                  truncation=True, return_tensors="pt")
            self._text_cache[text] = (out["input_ids"][0], out["attention_mask"][0].bool())
        return self._text_cache[text]

    def demo(self, i: int) -> dict[str, torch.Tensor]:
        """Demo-side tensors of trajectory i (the only inputs of the video encoder)."""
        cfg = self.cfg
        ri, ep = self.items[i]
        rec = torch.load(cache_path(cfg.cache_dir, self.repos[ri].repo, ep), map_location="cpu", weights_only=False)
        if rec["feats"].shape[0] != cfg.num_slots:
            raise ValueError(f"cache has {rec['feats'].shape[0]} slots, expected {cfg.num_slots}")
        feats = pool_spatial(rec["feats"], int(cfg.spatial_tokens))
        return {
            "demo.feats": feats,
            "demo.time": rec["time_ratio"].float(),
            "demo.camera_valid": torch.ones(feats.shape[0], feats.shape[1], dtype=torch.bool),
        }

    def segment_starts(self, length: int, mode: str | None = None) -> list[int]:
        mode = mode or self.cfg.segment_mode
        s, seg = int(self.cfg.segments_per_trajectory), int(self.cfg.segment_len)
        if mode == "fixed":
            starts = list(range(0, length, seg))
            return [starts[j % len(starts)] for j in range(s)]
        edges = np.linspace(0, length, s + 1).astype(int)
        return [int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item()) for j in range(s)]

    def target(self, i: int, starts: list[int], generic: list[bool] | None = None) -> dict[str, torch.Tensor]:
        """Target-side tensors of trajectory i for the given segment starts (labels + text + segment query)."""
        cfg = self.cfg
        ri, ep = self.items[i]
        repo = self.repos[ri]
        length = len(repo.states[ep])
        if generic is None:
            generic = [bool(torch.rand(()) < cfg.generic_text_prob) for _ in starts]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK, "text.input_ids", "text.attention_mask")}
        tau, steps = [], []
        for f, g in zip(starts, generic):
            q = repo.query(ep, f, self.chunk_size)
            for key in (OBS_STATE, ACTION, ACTION_VALID_MASK):
                out[key].append(q[key])
            ids, attn = self.tokens(GENERIC_TEXT if g else canonical_text(repo.repo))
            out["text.input_ids"].append(ids)
            out["text.attention_mask"].append(attn)
            valid = min(self.chunk_size, length - f)
            tau.append([f / max(1, length - 1), (f + valid - 1) / max(1, length - 1)])
            steps.append(valid / self.chunk_size)
        batch = {key: torch.stack(v) for key, v in out.items()}
        batch.update({
            "seg.tau": torch.tensor(tau, dtype=torch.float32),
            "seg.steps": torch.tensor(steps, dtype=torch.float32),
            "seg.start": torch.tensor(starts, dtype=torch.long),
            "text.is_generic": torch.tensor(generic, dtype=torch.bool),
        })
        return batch

    def meta(self, i: int) -> dict[str, torch.Tensor]:
        ri, ep = self.items[i]
        return {
            "demo.index": torch.tensor(i, dtype=torch.long),
            "demo.task_index": torch.tensor(self.task_index[task_name(self.repos[ri].repo)], dtype=torch.long),
            "demo.length": torch.tensor(len(self.repos[ri].states[ep]), dtype=torch.long),
        }

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ri, ep = self.items[i]
        starts = self.segment_starts(len(self.repos[ri].states[ep]))
        return {**self.demo(i), **self.target(i, starts), **self.meta(i)}


def make_bpv4cdx_ae_dataset(train_cfg) -> tuple[BPV4CDXAEDataset, dict]:
    cfg = train_cfg.dataset
    if not cfg.repo_id_file:
        raise ValueError("bpv4cdx_ae needs dataset.repo_id_file")
    repo_ids = read_repo_list(cfg.repo_id_file, cfg.data_root)
    ds = BPV4CDXAEDataset(cfg, train_cfg, repo_ids, chunk_size=int(train_cfg.policy.chunk_size))
    return ds, ds.stats
