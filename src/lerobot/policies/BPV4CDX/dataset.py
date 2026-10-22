"""bpv4cdx Stage A-AE dataset: one item = one demo trajectory + S target segments of that same trajectory.

Demo side: 32-slot cached ViT features (pooled to 5 tokens per image, or all 64), slot times, camera validity.
Target side per segment: tau_start / tau_end / valid length, the 50-step action chunk after the segment start
(delta w.r.t. the segment-start state on the 6 joints of each arm, grippers absolute, normalized with the external
stats, exactly like bpv4_bpen / q35), the segment-start state (used only by the AE-S0 branch) and the canonical /
generic / dropped text tokens (48 tokens; the model prepends its own BOS).
The demo action is only a label: nothing on the target side is fed to the video encoder.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path as _task_variant_cache_path
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .configuration_bpv4cdx import BPV4CDXAEDatasetConfig
from .paths import GENERIC_TEXT, load_manifest, manifest_repos, read_repo_list, rel_id, split_episodes
from .repo_data import RepoData


def demo_cache_path(cache_dir: str | Path, repo: str | Path, ep: int, layout: str = "task_variant") -> Path:
    """task_variant: <cache>/<task>/<variant>/epXXXXX.pt (AE-1 / AE-2); rel: <cache>/<relative repo id>/epXXXXX.pt."""
    if layout == "rel":
        return Path(cache_dir) / rel_id(repo) / f"ep{int(ep):05d}.pt"
    return _task_variant_cache_path(cache_dir, repo, ep)


def load_cache_index(path: str | Path) -> set[str]:
    """Lines `<relative repo id>|<episode>` of the trajectories whose cache file exists."""
    return {line.strip() for line in Path(path).read_text().splitlines() if line.strip()}


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
    def __init__(self, cfg: BPV4CDXAEDatasetConfig, train_cfg, repo_ids: list[str], chunk_size: int,
                 heldout: dict[str, list[int]] | None = None) -> None:
        """heldout: repo path -> held-out episodes (split manifest); None = the val_holdout_last rule."""
        if int(chunk_size) != int(cfg.segment_len):
            raise ValueError("dataset.segment_len must equal policy.chunk_size")
        self.cfg, self.chunk_size = cfg, int(chunk_size)
        store = cfg.array_store or None
        # datasets / tqdm progress bars are not thread-safe when created concurrently ('tqdm' has no '_lock')
        import datasets
        from tqdm import tqdm

        datasets.disable_progress_bars()
        tqdm.get_lock()
        with ThreadPoolExecutor(max(1, int(cfg.load_threads))) as pool:
            self.repos = list(pool.map(lambda r: RepoData(r, cfg, train_cfg, store_root=store), repo_ids))
        self.heldout = heldout
        tasks = sorted({repo.task for repo in self.repos})
        self.task_index = {t: i for i, t in enumerate(tasks)}
        self._tokenizer = None
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._build_items()

    def cache_file(self, ri: int, ep: int) -> Path:
        return demo_cache_path(self.cfg.cache_dir, self.repos[ri].repo, ep, self.cfg.cache_layout)

    def _build_items(self) -> None:
        cfg = self.cfg
        whitelist = set(cfg.episode_whitelist)
        available = load_cache_index(cfg.cache_index) if cfg.cache_index else None
        self.items: list[tuple[int, int]] = []
        missing = 0
        for ri, repo in enumerate(self.repos):
            held = self.heldout.get(repo.repo) if self.heldout is not None else None
            for ep in split_episodes(repo.repo, repo.total_episodes, cfg.split, dict(cfg.val_holdout_last), held):
                if whitelist and relative_key(repo.repo, ep) not in whitelist and f"{repo.rel}|{ep}" not in whitelist:
                    continue
                if available is not None:
                    if f"{repo.rel}|{ep}" not in available:
                        missing += 1
                        continue
                elif not self.cache_file(ri, ep).is_file():
                    raise FileNotFoundError(f"missing demo cache for {repo.repo} episode {ep}; run BPV4CDX.build_cache")
                self.items.append((ri, ep))
        if missing:
            logging.info("bpv4cdx_ae: %d trajectories of split=%s skipped (not in the cache index)", missing, cfg.split)
        if cfg.max_trajectories:
            self.items = self.items[: int(cfg.max_trajectories)]
        self.num_episodes = len(self.items)
        self.num_frames = sum(len(self.repos[ri].states[ep]) for ri, ep in self.items)
        logging.info("bpv4cdx_ae dataset split=%s: %d trajectories, %d frames, %d repos",
                     cfg.split, self.num_episodes, self.num_frames, len(self.repos))

    def derive(self, **cfg_overrides) -> BPV4CDXAEDataset:
        """Same repos (no reload), other dataset settings, e.g. derive(split="val", segment_mode="fixed")."""
        new = copy.copy(self)
        new.cfg = dataclasses.replace(self.cfg, **cfg_overrides)
        new._build_items()
        return new

    def make_heldout_monitor(self):
        """Used by lerobot_train when present: HOM, the held-out generation error at every checkpoint save."""
        if not self.cfg.heldout_monitor or self.cfg.split != "train":
            return None
        from .heldout_monitor import HeldoutMonitor

        return HeldoutMonitor(self)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self):
        out = {}
        for repo in self.repos:  # one entry per robot type (the external stats are per robot type)
            out.setdefault(repo.robot_type, repo.stats)
        return out

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
        rec = torch.load(self.cache_file(ri, ep), map_location="cpu", weights_only=False)
        if rec["feats"].shape[0] != cfg.num_slots:
            raise ValueError(f"cache has {rec['feats'].shape[0]} slots, expected {cfg.num_slots}")
        stored = int(rec.get("spatial", rec["feats"].shape[2]))
        if stored == int(cfg.spatial_tokens):
            feats = rec["feats"]  # pooled at cache time (AE-3 stores P5 only)
        elif stored == 64:
            feats = pool_spatial(rec["feats"], int(cfg.spatial_tokens))
        else:
            raise ValueError(f"cache stores {stored} spatial tokens, dataset.spatial_tokens={cfg.spatial_tokens}")
        valid = rec.get("camera_valid")
        return {
            "demo.feats": feats,
            "demo.time": rec["time_ratio"].float(),
            "demo.camera_valid": valid.bool() if valid is not None else torch.ones(feats.shape[:2], dtype=torch.bool),
        }

    def segment_starts(self, length: int, mode: str | None = None) -> list[int]:
        mode = mode or self.cfg.segment_mode
        s, seg = int(self.cfg.segments_per_trajectory), int(self.cfg.segment_len)
        if mode == "fixed":
            starts = list(range(0, length, seg))
            return [starts[j % len(starts)] for j in range(s)]
        # random grid offset, then one random start from each of S equal time strata of the grid
        offset = int(torch.randint(0, min(seg, length), ()).item())
        grid = list(range(offset, length, seg))
        if len(grid) < s:
            return sorted(grid[int(j)] for j in torch.randint(0, len(grid), (s,)).tolist())
        edges = np.linspace(0, len(grid), s + 1).astype(int)
        return [grid[int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item())] for j in range(s)]

    def target(self, i: int, starts: list[int], generic: list[bool] | None = None,
               dropped: list[bool] | None = None) -> dict[str, torch.Tensor]:
        """Target-side tensors of trajectory i for the given segment starts (labels + text + segment query).
        Text per segment: dropped (attention mask all zero) with text_drop_prob, else generic with generic_text_prob,
        else canonical. Explicit `generic` / `dropped` lists (evaluation) bypass the draws."""
        cfg = self.cfg
        ri, ep = self.items[i]
        repo = self.repos[ri]
        length = len(repo.states[ep])
        if dropped is None:
            dropped = [False] * len(starts) if generic is not None else \
                [bool(torch.rand(()) < cfg.text_drop_prob) for _ in starts]
        if generic is None:
            generic = [bool(not d and torch.rand(()) < cfg.generic_text_prob) for d in dropped]
        out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK, "text.input_ids", "text.attention_mask")}
        tau, steps = [], []
        for f, g, d in zip(starts, generic, dropped):
            q = repo.query(ep, f, self.chunk_size)
            for key in (OBS_STATE, ACTION, ACTION_VALID_MASK):
                out[key].append(q[key])
            ids, attn = self.tokens(GENERIC_TEXT if g else repo.text)
            out["text.input_ids"].append(ids)
            out["text.attention_mask"].append(torch.zeros_like(attn) if d else attn)
            valid = min(self.chunk_size, length - f)
            tau.append([f / max(1, length - 1), (f + valid - 1) / max(1, length - 1)])
            steps.append(valid / self.chunk_size)
        batch = {key: torch.stack(v) for key, v in out.items()}
        batch.update({
            "seg.tau": torch.tensor(tau, dtype=torch.float32),
            "seg.steps": torch.tensor(steps, dtype=torch.float32),
            "seg.start": torch.tensor(starts, dtype=torch.long),
            "text.is_generic": torch.tensor(generic, dtype=torch.bool),
            "text.is_dropped": torch.tensor(dropped, dtype=torch.bool),
        })
        return batch

    def meta(self, i: int) -> dict[str, torch.Tensor]:
        ri, ep = self.items[i]
        return {
            "demo.index": torch.tensor(i, dtype=torch.long),
            "demo.task_index": torch.tensor(self.task_index[self.repos[ri].task], dtype=torch.long),
            "demo.length": torch.tensor(len(self.repos[ri].states[ep]), dtype=torch.long),
        }

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        ri, ep = self.items[i]
        starts = self.segment_starts(len(self.repos[ri].states[ep]))
        return {**self.demo(i), **self.target(i, starts), **self.meta(i)}

    def make_train_sampler(self, seed: int) -> torch.utils.data.Sampler | None:
        """Used by lerobot_train when present: length-proportional, task-stratified trajectory draws."""
        if not self.cfg.length_weighted_sampling:
            return None
        lengths = [len(self.repos[ri].states[ep]) for ri, ep in self.items]
        tasks = [self.repos[ri].task for ri, _ in self.items]
        return LengthTaskSampler(lengths, tasks, seed)


class LengthTaskSampler(torch.utils.data.Sampler[int]):
    """One epoch = len(dataset) draws. Each task gets a share of the draws proportional to its total frames
    (largest remainder); inside a task, trajectories are drawn with replacement proportional to their length; the
    epoch is then shuffled. The order depends only on (seed, epoch), so every rank builds the same sequence and the
    accelerate batch sharding splits it consistently; the epoch counter advances on every __iter__."""

    def __init__(self, lengths: list[int], tasks: list[str], seed: int) -> None:
        self.lengths = np.asarray(lengths, dtype=np.float64)
        self.tasks = np.asarray(tasks)
        self.seed, self.epoch = int(seed), 0

    def __len__(self) -> int:
        return len(self.lengths)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        n = len(self.lengths)
        uniq = np.unique(self.tasks)
        frames = np.array([self.lengths[self.tasks == t].sum() for t in uniq])
        share = frames / frames.sum() * n
        count = np.floor(share).astype(int)
        count[np.argsort(-(share - count))[: n - count.sum()]] += 1
        picks = []
        for t, c in zip(uniq, count):
            idx = np.flatnonzero(self.tasks == t)
            picks.append(rng.choice(idx, size=int(c), replace=True, p=self.lengths[idx] / self.lengths[idx].sum()))
        order = np.concatenate(picks)
        rng.shuffle(order)
        return iter(order.tolist())


def dataset_repos(cfg) -> tuple[list[str], dict[str, list[int]] | None]:
    """(repo paths, held-out map or None) from the split manifest (AE-3) or from repo_id_file (AE-1 / AE-2)."""
    if cfg.split_manifest:
        if not cfg.data_roots:
            raise ValueError("dataset.split_manifest needs dataset.data_roots")
        manifest = load_manifest(cfg.split_manifest)
        pairs = manifest_repos(manifest, cfg.data_roots, cfg.manifest_sources or None)
        if not pairs:
            raise FileNotFoundError(f"no repo of {cfg.split_manifest} found under {cfg.data_roots}")
        logging.info("bpv4cdx_ae: %d / %d manifest repos found locally", len(pairs), len(manifest["repos"]))
        return [p for p, _ in pairs], {p: list(e["heldout"]) for p, e in pairs}
    if not cfg.repo_id_file:
        raise ValueError("bpv4cdx_ae needs dataset.repo_id_file or dataset.split_manifest")
    return read_repo_list(cfg.repo_id_file, cfg.data_root), None


def make_bpv4cdx_ae_dataset(train_cfg) -> tuple[BPV4CDXAEDataset, dict]:
    cfg = train_cfg.dataset
    repo_ids, heldout = dataset_repos(cfg)
    ds = BPV4CDXAEDataset(cfg, train_cfg, repo_ids, chunk_size=int(train_cfg.policy.chunk_size), heldout=heldout)
    return ds, ds.stats
