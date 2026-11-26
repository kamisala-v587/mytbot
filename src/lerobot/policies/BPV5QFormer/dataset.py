"""BPv5 Stage A dataset: wall P5 demo cache + 80-D delta targets + task / generic text.

One item = one trajectory: demo feats from cache, S segment labels (delta action chunks), text tokens.
"""

from __future__ import annotations

import copy
import dataclasses
import gzip
import hashlib
import json
import logging
import multiprocessing as mp
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.dataset import LengthTaskSampler
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from . import data80d as D
from .configuration_bpv5qformer import BPV5QFormerDatasetConfig
from .modeling_bpv5qformer import ACTION_DIM_MASK

GENERIC_TEXT = D.GENERIC_TEXT


def _init_timer(cfg: BPV5QFormerDatasetConfig) -> float | None:
    return time.perf_counter() if getattr(cfg, "init_timing", False) else None


def _log_init_timing(cfg: BPV5QFormerDatasetConfig, component: str, started: float | None) -> None:
    if started is not None:
        logging.info(
            "bpv5 init_timing split=%s rank=%s local_rank=%s component=%s seconds=%.6f",
            cfg.split,
            os.environ.get("RANK", "0"),
            os.environ.get("LOCAL_RANK", "0"),
            component,
            time.perf_counter() - started,
        )


def _scan_fingerprint(cfg: BPV5QFormerDatasetConfig) -> list:
    """Fields that change which (repo, episode, length) rows exist.

    Split, holdout, dist_loading and max_trajectories are applied after the index
    is loaded, so they do not belong here.
    """
    return [
        str(cfg.cache_dir),
        list(cfg.repo_ids or []),
        list(cfg.sources or []),
        cfg.data_root,
        cfg.stats_root,
        cfg.stats_file,
        cfg.action_layout,
        cfg.delta_mode,
    ]


def scan_index_path(cfg: BPV5QFormerDatasetConfig) -> Path:
    digest = hashlib.sha256(
        json.dumps(_scan_fingerprint(cfg), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return Path(cfg.cache_dir) / "_scan_index" / f"{digest}.json.gz"


def _membership_fingerprint(cfg: BPV5QFormerDatasetConfig) -> tuple:
    """Fields that require rebuilding the in-memory item list."""
    return (
        cfg.cache_dir,
        tuple(cfg.repo_ids or ()),
        tuple(cfg.sources or ()),
        cfg.data_root,
        cfg.stats_root,
        cfg.stats_file,
        cfg.action_layout,
        cfg.delta_mode,
        tuple(cfg.drop_action_slots or ()),
        tuple(cfg.keep_action_slots or ()),
        cfg.max_trajectories,
        cfg.max_trajectories_seed,
        cfg.weight_rules_path,
        cfg.dist_loading,
    )


class BPV5QFormerDataset(torch.utils.data.Dataset):
    def __init__(self, cfg: BPV5QFormerDatasetConfig, chunk_size: int) -> None:
        started = _init_timer(cfg)
        if int(chunk_size) != int(cfg.segment_len):
            raise ValueError("dataset.segment_len must equal policy.chunk_size")
        self.cfg = cfg
        self.chunk_size = int(chunk_size)
        D.configure(
            root=cfg.data_root,
            stats_root=cfg.stats_root,
            stats_file=cfg.stats_file,
            action_layout=cfg.action_layout,
            delta_mode=cfg.delta_mode,
        )
        logging.info(
            "bpv5 data bind: root=%s layout=%s delta_mode=%s stats_file=%s drop_slots=%s",
            D.ROOT,
            D.ACTION_LAYOUT,
            D.DELTA_MODE,
            D.STATS_FILE,
            list(cfg.drop_action_slots or ()),
        )
        if cfg.keep_action_slots:
            logging.info("bpv5 keep_action_slots=%s", list(cfg.keep_action_slots))
        self._frames = D.FrameReader()
        self._tokenizer = None
        self._text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._norm: dict[str, tuple] = {}
        self._task_keys: dict[tuple[str, int], str] = {}
        self._rows_by_repo: dict[str, list[tuple[str, int, int]]] = {}
        self._membership_key: tuple | None = None
        self._missing_cache = 0
        self._repo_weights: dict[str, float] | None = None
        self._build_items()
        _log_init_timing(cfg, "dataset_construct_total", started)

    def _rows_for_this_rank(
        self, by_repo: dict[str, list[tuple[str, int, int]]]
    ) -> dict[str, list[tuple[str, int, int]]]:
        rels = sorted(by_repo)
        frames, episodes = _repo_mass(by_repo)
        self._repo_weights = repo_sampling_weights(rels, frames, episodes, self.cfg)
        if not self.cfg.dist_loading:
            return by_repo
        if self._repo_weights:
            keep_list = assign_repos_by_weight(rels, self._repo_weights)
        else:
            keep_list = assign_repos_for_rank(rels, frames)
        keep = set(keep_list)
        return {rel: rows for rel, rows in by_repo.items() if rel in keep}

    def _apply_split(self, by_repo: dict[str, list[tuple[str, int, int]]]) -> None:
        cfg = self.cfg
        train_items: list[tuple[str, int, int]] = []
        val_items: list[tuple[str, int, int]] = []
        hold_n = int(cfg.val_holdout_last_per_repo)
        for rel in sorted(by_repo):
            rows = by_repo[rel]
            if hold_n > 0 and len(rows) > hold_n:
                val_items.extend(rows[-hold_n:])
                train_items.extend(rows[:-hold_n])
            elif hold_n > 0 and len(rows) > 1:
                val_items.extend(rows[-1:])
                train_items.extend(rows[:-1])
            else:
                train_items.extend(rows)

        if cfg.split == "train":
            items = train_items
        elif cfg.split == "val":
            items = val_items if hold_n > 0 else train_items
        else:
            items = train_items + val_items

        if cfg.max_trajectories and len(items) > int(cfg.max_trajectories):
            rng = np.random.default_rng(int(cfg.max_trajectories_seed))
            idx = rng.choice(len(items), size=int(cfg.max_trajectories), replace=False)
            items = [items[int(i)] for i in sorted(idx.tolist())]
        self.items = items
        self.num_episodes = len(items)
        self.num_frames = sum(length for _, _, length in items)
        logging.info(
            "bpv5_qformer dataset split=%s: %d trajectories, %d frames, %d repos "
            "(skipped %d without cache; holdout_last_per_repo=%d -> train %d / val %d before max_trajectories)",
            cfg.split,
            self.num_episodes,
            self.num_frames,
            len({r for r, _, _ in items}),
            self._missing_cache,
            hold_n,
            len(train_items),
            len(val_items),
        )
        if not items:
            raise FileNotFoundError(
                f"no trajectories with cache under {cfg.cache_dir} for "
                f"repos={cfg.repo_ids or cfg.sources or 'all'} split={cfg.split}"
            )

    def _build_items(self) -> None:
        key = _membership_fingerprint(self.cfg)
        if self._rows_by_repo and self._membership_key == key:
            logging.info("bpv5 dataset reuse in-memory repo index (%d repos)", len(self._rows_by_repo))
            started = _init_timer(self.cfg)
            self._apply_split(self._rows_by_repo)
            _log_init_timing(self.cfg, "split_pair_build", started)
            return
        by_repo, missing = load_or_build_scan_index(self.cfg)
        started = _init_timer(self.cfg)
        by_repo = self._rows_for_this_rank(by_repo)
        _log_init_timing(self.cfg, "rank_partition", started)
        self._rows_by_repo = by_repo
        self._missing_cache = missing
        self._membership_key = key
        started = _init_timer(self.cfg)
        self._apply_split(self._rows_by_repo)
        _log_init_timing(self.cfg, "split_pair_build", started)

    def derive(self, **cfg_overrides) -> BPV5QFormerDataset:
        """Same cache root, other dataset settings (e.g. derive(split='val', segment_mode='fixed')).

        Reuses the parent repo index when membership fields are unchanged.
        """
        new = copy.copy(self)
        new.cfg = dataclasses.replace(self.cfg, **cfg_overrides)
        started = _init_timer(new.cfg)
        new._text_cache = {}
        new._norm = dict(self._norm)
        new._task_keys = dict(self._task_keys)
        new._rows_by_repo = self._rows_by_repo
        new._membership_key = self._membership_key
        new._missing_cache = self._missing_cache
        new._repo_weights = self._repo_weights
        new._build_items()
        _log_init_timing(new.cfg, "dataset_derive_total", started)
        return new

    def make_heldout_monitor(self):
        """Used by lerobot_train: held-out generation error (matched/blank/alt/wrong) at every checkpoint save."""
        if not self.cfg.heldout_monitor or self.cfg.split != "train":
            return None
        from .heldout_monitor import HeldoutMonitor

        return HeldoutMonitor(self)

    def __len__(self) -> int:
        return len(self.items)

    @property
    def stats(self) -> dict:
        # 80-D group stats live on disk; training does not need a merged stats dict
        return {}

    def norm(self, rel: str):
        if rel not in self._norm:
            self._norm[rel] = D.norm_vectors(rel)
        return self._norm[rel]

    def task_key(self, rel: str, ep_idx: int) -> str:
        key = (rel, int(ep_idx))
        if key not in self._task_keys:
            self._task_keys[key] = D.task_text(rel, ep_idx)
        return self._task_keys[key]

    def tokens(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if text not in self._text_cache:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.qwen3_5_processor_path)
            out = self._tokenizer(
                text,
                max_length=self.cfg.language_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            self._text_cache[text] = (out["input_ids"][0], out["attention_mask"][0].bool())
        return self._text_cache[text]

    def segment_starts(self, length: int, mode: str | None = None) -> list[int]:
        mode = mode or self.cfg.segment_mode
        s, seg = int(self.cfg.segments_per_trajectory), int(self.cfg.segment_len)
        if mode == "fixed":
            starts = list(range(0, max(1, length), seg))
            if not starts:
                starts = [0]
            return [starts[j % len(starts)] for j in range(s)]
        offset = int(torch.randint(0, min(seg, max(1, length)), ()).item())
        grid = list(range(offset, length, seg)) or [0]
        if len(grid) < s:
            return sorted(grid[int(j)] for j in torch.randint(0, len(grid), (s,)).tolist())
        edges = np.linspace(0, len(grid), s + 1).astype(int)
        return [
            grid[int(torch.randint(edges[j], max(edges[j] + 1, edges[j + 1]), ()).item())] for j in range(s)
        ]

    def demo(self, i: int) -> dict[str, torch.Tensor]:
        rel, ep_idx, _ = self.items[i]
        rec = torch.load(D.cache_path(self.cfg.cache_dir, rel, ep_idx), map_location="cpu", weights_only=True)
        feats = rec["feats"]
        if feats.shape[0] != self.cfg.num_slots:
            raise ValueError(f"{rel} ep{ep_idx}: cache slots {feats.shape[0]} != {self.cfg.num_slots}")
        if int(feats.shape[2]) != int(self.cfg.spatial_tokens):
            raise ValueError(
                f"{rel} ep{ep_idx}: cache spatial {feats.shape[2]} != dataset.spatial_tokens={self.cfg.spatial_tokens}"
            )
        valid = rec.get("camera_valid")
        return {
            "demo.feats": feats,
            "demo.time": rec["time_ratio"].float(),
            "demo.camera_valid": valid.bool() if valid is not None else torch.ones(feats.shape[:2], dtype=torch.bool),
        }

    def target(
        self,
        i: int,
        starts: list[int],
        *,
        state: np.ndarray,
        action: np.ndarray,
        length: int,
        generic: list[bool] | None = None,
        dropped: list[bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        rel, ep_idx, _ = self.items[i]
        a_mean, a_std, s_mean, s_std, a_mask, _s_mask, d_mask = self.norm(rel)
        a_mask = a_mask.copy()
        for dim in cfg.drop_action_slots or ():
            a_mask[int(dim)] = False
        if cfg.keep_action_slots:
            keep = np.zeros_like(a_mask)
            for dim in cfg.keep_action_slots:
                keep[int(dim)] = True
            a_mask &= keep
        text = D.task_text(rel, ep_idx)
        if dropped is None:
            dropped = (
                [False] * len(starts)
                if generic is not None
                else [bool(torch.rand(()) < cfg.text_drop_prob) for _ in starts]
            )
        if generic is None:
            generic = [bool(not d and torch.rand(()) < cfg.generic_text_prob) for d in dropped]

        actions, valids, states, ids, attns = [], [], [], [], []
        tau, steps = [], []
        for f, g, d in zip(starts, generic, dropped):
            f = min(int(f), max(0, length - 1))
            # delta_mask (not state_mask): joints subtract state; grippers stay absolute under tbot_aloha
            chunk, step_ok, state_t = D.delta_chunk(state, action, d_mask, a_mean, a_std, f, self.chunk_size)
            actions.append(torch.from_numpy(chunk))
            valids.append(torch.from_numpy(step_ok))
            states.append(torch.from_numpy(D.normalize_state(state_t, s_mean, s_std)))
            tok_ids, tok_attn = self.tokens(GENERIC_TEXT if g else text)
            ids.append(tok_ids)
            attns.append(torch.zeros_like(tok_attn) if d else tok_attn)
            valid_n = int(step_ok.sum())
            tau.append([f / max(1, length - 1), (f + max(valid_n, 1) - 1) / max(1, length - 1)])
            steps.append(valid_n / self.chunk_size)

        return {
            ACTION: torch.stack(actions),
            ACTION_VALID_MASK: torch.stack(valids),
            ACTION_DIM_MASK: torch.from_numpy(a_mask.copy()),
            OBS_STATE: torch.stack(states),
            "text.input_ids": torch.stack(ids),
            "text.attention_mask": torch.stack(attns),
            "seg.tau": torch.tensor(tau, dtype=torch.float32),
            "seg.steps": torch.tensor(steps, dtype=torch.float32),
            "seg.start": torch.tensor(starts, dtype=torch.long),
            "text.is_generic": torch.tensor(generic, dtype=torch.bool),
            "text.is_dropped": torch.tensor(dropped, dtype=torch.bool),
        }

    def meta(self, i: int, length: int) -> dict:
        rel, ep_idx, _ = self.items[i]
        return {
            "demo.index": torch.tensor(i, dtype=torch.long),
            "demo.length": torch.tensor(length, dtype=torch.long),
            "demo.episode": torch.tensor(ep_idx, dtype=torch.long),
            "demo.repo": rel,
            "demo.robot": str(D.info(rel).get("robot_type") or rel.split("/")[0]),
        }

    def __getitem__(self, i: int) -> dict:
        t0 = time.perf_counter()
        rel, ep_idx, _ = self.items[i]
        t_frames = time.perf_counter()
        ep = D.episodes(rel)[ep_idx]
        state, action = self._frames.episode(rel, ep)
        frames_s = time.perf_counter() - t_frames
        length = int(state.shape[0])
        starts = self.segment_starts(length)
        t_cache = time.perf_counter()
        demo = self.demo(i)
        cache_s = time.perf_counter() - t_cache
        t_xform = time.perf_counter()
        target = self.target(i, starts, state=state, action=action, length=length)
        xform_s = time.perf_counter() - t_xform
        return {
            **demo,
            **target,
            **self.meta(i, length),
            # Worker-side wall time. Main-process next() does not see this when prefetch is full.
            "_timing": {
                "getitem_s": torch.tensor(time.perf_counter() - t0, dtype=torch.float32),
                "cache_s": torch.tensor(cache_s, dtype=torch.float32),
                "frames_s": torch.tensor(frames_s, dtype=torch.float32),
                "xform_s": torch.tensor(xform_s, dtype=torch.float32),
            },
        }

    def make_train_sampler(self, seed: int) -> torch.utils.data.Sampler | None:
        if self.cfg.weight_rules_path and not self.cfg.homogeneous_batch:
            logging.warning(
                "bpv5 weight_rules_path=%s is ignored because homogeneous_batch=false",
                self.cfg.weight_rules_path,
            )
        if self.cfg.homogeneous_batch or not self.cfg.length_weighted_sampling:
            return None
        lengths = [length for _, _, length in self.items]
        # stratify by source (top-level) so one robot type cannot starve others in a small smoke set
        tasks = [rel.split("/")[0] for rel, _, _ in self.items]
        return LengthTaskSampler(lengths, tasks, seed)

    def make_train_batch_sampler(self, batch_size: int, seed: int):
        if not self.cfg.homogeneous_batch:
            return None
        return BPV5HomogeneousBatchSampler(
            self.items,
            int(batch_size),
            int(seed),
            repo_weights=self._repo_weights,
        )


_SCAN_CACHE_DIR = ""


def _scan_worker_init(payload: dict) -> None:
    global _SCAN_CACHE_DIR
    _SCAN_CACHE_DIR = payload["cache_dir"]
    D.configure(**payload["configure"])


def _scan_one_repo(rel: str) -> tuple[str, list[list[int]], int, str | None]:
    """One leaf: cached episodes that also have meta length and loadable stats."""
    cached_eps: list[int] = []
    cdir = Path(_SCAN_CACHE_DIR) / rel
    if cdir.is_dir():
        for name in os.listdir(cdir):
            if name.startswith("ep") and name.endswith(".pt"):
                try:
                    cached_eps.append(int(name[2:-3]))
                except ValueError:
                    continue
    if not cached_eps:
        return rel, [], 0, None
    if D.STATS_FILE is None and rel not in D.stats_index():
        return rel, [], 0, "no norm stats"
    try:
        D.norm_vectors(rel)
    except Exception as e:  # noqa: BLE001
        return rel, [], 0, f"stats error: {e}"
    try:
        eps = D.episodes(rel)
    except Exception as e:  # noqa: BLE001
        return rel, [], 0, str(e)
    rows: list[list[int]] = []
    missing = 0
    for ep_idx in sorted(cached_eps):
        ep = eps.get(ep_idx)
        if ep is None:
            missing += 1
            continue
        rows.append([int(ep_idx), int(ep["length"])])
    return rel, rows, missing, None


def _configure_payload(cfg: BPV5QFormerDatasetConfig) -> dict:
    return {
        "cache_dir": str(cfg.cache_dir),
        "configure": {
            "root": cfg.data_root,
            "stats_root": cfg.stats_root,
            "stats_file": cfg.stats_file,
            "action_layout": cfg.action_layout,
            "delta_mode": cfg.delta_mode,
        },
    }


def _candidate_repos(cfg: BPV5QFormerDatasetConfig) -> list[str]:
    D.configure(**_configure_payload(cfg)["configure"])
    return list(cfg.repo_ids) if cfg.repo_ids else D.repos(cfg.sources or None)


def try_load_scan_index(
    cfg: BPV5QFormerDatasetConfig,
) -> tuple[dict[str, list[tuple[str, int, int]]], int] | None:
    path = scan_index_path(cfg)
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as f:
        payload = json.load(f)
    if payload.get("version") != 1 or payload.get("fingerprint") != _scan_fingerprint(cfg):
        logging.warning("bpv5 scan index ignored (fingerprint mismatch): %s", path)
        return None
    by_repo = {
        rel: [(rel, int(ep), int(length)) for ep, length in rows]
        for rel, rows in payload["repos"].items()
        if rows
    }
    return by_repo, int(payload.get("missing_cache") or 0)


def build_scan_index(
    cfg: BPV5QFormerDatasetConfig, workers: int
) -> tuple[dict[str, list[tuple[str, int, int]]], int]:
    """Scan every candidate leaf once. Callers persist the result."""
    rels = _candidate_repos(cfg)
    workers = max(1, int(workers))
    logging.info("bpv5 dataset scan: %d repos, workers=%d", len(rels), workers)
    t0 = time.time()
    by_repo: dict[str, list[tuple[str, int, int]]] = {}
    missing = 0
    skipped = 0
    done = 0

    def _take(result: tuple[str, list[list[int]], int, str | None]) -> None:
        nonlocal missing, skipped, done
        rel, rows, miss, reason = result
        done += 1
        missing += miss
        if reason:
            skipped += 1
            logging.warning("bpv5 skip repo %s: %s", rel, reason)
        elif rows:
            by_repo[rel] = [(rel, int(ep), int(length)) for ep, length in rows]
        if done == 1 or done % 20 == 0 or done == len(rels):
            logging.info(
                "bpv5 dataset scan progress %d/%d repos (%.1f min, kept=%d miss_cache=%d)",
                done,
                len(rels),
                (time.time() - t0) / 60.0,
                sum(len(v) for v in by_repo.values()),
                missing,
            )

    if workers == 1:
        _scan_worker_init(_configure_payload(cfg))
        for rel in rels:
            _take(_scan_one_repo(rel))
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context("spawn"),
            initializer=_scan_worker_init,
            initargs=(_configure_payload(cfg),),
        ) as pool:
            futures = [pool.submit(_scan_one_repo, rel) for rel in rels]
            for fut in as_completed(futures):
                _take(fut.result())
    logging.info(
        "bpv5 dataset scan done: %d repos, %d trajectories, miss_cache=%d skipped=%d (%.1f min)",
        len(by_repo),
        sum(len(v) for v in by_repo.values()),
        missing,
        skipped,
        (time.time() - t0) / 60.0,
    )
    return by_repo, missing


def save_scan_index(
    cfg: BPV5QFormerDatasetConfig,
    by_repo: dict[str, list[tuple[str, int, int]]],
    missing_cache: int,
) -> Path:
    path = scan_index_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "fingerprint": _scan_fingerprint(cfg),
        "missing_cache": int(missing_cache),
        "repos": {
            rel: [[ep, length] for _rel, ep, length in rows]
            for rel, rows in sorted(by_repo.items())
        },
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)
    logging.info("bpv5 scan index written: %s (%d repos)", path, len(by_repo))
    return path


def load_or_build_scan_index(
    cfg: BPV5QFormerDatasetConfig,
) -> tuple[dict[str, list[tuple[str, int, int]]], int]:
    """Full-repo index. dist_loading subsets it afterwards; it is not baked in."""
    path = scan_index_path(cfg)
    started = _init_timer(cfg)
    hit = try_load_scan_index(cfg)
    _log_init_timing(cfg, "scan_index_load", started)
    if hit is not None:
        logging.info(
            "bpv5 scan index hit: %s (%d repos, miss_cache=%d)",
            path,
            len(hit[0]),
            hit[1],
        )
        return hit
    rank, world = _rank_world()
    if rank == 0:
        logging.info("bpv5 scan index miss, building %s", path)
        started = _init_timer(cfg)
        by_repo, missing = build_scan_index(cfg, int(cfg.load_threads))
        _log_init_timing(cfg, "scan_index_build", started)
        started = _init_timer(cfg)
        save_scan_index(cfg, by_repo, missing)
        _log_init_timing(cfg, "scan_index_save", started)
    if world > 1:
        import torch.distributed as dist

        started = _init_timer(cfg)
        dist.barrier()
        _log_init_timing(cfg, "scan_index_barrier_wait", started)
    started = _init_timer(cfg)
    hit = try_load_scan_index(cfg)
    _log_init_timing(cfg, "scan_index_reload", started)
    if hit is None:
        raise FileNotFoundError(f"bpv5 scan index was not written: {path}")
    return hit


def _rank_world() -> tuple[int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _repo_mass(
    by_repo: dict[str, list[tuple[str, int, int]]],
) -> tuple[dict[str, int], dict[str, int]]:
    frames: dict[str, int] = {}
    episodes: dict[str, int] = {}
    for rel, rows in by_repo.items():
        episodes[rel] = len(rows)
        frames[rel] = sum(int(length) for _, _, length in rows) or max(len(rows), 1)
    return frames, episodes


def repo_sampling_weights(
    rels: list[str],
    frames: dict[str, int],
    episodes: dict[str, int],
    cfg: BPV5QFormerDatasetConfig,
) -> dict[str, float] | None:
    """Global leaf weights from weight_rules YAML. None = sample proportional to frames."""
    path = cfg.weight_rules_path
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"dataset.weight_rules_path not found: {path}")
    from omegaconf import OmegaConf

    from lerobot.datasets.factory import compute_repo_weights, group_repo_ids_by_rules

    groups_cfg = OmegaConf.load(path)
    weights = compute_repo_weights(list(rels), frames, episodes, groups_cfg)
    repo_to_group, _, ordered = group_repo_ids_by_rules(list(rels), groups_cfg)
    share: dict[str, float] = defaultdict(float)
    for rel, w in weights.items():
        share[repo_to_group.get(rel, "__default__")] += float(w)
    summary = ", ".join(f"{name}={share.get(name, 0.0):.3f}" for name in ordered if share.get(name, 0.0) > 0)
    logging.info("bpv5 weight_rules %s global group share: %s", path, summary or "(empty)")
    return weights


def assign_repos_for_rank(rels: list[str], frames_map: dict[str, int] | None = None) -> list[str]:
    """Keep only this rank's leaves. Same balance rule as tbot dist_loading."""
    from lerobot.datasets.factory import compute_balanced_repo_assignment

    rank, world = _rank_world()
    if world <= 1:
        return list(rels)
    if frames_map is None:
        frames_map = {}
        for rel in rels:
            info = D.info(rel)
            frames = int(info.get("total_frames") or 0)
            if frames <= 0:
                frames = int(info.get("total_episodes") or 1)
            frames_map[rel] = frames
    buckets = compute_balanced_repo_assignment(list(rels), frames_map, world)
    mine = buckets[rank]
    logging.info(
        "bpv5 dist_loading rank %d/%d: %d/%d repos, frames=%d",
        rank,
        world,
        len(mine),
        len(rels),
        sum(frames_map.get(rel, 0) for rel in mine),
    )
    return mine


def assign_repos_by_weight(rels: list[str], weights: dict[str, float]) -> list[str]:
    """Partition leaves so each rank's sampling-weight mass is close to 1/world.

    Local renormalization then tracks the global weight_rules mix. A single leaf
    cannot exceed 1/world of global steps, because it stays on one rank.
    """
    rank, world = _rank_world()
    if world <= 1:
        return list(rels)
    order = sorted(rels, key=lambda rel: (-float(weights.get(rel, 0.0)), rel))
    buckets: list[list[str]] = [[] for _ in range(world)]
    loads = [0.0] * world
    for rel in order:
        target = min(range(world), key=lambda i: (loads[i], i))
        buckets[target].append(rel)
        loads[target] += float(weights.get(rel, 0.0))
    cap = 1.0 / world
    for rel in order:
        mass = float(weights.get(rel, 0.0))
        if mass > cap + 1e-6:
            logging.warning(
                "bpv5 weight_rules: leaf %s weight %.4f > 1/%d=%.4f; "
                "its global share is capped at %.4f (one leaf per rank)",
                rel,
                mass,
                world,
                cap,
                cap,
            )
    mine = buckets[rank]
    logging.info(
        "bpv5 dist_loading weight-balanced rank %d/%d: %d/%d repos, weight_mass=%.4f",
        rank,
        world,
        len(mine),
        len(rels),
        loads[rank],
    )
    return mine


class BPV5HomogeneousBatchSampler(torch.utils.data.Sampler):
    """Each batch is batch_size trajectories from one leaf repo.

    Without weight_rules, a repo is chosen proportional to its frame count.
    With weight_rules, repo probabilities are the YAML weights (renormalized on this rank).
    Trajectories inside the chosen repo are drawn with replacement.
    """

    def __init__(
        self,
        items: list[tuple[str, int, int]],
        batch_size: int,
        seed: int,
        repo_weights: dict[str, float] | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        groups: dict[str, list[int]] = defaultdict(list)
        frames: dict[str, int] = defaultdict(int)
        for i, (rel, _ep, length) in enumerate(items):
            groups[rel].append(i)
            frames[rel] += int(length)
        self.rels = sorted(groups)
        if not self.rels:
            raise ValueError("homogeneous batch sampler requires at least one trajectory")
        self.groups = groups
        if repo_weights:
            raw = np.array([float(repo_weights.get(rel, 0.0)) for rel in self.rels], dtype=np.float64)
            if float(raw.sum()) <= 0:
                raise ValueError("weight_rules assigned zero mass to every repo on this rank")
            self.weights = raw / raw.sum()
            mode = "weight_rules"
        else:
            raw = np.array([frames[rel] for rel in self.rels], dtype=np.float64)
            self.weights = raw / raw.sum()
            mode = "frames"
        self.batch_size = int(batch_size)
        self.num_batches = max(1, len(items) // self.batch_size)
        self.seed = int(seed)
        self.epoch = 0
        logging.info(
            "bpv5 homogeneous_batch: mode=%s repos=%d batch_size=%d batches/epoch=%d",
            mode,
            len(self.rels),
            self.batch_size,
            self.num_batches,
        )

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        self.epoch += 1
        for _ in range(self.num_batches):
            rel = self.rels[int(rng.choice(len(self.rels), p=self.weights))]
            idx = self.groups[rel]
            pick = rng.choice(idx, size=self.batch_size, replace=True)
            yield [int(i) for i in pick]


def _stack_timing(items: list[dict]) -> dict[str, torch.Tensor] | None:
    timings = [it.get("_timing") for it in items]
    if not timings or any(not isinstance(t, dict) for t in timings):
        return None
    keys = timings[0].keys()
    return {k: torch.stack([t[k].float().reshape(()) for t in timings]) for k in keys}


def collate_bpv5(items: list[dict]) -> dict:
    """Stack tensors; keep string fields as lists; broadcast per-sample ACTION_DIM_MASK."""
    timing = _stack_timing(items)
    keys = [k for k in items[0].keys() if k != "_timing"]
    out = {}
    for k in keys:
        vals = [it[k] for it in items]
        if torch.is_tensor(vals[0]):
            out[k] = torch.stack(vals)
        else:
            out[k] = vals
    if timing is not None:
        out["_timing"] = timing
    return out


def make_bpv5_qformer_dataset(train_cfg) -> tuple[BPV5QFormerDataset, dict]:
    cfg = train_cfg.dataset
    for key in ("num_slots", "spatial_tokens", "max_action_dim", "max_state_dim"):
        if int(getattr(cfg, key)) != int(getattr(train_cfg.policy, key)):
            raise ValueError(f"dataset.{key} != policy.{key}")
    ds = BPV5QFormerDataset(cfg, chunk_size=int(train_cfg.policy.chunk_size))
    return ds, ds.stats
