"""Shared helpers for the gate scripts: configs, datasets, policies, batches, statistics, result files."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .configuration_bpv4cdx import BPV4CDXAEDatasetConfig
from .dataset import BPV4CDXAEDataset, dataset_repos


def apply_local_paths(cfg):
    """Rewrite machine-specific paths of a (checkpoint) config from BPV4CDX_* environment variables, if set:
    BPV4CDX_DATA_ROOT, BPV4CDX_QWEN_PATH, BPV4CDX_STATS_ROOT, BPV4CDX_REPO (repo lists under configs/ds_ids/pro6k),
    BPV4CDX_CACHE_ROOT + BPV4CDX_CACHE_PREFIX (cache dir = <root>/<prefix><sampler>, prefix default "bpv4cdx_")."""
    env = os.environ
    d, p = cfg.dataset, cfg.policy
    if env.get("BPV4CDX_DATA_ROOT"):
        d.data_root = env["BPV4CDX_DATA_ROOT"]
    if env.get("BPV4CDX_QWEN_PATH"):
        d.qwen3_5_processor_path = env["BPV4CDX_QWEN_PATH"]
        p.qwen3_5_pretrained_path = env["BPV4CDX_QWEN_PATH"]
    if env.get("BPV4CDX_STATS_ROOT"):
        d.external_stats_root = env["BPV4CDX_STATS_ROOT"]
    if env.get("BPV4CDX_REPO") and d.repo_id_file:
        d.repo_id_file = str(Path(env["BPV4CDX_REPO"]) / "configs/ds_ids/pro6k" / Path(d.repo_id_file).name)
    manifest = bool(getattr(d, "split_manifest", ""))
    if env.get("BPV4CDX_CACHE_ROOT") and not manifest:
        d.cache_dir = str(Path(env["BPV4CDX_CACHE_ROOT"]) / f"{env.get('BPV4CDX_CACHE_PREFIX', 'bpv4cdx_')}{d.sampler}")
    # AE-3 (split manifest): cache dir <BPV4CDX_CACHE_ROOT>/bpv4cdx_ae3_<sampler> with its _index.txt; BPV4CDX_DATA_ROOTS
    # (":"-separated), BPV4CDX_MANIFEST, BPV4CDX_ARRAY_STORE (or "none"), BPV4CDX_CACHE_INDEX (or "none")
    if manifest:
        if env.get("BPV4CDX_CACHE_ROOT"):
            d.cache_dir = str(Path(env["BPV4CDX_CACHE_ROOT"]) / f"bpv4cdx_ae3_{d.sampler}")
            if d.cache_index:
                d.cache_index = str(Path(d.cache_dir) / "_index.txt")
        if env.get("BPV4CDX_DATA_ROOTS"):
            d.data_roots = [r for r in env["BPV4CDX_DATA_ROOTS"].split(":") if r]
        if env.get("BPV4CDX_MANIFEST"):
            d.split_manifest = env["BPV4CDX_MANIFEST"]
        for var, field_name in (("BPV4CDX_ARRAY_STORE", "array_store"), ("BPV4CDX_CACHE_INDEX", "cache_index")):
            if env.get(var):
                setattr(d, field_name, "" if env[var] == "none" else env[var])
    return cfg


def load_cfg(config_or_ckpt: str | Path, overrides: list[str] | None = None):
    """TrainPipelineConfig from a .jsonc config or a checkpoint `pretrained_model` dir; local paths from BPV4CDX_*
    environment variables (see apply_local_paths), then draccus CLI overrides win."""
    from lerobot.configs.train import TrainPipelineConfig

    path = Path(config_or_ckpt)
    weights = path / "model.safetensors" if path.is_dir() else None
    if path.is_dir():
        path = path / "train_config.json"
    cfg = TrainPipelineConfig.from_pretrained(str(path))
    apply_local_paths(cfg)
    if weights is not None and weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), "pt") as f:
            if "model.bos" not in set(f.keys()):
                cfg.policy.prefix_bos = False  # trained before the BOS vector existed
    for arg in overrides or []:
        if not arg.startswith("--") or "=" not in arg:
            raise ValueError(f"override must look like --section.field=value, got {arg!r}")
        key, value = arg[2:].split("=", 1)
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        obj, *rest = key.split(".")
        target = cfg if not rest else getattr(cfg, obj)
        for name in rest[:-1]:
            target = getattr(target, name)
        field_name = rest[-1] if rest else obj
        if not hasattr(target, field_name):
            raise AttributeError(f"unknown config field {key}")
        setattr(target, field_name, value)
    return cfg


def make_dataset(cfg, **dataset_overrides) -> BPV4CDXAEDataset:
    dcfg: BPV4CDXAEDatasetConfig = dataclasses.replace(cfg.dataset, **dataset_overrides)
    for key in ("num_slots", "spatial_tokens"):
        if int(getattr(dcfg, key)) != int(getattr(cfg.policy, key)):
            raise ValueError(f"dataset.{key} != policy.{key}")
    holder = type("TrainCfg", (), {"dataset": dcfg})()
    if dataset_overrides.get("repo_id_file"):
        # an explicit repo list (e.g. bp_probe) replaces the manifest
        dcfg = dataclasses.replace(dcfg, split_manifest="")
        holder = type("TrainCfg", (), {"dataset": dcfg})()
    repos, heldout = dataset_repos(dcfg)
    return BPV4CDXAEDataset(dcfg, holder, repos, chunk_size=int(cfg.policy.chunk_size), heldout=heldout)


def make_policy(cfg, ckpt: str | Path | None = None, device: str = "cuda"):
    from .modeling_bpv4cdx import BPV4CDXAEPolicy

    if ckpt:
        # the checkpoint's own config carries the training machine's paths; use the (path-rewritten) one instead
        policy = BPV4CDXAEPolicy.from_pretrained(str(ckpt), config=cfg.policy)
    else:
        policy = BPV4CDXAEPolicy(cfg.policy)
    return policy.to(device).eval()


def to_device(batch: dict, device: str = "cuda") -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def collate(items: list[dict]) -> dict:
    return default_collate(items)


def masked_error(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per segment mean squared error over valid steps and real action dims: [N, H, A] -> [N]."""
    a = target.shape[-1]
    w = valid[..., None].float()
    return ((pred[..., :a].float() - target.float()) ** 2 * w).sum((1, 2)) / (w.sum((1, 2)) * a).clamp_min(1)


def paired_bootstrap_ratio(num: np.ndarray, den: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Ratio of means num / den with a cluster (group = trajectory) bootstrap 95% CI: all segments of a resampled
    trajectory enter together; 2000 resamples, 2.5 / 97.5 percentiles."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    ratios = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        ratios.append(num[pick].mean() / max(1e-12, den[pick].mean()))
    lo, hi = np.percentile(ratios, [2.5, 97.5])
    return {"ratio": float(num.mean() / max(1e-12, den.mean())), "ci95": [float(lo), float(hi)], "n": int(len(num)),
            "groups": int(len(uniq))}


def bootstrap_mean(values: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean with a 95% bootstrap CI (resampling the given units, e.g. per-task statistics)."""
    values = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if not len(values):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95": [float(lo), float(hi)], "n": int(len(values))}


class ActionInverter:
    """Normalized delta action chunk of one repo -> absolute action (inverse of the dataset transforms: mean_std
    normalization, then + segment-start state on the delta dims). Checked against the raw actions on first use."""

    def __init__(self, repo) -> None:
        from lerobot.policies.q35.transform_q35 import Q35DeltaActionTransformFn
        from lerobot.transforms.core import NormalizeTransformFn
        from lerobot.utils.constants import ACTION

        norm = next(t for t in repo.transforms if isinstance(t, NormalizeTransformFn))
        delta = next(t for t in repo.transforms if isinstance(t, Q35DeltaActionTransformFn))
        if norm.mode != "mean_std":
            raise ValueError("ActionInverter supports mean_std normalization only")
        # one stats entry per composed action field (FEATURE_MAPPING order); aloha has the single field `action`
        keys = repo.emb.action_keys if hasattr(repo, "emb") else [ACTION]
        self.mean = np.concatenate([np.asarray(norm.norm_stats[k]["mean"], np.float64).reshape(-1) for k in keys])
        self.std = np.concatenate([np.asarray(norm.norm_stats[k]["std"], np.float64).reshape(-1) for k in keys])
        dim = len(self.mean)
        self.mask = np.asarray(delta.mask if delta.mask is not None else [True] * dim, dtype=bool)[:dim]
        self.skip = bool(getattr(delta, "skip", False))

    def __call__(self, normalized: np.ndarray, start_state: np.ndarray) -> np.ndarray:
        """[H, >=A] normalized delta, [S] raw state at the segment start -> [H, A] absolute."""
        a = len(self.mean)
        x = np.asarray(normalized, np.float64)[:, :a] * (self.std + 1e-6) + self.mean
        if not self.skip:
            x = x + np.where(self.mask, np.asarray(start_state, np.float64)[:a], 0.0)
        return x


def frozen_checksum(policy) -> dict:
    """Cheap checksum of the frozen parameters (per-tensor sum and sum of squares, hashed) + trainable names."""
    h, frozen, trainable = hashlib.sha256(), 0, []
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if p.requires_grad:
                trainable.append(name)
                continue
            x = p.detach().float()
            h.update(name.encode())
            h.update(np.array([float(x.sum()), float((x * x).sum())], dtype=np.float64).tobytes())
            frozen += p.numel()
    return {"frozen_params": int(frozen), "frozen_sha256": h.hexdigest(), "trainable_tensors": len(trainable),
            "trainable_sha256": hashlib.sha256("\n".join(trainable).encode()).hexdigest()}


def reproducibility(cfg, ckpt: str | Path | None = None, policy=None, ds=None) -> dict:
    """Result-file header (design doc 7.12): config, seed, data list and split, cache / stats hashes, sampler
    version, frozen-parameter checksum and trainable list."""
    d, p = cfg.dataset, cfg.policy
    repo_file = Path(d.split_manifest) if getattr(d, "split_manifest", "") else (Path(d.repo_id_file) if d.repo_id_file else None)
    robot = sorted(ds.stats)[0] if ds is not None and len(ds.repos) else "aloha"
    stats = Path(d.external_stats_root or "") / robot / d.action_mode / "stats.json"
    info = {
        "checkpoint": str(ckpt) if ckpt else None,
        "config": {"dataset": dataclasses.asdict(d), "policy_type": p.type, "seed": cfg.seed, "steps": cfg.steps,
                   "batch_size": cfg.batch_size},
        "repo_id_file": str(repo_file) if repo_file else None,
        "repo_id_file_sha256": hashlib.sha256(repo_file.read_bytes()).hexdigest() if repo_file and repo_file.is_file() else None,
        "split": {"split": d.split, "val_holdout_last": dict(d.val_holdout_last)},
        "stats_json": str(stats),
        "stats_sha256": hashlib.sha256(stats.read_bytes()).hexdigest() if stats.is_file() else None,
        "cache_dir": d.cache_dir,
    }
    if ds is not None and len(ds):
        h, versions = hashlib.sha256(), set()
        picks = sorted({0, len(ds) // 2, len(ds) - 1})
        for i in picks:
            ri, ep = ds.items[i]
            f = ds.cache_file(ri, ep)
            h.update(f.read_bytes())
            rec = torch.load(f, map_location="cpu", weights_only=False)
            versions.add(json.dumps(rec.get("sampler_version") or {"name": rec.get("sampler"), "version": "unrecorded"},
                                    sort_keys=True))
        info.update({"episodes": len(ds), "cache_sample_sha256": h.hexdigest(), "cache_sample_items": picks,
                     "sampler_versions": sorted(versions),
                     "sampler_matches_config": all(json.loads(v)["name"] == d.sampler for v in versions)})
    if policy is not None:
        info["parameters"] = frozen_checksum(policy)
    return info


def code_revision() -> str:
    root = Path(__file__).resolve().parents[4]
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        rev = "unknown"
    h = hashlib.sha1()
    for f in sorted(Path(__file__).parent.glob("*.py")):
        h.update(f.read_bytes())
    return f"{rev}+bpv4cdx-{h.hexdigest()[:10]}"


def write_result(out_dir: str | Path, name: str, result: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"code_revision": code_revision(), **result}
    path = out / f"{name}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False, default=float))
    return path
