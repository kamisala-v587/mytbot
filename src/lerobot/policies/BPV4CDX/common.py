"""Shared helpers for the gate scripts: configs, datasets, policies, batches, statistics, result files."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import default_collate

from .configuration_bpv4cdx import BPV4CDXAEDatasetConfig
from .dataset import BPV4CDXAEDataset
from .paths import read_repo_list


def load_cfg(config_or_ckpt: str | Path, overrides: list[str] | None = None):
    """TrainPipelineConfig from a .jsonc config or a checkpoint `pretrained_model` dir (+ draccus CLI overrides)."""
    from lerobot.configs.train import TrainPipelineConfig

    path = Path(config_or_ckpt)
    if path.is_dir():
        path = path / "train_config.json"
    return TrainPipelineConfig.from_pretrained(str(path), cli_args=list(overrides or []))


def make_dataset(cfg, **dataset_overrides) -> BPV4CDXAEDataset:
    dcfg: BPV4CDXAEDatasetConfig = dataclasses.replace(cfg.dataset, **dataset_overrides)
    holder = type("TrainCfg", (), {"dataset": dcfg})()
    repos = read_repo_list(dcfg.repo_id_file, dcfg.data_root)
    return BPV4CDXAEDataset(dcfg, holder, repos, chunk_size=int(cfg.policy.chunk_size))


def make_policy(cfg, ckpt: str | Path | None = None, device: str = "cuda"):
    from .modeling_bpv4cdx import BPV4CDXAEPolicy

    if ckpt:
        policy = BPV4CDXAEPolicy.from_pretrained(str(ckpt))
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
    """Ratio of means num / den with a cluster (group) bootstrap 95% CI."""
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
