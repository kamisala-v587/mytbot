"""Benchmark the exact training dataloader path used by lerobot_train.

Run with the same distributed shape as training, for example:
cd /vla/workspace/my_tbot
conda activate bptbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1

accelerate launch --num_processes=8 -m tools.bpva_benchmark.train_dataloader_benchmark \
      --config-path=/vla/workspace/my_tbot/configs/bpvav2_pretrain_test.jsonc

The script intentionally stops before policy/optimizer creation and only measures the
training dataloader construction and iteration path.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from accelerate import Accelerator
from accelerate.utils import send_to_device
from torch.utils.data import Dataset, IterableDataset

from lerobot.datasets.sampler import MultiLeRobotWeightedSampler
from lerobot.policies.names import is_tbot_sa1_wan
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import gather_object

from .config_utils import register_bpva_configs
from .train_benchmark import nonnegative_int, positive_int


@dataclass
class SampleLoadRecord:
    rank: int
    step: int
    sample_in_batch: int | None
    repo_id: str | None
    dataset_index: int | None
    global_index: int | None
    local_index: int | None
    elapsed_s: float | None
    pid: int


@dataclass
class BatchLoadRecord:
    rank: int
    step: int
    batch_size: int | None
    elapsed_s: float
    send_to_device_s: float | None
    sample_mean_s: float | None
    sample_max_s: float | None
    repos: dict[str, int]
    pid: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="完全复用 lerobot_train dataloader 构建路径的加载速度 benchmark"
    )
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-dir", default="outputs/bpva_benchmark/train_dataloader")
    parser.add_argument("--exact-output-dir", action="store_true")
    parser.add_argument("--warmup-batches", type=nonnegative_int, default=5)
    parser.add_argument("--measure-batches", type=positive_int, default=100)
    parser.add_argument("--num-workers", type=nonnegative_int)
    parser.add_argument(
        "--skip-send-to-device",
        action="store_true",
        help="dataset.dist_loading=true 时跳过训练循环中的 send_to_device 计时",
    )
    parser.add_argument(
        "--record-warmup",
        action="store_true",
        help="也把 warmup batch 写入明细，默认只记录 measure 阶段",
    )
    parser.add_argument(
        "--out-of-order",
        action="store_true",
        help="设置 DataLoader(in_order=False)，允许 worker 先完成的 batch 先返回，用于验证顺序交付是否造成长尾等待",
    )
    return parser


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _append_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            payload = asdict(row) if hasattr(row, "__dataclass_fields__") else row
            f.write(json.dumps(_json_safe(payload), ensure_ascii=False) + "\n")


def _load_cfg(path: str):
    register_bpva_configs()
    # Explicitly import BPVAv2 as well; older helper only needed BPVA.
    import lerobot.policies.BPVAv2.configuration_bpva  # noqa: F401
    from lerobot.configs.train import TrainPipelineConfig

    cfg = copy.deepcopy(TrainPipelineConfig.from_pretrained(path))
    if cfg.policy is None:
        raise ValueError("配置缺少 policy，无法模拟训练 dataloader 分支")
    return cfg


def _is_fastwam_policy_type(policy_type: str | None) -> bool:
    return policy_type == "fastwam" or is_tbot_sa1_wan(policy_type)


def _fastwam_policy_module(policy_type: str) -> str:
    return "TBot_SA1_Wan" if is_tbot_sa1_wan(policy_type) else "fastwam"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _config_type_name(cfg_obj: Any) -> str:
    configured_type = getattr(cfg_obj, "type", None)
    if configured_type is not None:
        return str(configured_type)
    return cfg_obj.__class__.__name__


def _make_output_dir(base: str, exact: bool, accelerator: Accelerator) -> Path:
    base_path = Path(base)
    if exact:
        output_dir = base_path
    else:
        stamp = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
        output_dir = base_path / stamp
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    return output_dir


def _tensor_scalar_to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.detach().flatten()[0].item())
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return float(value.detach().flatten()[0].item())
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _batch_size(batch: Any) -> int | None:
    if isinstance(batch, dict):
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
    return None


def _as_list(value: Any, size: int) -> list[Any]:
    if value is None:
        return [None] * size
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().flatten().tolist()
        return flat[:size] + [None] * max(0, size - len(flat))
    if isinstance(value, (list, tuple)):
        return list(value)[:size] + [None] * max(0, size - len(value))
    return [value] * size


def _repo_ids_for_dataset(dataset: Any) -> list[str]:
    repos = getattr(dataset, "datasets", None)
    if repos is not None:
        return [str(getattr(ds, "repo_id", f"dataset_{i}")) for i, ds in enumerate(repos)]
    repo_id = getattr(dataset, "repo_id", None)
    if repo_id is not None:
        return [str(repo_id)]
    return []


def _locate_index(dataset: Any, index: int) -> tuple[int | None, int | None, str | None]:
    locator = getattr(dataset, "_locate_dataset", None)
    repos = getattr(dataset, "datasets", None)
    if callable(locator) and repos is not None:
        ds_idx, local_idx = locator(index)
        repo_id = str(getattr(repos[ds_idx], "repo_id", f"dataset_{ds_idx}"))
        return int(ds_idx), int(local_idx), repo_id
    repo_id = getattr(dataset, "repo_id", None)
    return None, int(index), str(repo_id) if repo_id is not None else None


class TimedMapDataset(Dataset):
    """Measure per-sample __getitem__ latency while preserving DataLoader behavior."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.repo_ids = _repo_ids_for_dataset(dataset)
        self._repo_to_dataset_index = {repo_id: i for i, repo_id in enumerate(self.repo_ids)}
        for name in ("dataset_weights", "num_frames", "num_episodes", "meta", "datasets", "_lengths", "_cum_lengths"):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)

    def _locate_dataset(self, idx: int) -> tuple[int, int]:
        locator = getattr(self.dataset, "_locate_dataset", None)
        if callable(locator):
            return locator(idx)
        return 0, idx

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start = time.perf_counter()
        sample = self.dataset[idx]
        elapsed = time.perf_counter() - start
        ds_idx, local_idx, repo_id = _locate_index(self.dataset, int(idx))
        if not isinstance(sample, dict):
            sample = {"sample": sample}
        sample = dict(sample)
        if ds_idx is None and repo_id in self._repo_to_dataset_index:
            ds_idx = self._repo_to_dataset_index[repo_id]
        sample["__benchmark_global_index"] = torch.tensor(int(idx), dtype=torch.long)
        sample["__benchmark_local_index"] = torch.tensor(-1 if local_idx is None else int(local_idx), dtype=torch.long)
        sample["__benchmark_dataset_index"] = torch.tensor(-1 if ds_idx is None else int(ds_idx), dtype=torch.long)
        sample["__benchmark_sample_load_s"] = torch.tensor(elapsed, dtype=torch.float64)
        sample["__benchmark_repo_id"] = repo_id or "<unknown>"
        return sample


class TimedIterableDataset(IterableDataset):
    """Best-effort timing wrapper for streaming datasets."""

    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self.repo_ids = _repo_ids_for_dataset(dataset)
        for name in ("dataset_weights", "num_frames", "num_episodes", "meta", "datasets"):
            if hasattr(dataset, name):
                setattr(self, name, getattr(dataset, name))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for sample in self.dataset:
            # For IterableDataset, timing covers the iterator next() body before yield.
            elapsed = 0.0
            if not isinstance(sample, dict):
                sample = {"sample": sample}
            else:
                sample = dict(sample)
            ds_idx = _tensor_scalar_to_int(sample.get("dataset_index"))
            repo_id = None
            if ds_idx is not None and 0 <= ds_idx < len(self.repo_ids):
                repo_id = self.repo_ids[ds_idx]
            sample["__benchmark_global_index"] = torch.tensor(-1, dtype=torch.long)
            sample["__benchmark_local_index"] = torch.tensor(-1, dtype=torch.long)
            sample["__benchmark_dataset_index"] = torch.tensor(-1 if ds_idx is None else ds_idx, dtype=torch.long)
            sample["__benchmark_sample_load_s"] = torch.tensor(elapsed, dtype=torch.float64)
            sample["__benchmark_repo_id"] = repo_id or "<unknown>"
            yield sample

    def __len__(self) -> int:
        length = getattr(self.dataset, "__len__", None)
        if callable(length):
            return len(self.dataset)
        raise TypeError(f"{self.__class__.__name__} has no length")

    def __getattr__(self, name: str) -> Any:
        return getattr(self.dataset, name)


def _wrap_dataset_for_timing(dataset: Any) -> Any:
    if isinstance(dataset, IterableDataset):
        return TimedIterableDataset(dataset)
    return TimedMapDataset(dataset)


def _build_training_dataloader(
    dataset: Any,
    cfg: Any,
    device: torch.device,
    fastwam_worker_init_fn: Any,
    *,
    sampler_dataset: Any | None = None,
    in_order: bool = True,
):
    # This mirrors lerobot.scripts.lerobot_train exactly for the training dataloader branch.
    sampler_dataset = sampler_dataset or dataset
    if _is_fastwam_policy_type(cfg.policy.type):
        fastwam_module = _fastwam_policy_module(cfg.policy.type)
        ResumableEpochSampler = import_module(
            f"lerobot.policies.{fastwam_module}.core.utils.samplers"
        ).ResumableEpochSampler
        shuffle = False
        sampler = ResumableEpochSampler(
            dataset=sampler_dataset,
            seed=0 if cfg.seed is None else cfg.seed,
            batch_size=cfg.batch_size,
            num_processes=1 if cfg.dataset.dist_loading else torch.distributed.get_world_size()
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1,
        )
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = fastwam_worker_init_fn
    elif not cfg.dataset.streaming and hasattr(dataset, "dataset_weights") and dataset.dataset_weights is not None:
        shuffle = False
        sampler = MultiLeRobotWeightedSampler(dataset=sampler_dataset)
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None
    elif cfg.dataset.streaming:
        shuffle = False
        sampler = None
        num_workers = 1
        prefetch_factor = 4
        worker_init_fn = None
    else:
        shuffle = True
        sampler = None
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None

    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=prefetch_factor,
        worker_init_fn=worker_init_fn,
        in_order=in_order,
    )


def _extract_sample_records(batch: Any, rank: int, step: int) -> list[SampleLoadRecord]:
    if not isinstance(batch, dict):
        return []
    size = _batch_size(batch) or 0
    repos = _as_list(batch.get("__benchmark_repo_id"), size)
    dataset_indices = _as_list(batch.get("__benchmark_dataset_index"), size)
    global_indices = _as_list(batch.get("__benchmark_global_index"), size)
    local_indices = _as_list(batch.get("__benchmark_local_index"), size)
    elapsed = _as_list(batch.get("__benchmark_sample_load_s"), size)
    rows: list[SampleLoadRecord] = []
    for i in range(size):
        rows.append(
            SampleLoadRecord(
                rank=rank,
                step=step,
                sample_in_batch=i,
                repo_id=None if repos[i] in (None, "<unknown>") else str(repos[i]),
                dataset_index=_tensor_scalar_to_int(dataset_indices[i]),
                global_index=_tensor_scalar_to_int(global_indices[i]),
                local_index=_tensor_scalar_to_int(local_indices[i]),
                elapsed_s=_coerce_optional_float(elapsed[i]),
                pid=os.getpid(),
            )
        )
    return rows


def _strip_benchmark_keys(batch: Any) -> Any:
    if not isinstance(batch, dict):
        return batch
    return {k: v for k, v in batch.items() if not k.startswith("__benchmark_")}


def _summarize_values(values: list[float]) -> dict[str, Any]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return {"count": 0}
    def pct(q: float) -> float:
        pos = (len(clean) - 1) * q / 100.0
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return clean[lo]
        return clean[lo] * (hi - pos) + clean[hi] * (pos - lo)
    total = sum(clean)
    return {
        "count": len(clean),
        "total_s": total,
        "mean_s": total / len(clean),
        "min_s": clean[0],
        "max_s": clean[-1],
        "p50_s": pct(50),
        "p90_s": pct(90),
        "p95_s": pct(95),
        "p99_s": pct(99),
    }


def _repo_summary(sample_records: list[SampleLoadRecord]) -> dict[str, Any]:
    by_repo: dict[str, list[float]] = defaultdict(list)
    counts = Counter()
    for row in sample_records:
        repo = row.repo_id or "<unknown>"
        counts[repo] += 1
        if row.elapsed_s is not None:
            by_repo[repo].append(row.elapsed_s)
    return {
        repo: {"samples": counts[repo], "sample_load_s": _summarize_values(times)}
        for repo, times in sorted(by_repo.items(), key=lambda item: (-len(item[1]), item[0]))
    }


def _rank_payload(records: list[BatchLoadRecord], samples: list[SampleLoadRecord], metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata,
        "batch_load_s": _summarize_values([r.elapsed_s for r in records]),
        "send_to_device_s": _summarize_values([r.send_to_device_s for r in records if r.send_to_device_s is not None]),
        "sample_load_s": _summarize_values([r.elapsed_s for r in samples if r.elapsed_s is not None]),
        "repo_summary": _repo_summary(samples),
        "batches": [asdict(r) for r in records],
    }


def main(
    argv: list[str] | None = None,
    *,
    load_config: Callable[[str], Any] | None = None,
) -> None:
    args = build_parser().parse_args(argv)
    accelerator = Accelerator()
    output_dir = _make_output_dir(args.output_dir, args.exact_output_dir, accelerator)
    rank = accelerator.process_index
    if accelerator.is_main_process:
        print(f"[train-dataloader-benchmark] 输出目录: {output_dir}", flush=True)

    config_loader = _load_cfg if load_config is None else load_config
    cfg = config_loader(args.config_path)
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if cfg.dataset.dist_loading and accelerator.num_processes <= 1:
        raise ValueError("配置启用了 dataset.dist_loading=true，请使用 accelerate launch 多进程运行")

    fastwam_worker_init_fn = None
    if cfg.seed is not None:
        if _is_fastwam_policy_type(cfg.policy.type):
            fastwam_module = _fastwam_policy_module(cfg.policy.type)
            set_fastwam_global_seed = import_module(
                f"lerobot.policies.{fastwam_module}.core.utils.pytorch_utils"
            ).set_global_seed
            fastwam_worker_init_fn = set_fastwam_global_seed(cfg.seed, get_worker_init_fn=True)
        else:
            set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    cfg.policy.device = str(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    from lerobot.datasets.factory import make_dataset

    parallel_dataset_load = _env_flag("LEROBOT_PARALLEL_DATASET_LOAD", default=False)
    dataset_start = time.perf_counter()
    if parallel_dataset_load:
        dataset, data_stats = make_dataset(cfg)
        accelerator.wait_for_everyone()
    else:
        if accelerator.is_main_process:
            dataset, data_stats = make_dataset(cfg)
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            dataset, data_stats = make_dataset(cfg)
        accelerator.wait_for_everyone()
    dataset_init_s = time.perf_counter() - dataset_start

    from tools.bpva_benchmark.train_benchmark import dataset_inventory

    inventory = dataset_inventory(dataset)
    if accelerator.is_main_process:
        frames = inventory.get("total_frames")
        frames_text = f"{frames:,}" if frames is not None else "未知"
        print(
            f"[train-dataloader-benchmark] 数据集个数={inventory.get('dataset_count')} "
            f"总帧数={frames_text}",
            flush=True,
        )

    if accelerator.num_processes > 1:
        _ = gather_object(data_stats, accelerator)

    raw_dataset = dataset
    dataset = _wrap_dataset_for_timing(raw_dataset)
    dataloader = _build_training_dataloader(
        dataset,
        cfg,
        device,
        fastwam_worker_init_fn,
        sampler_dataset=raw_dataset,
        in_order=not args.out_of_order,
    )
    accelerator.wait_for_everyone()
    if not cfg.dataset.dist_loading:
        dataloader = accelerator.prepare(dataloader)

    rank_output = output_dir / f"rank_{rank:02d}"
    batch_jsonl = rank_output / "batches.jsonl"
    sample_jsonl = rank_output / "samples.jsonl"
    for path in (batch_jsonl, sample_jsonl):
        if path.exists():
            path.unlink()

    iterator = iter(dataloader)
    batch_records: list[BatchLoadRecord] = []
    sample_records: list[SampleLoadRecord] = []

    total_steps = args.warmup_batches + args.measure_batches
    for step in range(total_steps):
        phase = "warmup" if step < args.warmup_batches else "measure"
        measured_step = step - args.warmup_batches
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            batch = next(iterator)
        next_elapsed = time.perf_counter() - start

        samples = _extract_sample_records(batch, rank, measured_step if phase == "measure" else -step - 1)
        batch_size = _batch_size(batch)
        repo_counts = Counter(row.repo_id or "<unknown>" for row in samples)
        sample_times = [row.elapsed_s for row in samples if row.elapsed_s is not None]

        send_elapsed = None
        if cfg.dataset.dist_loading and not args.skip_send_to_device:
            stripped = _strip_benchmark_keys(batch)
            send_start = time.perf_counter()
            _ = send_to_device(stripped, device, non_blocking=True)
            send_elapsed = time.perf_counter() - send_start

        if phase == "measure" or args.record_warmup:
            record_step = measured_step if phase == "measure" else -step - 1
            batch_record = BatchLoadRecord(
                rank=rank,
                step=record_step,
                batch_size=batch_size,
                elapsed_s=next_elapsed,
                send_to_device_s=send_elapsed,
                sample_mean_s=(sum(sample_times) / len(sample_times)) if sample_times else None,
                sample_max_s=max(sample_times) if sample_times else None,
                repos=dict(repo_counts),
                pid=os.getpid(),
            )
            batch_records.append(batch_record)
            sample_records.extend(samples)
            _append_jsonl(batch_jsonl, [batch_record])
            _append_jsonl(sample_jsonl, samples)

        if accelerator.is_main_process and (step + 1 == args.warmup_batches or (phase == "measure" and (measured_step + 1) % 10 == 0)):
            print(
                f"[train-dataloader-benchmark] phase={phase} step={step + 1}/{total_steps} "
                f"last_next={next_elapsed:.4f}s",
                flush=True,
            )

    metadata = {
        "config_path": args.config_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "world_size": accelerator.num_processes,
        "local_process_index": accelerator.local_process_index,
        "pid": os.getpid(),
        "dataset_init_s": dataset_init_s,
        "policy_type": _config_type_name(cfg.policy),
        "dataset_type": _config_type_name(cfg.dataset),
        "dataset_dist_loading": bool(cfg.dataset.dist_loading),
        "dataset_streaming": bool(cfg.dataset.streaming),
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "warmup_batches": args.warmup_batches,
        "measure_batches": args.measure_batches,
        "skip_send_to_device": args.skip_send_to_device,
        "dataloader_in_order": not args.out_of_order,
        "repo_ids_for_rank": getattr(dataset, "repo_ids", []),
        "dataset_count": inventory.get("dataset_count"),
        "total_frames": inventory.get("total_frames"),
    }
    rank_summary = _rank_payload(batch_records, sample_records, metadata)
    _write_json(rank_output / "summary.json", rank_summary)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        partials = []
        for r in range(accelerator.num_processes):
            path = output_dir / f"rank_{r:02d}" / "summary.json"
            with path.open("r", encoding="utf-8") as f:
                partials.append(json.load(f))

        all_batches = [batch for partial in partials for batch in partial["batches"]]
        all_repo_summary: dict[str, list[float]] = defaultdict(list)
        all_repo_counts = Counter()
        for partial in partials:
            for repo, stats in partial.get("repo_summary", {}).items():
                all_repo_counts[repo] += int(stats.get("samples", 0))
            samples_path = output_dir / f"rank_{partial['metadata']['rank']:02d}" / "samples.jsonl"
            with samples_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    repo = row.get("repo_id") or "<unknown>"
                    elapsed = row.get("elapsed_s")
                    if elapsed is not None:
                        all_repo_summary[repo].append(float(elapsed))

        summary = {
            "metadata": {
                "config_path": args.config_path,
                "world_size": accelerator.num_processes,
                "output_dir": str(output_dir),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_count": inventory.get("dataset_count"),
                "total_frames": inventory.get("total_frames"),
            },
            "dataset_init_s_by_rank": [p["metadata"]["dataset_init_s"] for p in partials],
            "batch_load_s": _summarize_values([float(b["elapsed_s"]) for b in all_batches]),
            "send_to_device_s": _summarize_values([
                float(b["send_to_device_s"]) for b in all_batches if b.get("send_to_device_s") is not None
            ]),
            "sample_load_s_by_repo": {
                repo: {"samples": all_repo_counts[repo], "sample_load_s": _summarize_values(times)}
                for repo, times in sorted(all_repo_summary.items(), key=lambda item: (-len(item[1]), item[0]))
            },
            "rank_summaries": [
                {
                    "rank": p["metadata"]["rank"],
                    "dataset_init_s": p["metadata"]["dataset_init_s"],
                    "batch_load_s": p["batch_load_s"],
                    "sample_load_s": p["sample_load_s"],
                    "repo_count": len(p["metadata"].get("repo_ids_for_rank", [])),
                    "dataset_count": p["metadata"].get("dataset_count"),
                    "total_frames": p["metadata"].get("total_frames"),
                }
                for p in partials
            ],
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "repo_summary.json", summary["sample_load_s_by_repo"])
        frames = inventory.get("total_frames")
        frames_text = f"{frames:,}" if frames is not None else "未知"
        print(
            "[train-dataloader-benchmark] completed "
            f"batch_mean={summary['batch_load_s'].get('mean_s')} "
            f"dataset_count={inventory.get('dataset_count')} "
            f"total_frames={frames_text} "
            f"output={output_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
