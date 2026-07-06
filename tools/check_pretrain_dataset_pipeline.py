#!/usr/bin/env python
"""Validate TBot-SA1 pretrain dataset loading, weight-rule grouping, and transform pipeline.

This script mirrors the logic in ``lerobot.datasets.factory.make_dataset`` without
starting full training. It checks:

1. ``repo_id_file`` paths exist and expose LeRobot v3 ``meta/info.json``
2. ``weight_rules.yaml`` grouping and normalized sampling weights
3. ``dist_loading`` rank assignment (optional, per simulated world size)
4. TBot-SA1 preprocessor / transform chain on sample frames (optional)

Example::

    cd /home/jovyan/vla/workspace/mytbot
    source /home/jovyan/.conda/envs/tbot/bin/activate
    export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"

    python tools/check_pretrain_dataset_pipeline.py \\
        --config-path .config/pretrain_config.jsonc \\
        --repo-id-file .config/ds_ids/data.txt \\
        --weight-rules-path .config/weight_rules.yaml \\
        --world-size 8 \\
        --samples-per-group 1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import (  # noqa: E402
    _build_single_dataset,
    compute_group_balanced_repo_assignment,
    compute_repo_weights,
    find_info_json_path_for_repo,
    group_repo_ids_by_rules,
    load_info_for_repos,
    resolve_repo_ids,
)
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_IMAGES,
    OBS_PREFIX,
    OBS_STATE,
    SAMPLE_ACTION_LOSS_MASK,
)

DEFAULT_CONFIG = REPO_ROOT / ".config" / "pretrain_config.jsonc"
DEFAULT_REPO_ID_FILE = REPO_ROOT / ".config" / "ds_ids" / "data.txt"
DEFAULT_WEIGHT_RULES = REPO_ROOT / ".config" / "weight_rules.yaml"

EXPECTED_BATCH_KEYS = (
    OBS_STATE,
    ACTION,
    SAMPLE_ACTION_LOSS_MASK,
    f"{OBS_IMAGES}.image0",
    f"{OBS_IMAGES}.image1",
    f"{OBS_IMAGES}.image2",
    f"{OBS_IMAGES}.image0_mask",
    f"{OBS_IMAGES}.image1_mask",
    f"{OBS_IMAGES}.image2_mask",
    f"{OBS_PREFIX}pixel_values",
    f"{OBS_PREFIX}image_grid_thw",
    f"{OBS_PREFIX}input_ids",
    f"{OBS_PREFIX}attention_mask",
)


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.info.append(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check TBot-SA1 pretrain dataset grouping and preprocessor pipeline.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Train JSONC config (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--repo-id-file",
        type=Path,
        default=None,
        help="Override dataset.repo_id_file from config.",
    )
    parser.add_argument(
        "--weight-rules-path",
        type=Path,
        default=None,
        help="Override dataset.weight_rules_path from config.",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=8,
        help="Simulated distributed world size for dist_loading assignment.",
    )
    parser.add_argument(
        "--simulate-rank",
        type=int,
        default=0,
        help="Which rank to use when building sample datasets (dist_loading).",
    )
    parser.add_argument(
        "--samples-per-group",
        type=int,
        default=1,
        help="Number of repos per weight group to run through the transform pipeline.",
    )
    parser.add_argument(
        "--sample-indices",
        type=str,
        default="0,100,1000",
        help="Comma-separated dataset indices to fetch for each sampled repo.",
    )
    parser.add_argument(
        "--skip-preprocessor",
        action="store_true",
        help="Only run static checks (paths, grouping, weights, dist_loading).",
    )
    parser.add_argument(
        "--skip-qwen-processor",
        action="store_true",
        help="Run transforms until just before Qwen3_VLProcessorTransformFn (faster, no HF model load).",
    )
    parser.add_argument(
        "--strict-default-group",
        action="store_true",
        default=True,
        help="Treat unmatched weight-rule repos as errors (default: on).",
    )
    parser.add_argument(
        "--no-strict-default-group",
        action="store_false",
        dest="strict_default_group",
        help="Allow repos to fall into the default weight group.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable summary JSON at the end.",
    )
    return parser.parse_args()


def load_train_config(config_path: Path) -> TrainPipelineConfig:
    cfg = TrainPipelineConfig.from_pretrained(config_path)
    cfg.validate()
    return cfg


def apply_overrides(cfg: TrainPipelineConfig, args: argparse.Namespace) -> None:
    if args.repo_id_file is not None:
        cfg.dataset.repo_id_file = str(args.repo_id_file.expanduser().resolve())
    if args.weight_rules_path is not None:
        cfg.dataset.weight_rules_path = str(args.weight_rules_path.expanduser().resolve())


def check_repo_paths(cfg: TrainPipelineConfig, repo_ids: list[str], report: Report) -> tuple[list[str], list[str]]:
    missing: list[str] = []
    valid: list[str] = []
    for rid in repo_ids:
        info_path = find_info_json_path_for_repo(cfg, rid)
        if info_path is None or not info_path.is_file():
            missing.append(rid)
            continue
        valid.append(rid)
    if missing:
        report.error(f"{len(missing)} repo path(s) missing meta/info.json.")
        for path in missing[:10]:
            report.error(f"  missing: {path}")
        if len(missing) > 10:
            report.error(f"  ... +{len(missing) - 10} more")
    else:
        report.note(f"All {len(valid)} repo paths contain meta/info.json.")
    return valid, missing


def summarize_grouping(
    cfg: TrainPipelineConfig,
    repo_ids: list[str],
    weight_rules_path: Path,
    report: Report,
    *,
    strict_default: bool,
) -> tuple[dict[str, list[str]], dict[str, float] | None, dict[str, int], dict[str, int]]:
    if not weight_rules_path.is_file():
        report.error(f"weight_rules file not found: {weight_rules_path}")
        return {}, None, {}, {}

    groups_cfg = OmegaConf.load(weight_rules_path)
    _, group_to_repos, ordered_groups = group_repo_ids_by_rules(repo_ids, groups_cfg)

    default_repos = group_to_repos.get("__default__", [])
    if default_repos:
        msg = f"{len(default_repos)} repo(s) fell into default weight group."
        if strict_default:
            report.error(msg)
            for rid in default_repos[:10]:
                report.error(f"  default: {rid}")
            if len(default_repos) > 10:
                report.error(f"  ... +{len(default_repos) - 10} more")
        else:
            report.warn(msg)

    report.note("Weight-rule group counts:")
    for group_name in ordered_groups:
        repos = group_to_repos.get(group_name, [])
        if repos:
            report.note(f"  {group_name}: {len(repos)} repos")

    frames_map, episodes_map = load_info_for_repos(cfg, repo_ids)
    repo_weights_map = compute_repo_weights(repo_ids, frames_map, episodes_map, groups_cfg)

    group_weight_budget = defaultdict(float)
    for rid, weight in repo_weights_map.items():
        _, group_to_repos_tmp, _ = group_repo_ids_by_rules([rid], groups_cfg)
        for group_name, repos in group_to_repos_tmp.items():
            if rid in repos:
                group_weight_budget[group_name] += weight
                break

    report.note("Normalized sampling weight budget per group:")
    for group_name in ordered_groups:
        if group_name in group_weight_budget:
            report.note(f"  {group_name}: {group_weight_budget[group_name]:.6f}")

    weight_sum = sum(repo_weights_map.values())
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-5):
        report.error(f"Repo weights sum to {weight_sum:.8f}, expected 1.0")

    return group_to_repos, repo_weights_map, frames_map, episodes_map


def summarize_dist_loading(
    cfg: TrainPipelineConfig,
    repo_ids: list[str],
    frames_map: dict[str, int],
    weight_rules_path: Path,
    world_size: int,
    report: Report,
) -> None:
    if not cfg.dataset.dist_loading:
        report.note("dataset.dist_loading=false; every rank would load all repos.")
        return

    if world_size <= 1:
        report.warn("world-size <= 1 while dist_loading=true (training would raise).")
        return

    groups_cfg = OmegaConf.load(weight_rules_path)
    rank_to_repos = compute_group_balanced_repo_assignment(
        repo_ids,
        frames_map,
        world_size,
        groups_cfg,
    )

    report.note(f"Simulated dist_loading assignment for world_size={world_size}:")
    for rank, repos in enumerate(rank_to_repos):
        frame_load = sum(frames_map.get(rid, 0) for rid in repos)
        report.note(f"  rank {rank}: repos={len(repos)}, frames={frame_load}")

    empty_ranks = [rank for rank, repos in enumerate(rank_to_repos) if not repos]
    if empty_ranks:
        report.error(f"dist_loading produced empty ranks: {empty_ranks}")


def pick_sample_repos(
    group_to_repos: dict[str, list[str]],
    samples_per_group: int,
) -> list[tuple[str, str]]:
    picks: list[tuple[str, str]] = []
    for group_name, repos in sorted(group_to_repos.items()):
        if group_name == "__default__" or not repos:
            continue
        for rid in repos[:samples_per_group]:
            picks.append((group_name, rid))
    return picks


def describe_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        finite = torch.isfinite(value).all().item() if value.numel() else True
        return (
            f"Tensor shape={tuple(value.shape)} dtype={value.dtype} "
            f"device={value.device} finite={finite}"
        )
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return f"{type(value).__name__}({value!r})"


def validate_batch(sample: dict[str, Any], cfg: TrainPipelineConfig, report: Report, *, label: str) -> None:
    missing = [key for key in EXPECTED_BATCH_KEYS if key not in sample]
    if missing:
        report.error(f"{label}: missing keys after transforms: {missing}")
        return

    extra = sorted(set(sample.keys()) - set(EXPECTED_BATCH_KEYS))
    if extra:
        report.warn(f"{label}: unexpected extra keys: {extra}")

    state = sample[OBS_STATE]
    action = sample[ACTION]
    if not isinstance(state, torch.Tensor) or state.ndim == 0:
        report.error(f"{label}: {OBS_STATE} is not a tensor with batch/time dims.")
    elif state.shape[-1] > cfg.policy.max_state_dim:
        report.error(
            f"{label}: {OBS_STATE} last dim {state.shape[-1]} > max_state_dim={cfg.policy.max_state_dim}"
        )

    if not isinstance(action, torch.Tensor):
        report.error(f"{label}: {ACTION} is not a tensor.")
    elif action.shape[0] != cfg.policy.chunk_size:
        report.warn(
            f"{label}: action length {action.shape[0]} != policy.chunk_size={cfg.policy.chunk_size}"
        )
    if isinstance(action, torch.Tensor) and action.shape[-1] > cfg.policy.max_action_dim:
        report.error(
            f"{label}: action last dim {action.shape[-1]} > max_action_dim={cfg.policy.max_action_dim}"
        )

    for cam_idx in range(3):
        image_key = f"{OBS_IMAGES}.image{cam_idx}"
        image = sample[image_key]
        if not isinstance(image, torch.Tensor):
            report.error(f"{label}: {image_key} is not a tensor.")
            continue
        if image.ndim < 3:
            report.error(f"{label}: {image_key} has unexpected ndim={image.ndim}")

    for key in EXPECTED_BATCH_KEYS:
        value = sample[key]
        if isinstance(value, torch.Tensor) and value.numel() and not torch.isfinite(value).all():
            report.error(f"{label}: non-finite values in {key}")

    report.note(f"{label}: transform output looks valid.")
    for key in EXPECTED_BATCH_KEYS:
        report.note(f"  {key}: {describe_value(sample[key])}")


def build_transforms(cfg: TrainPipelineConfig, skip_qwen: bool):
    transforms = list(cfg.dataset.data_transforms.inputs)
    if skip_qwen:
        transforms = [
            transform
            for transform in transforms
            if transform.__class__.__name__ != "Qwen3_VLProcessorTransformFn"
            and transform.__class__.__name__ != "UnifyTBotSA1InputsTransformFn"
        ]
    return transforms


def run_preprocessor_samples(
    cfg: TrainPipelineConfig,
    picks: list[tuple[str, str]],
    sample_indices: list[int],
    report: Report,
    *,
    skip_qwen: bool,
) -> None:
    if skip_qwen:
        report.warn("Skipping Qwen3_VLProcessorTransformFn / UnifyTBotSA1InputsTransformFn checks.")

    if cfg.dataset.dist_loading and cfg.dataset.weight_rules_path:
        report.note(
            "Preprocessor sampling uses explicit repo picks; dist_loading rank filtering is not applied here."
        )

    original_transforms = cfg.dataset.data_transforms.inputs
    cfg.dataset.data_transforms = replace(
        cfg.dataset.data_transforms,
        inputs=build_transforms(cfg, skip_qwen),
    )

    for group_name, repo_id in picks:
        label = f"[{group_name}] {repo_id}"
        try:
            dataset, stats_copy, robot_type = _build_single_dataset(
                cfg,
                repo_id,
                image_transforms=None,
                seed_offset=0,
            )
        except Exception as exc:
            report.error(f"{label}: failed to build dataset: {exc}")
            report.error(traceback.format_exc())
            continue

        report.note(
            f"{label}: built dataset robot_type={robot_type}, "
            f"num_frames={dataset.num_frames}, num_episodes={dataset.num_episodes}"
        )

        for index in sample_indices:
            if index >= len(dataset):
                report.warn(f"{label}: index {index} out of range (len={len(dataset)}), skipped.")
                continue
            try:
                sample = dataset[index]
            except Exception as exc:
                report.error(f"{label}: __getitem__({index}) failed: {exc}")
                report.error(traceback.format_exc())
                continue

            if skip_qwen:
                report.note(f"{label} idx={index}: loaded raw transformed keys={sorted(sample.keys())}")
                for key, value in sorted(sample.items()):
                    report.note(f"  {key}: {describe_value(value)}")
            else:
                validate_batch(sample, cfg, report, label=f"{label} idx={index}")

    cfg.dataset.data_transforms = replace(cfg.dataset.data_transforms, inputs=original_transforms)


def print_report(report: Report) -> None:
    if report.info:
        print("\n[INFO]")
        for line in report.info:
            print(line)

    if report.warnings:
        print("\n[WARN]")
        for line in report.warnings:
            print(line)

    if report.errors:
        print("\n[ERROR]")
        for line in report.errors:
            print(line)

    status = "PASS" if report.ok else "FAIL"
    print(f"\n==> {status}: {len(report.errors)} error(s), {len(report.warnings)} warning(s)")


def main() -> int:
    args = parse_args()
    report = Report()

    config_path = args.config_path.expanduser().resolve()
    if not config_path.is_file():
        report.error(f"Config not found: {config_path}")
        print_report(report)
        return 1

    if args.strict_default_group:
        os.environ["LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_MODE"] = "off"
    else:
        os.environ.pop("LEROBOT_WEIGHT_RULES_DEFAULT_GROUP_MODE", None)

    try:
        cfg = load_train_config(config_path)
    except Exception as exc:
        report.error(f"Failed to load config {config_path}: {exc}")
        print_report(report)
        return 1

    apply_overrides(cfg, args)

    if cfg.policy is None or cfg.policy.type not in {"TBot_SA1", "tbot_sa1"}:
        report.warn(f"Expected policy.type=TBot_SA1, got {getattr(cfg.policy, 'type', None)!r}.")

    repo_id_file = Path(cfg.dataset.repo_id_file) if cfg.dataset.repo_id_file else None
    if repo_id_file is None:
        report.error("dataset.repo_id_file is not set.")
        print_report(report)
        return 1
    if not repo_id_file.is_file():
        report.error(f"repo_id_file not found: {repo_id_file}")
        print_report(report)
        return 1

    weight_rules_path = (
        Path(cfg.dataset.weight_rules_path)
        if cfg.dataset.weight_rules_path
        else DEFAULT_WEIGHT_RULES
    )

    report.note(f"Config: {config_path}")
    report.note(f"repo_id_file: {repo_id_file} ({sum(1 for _ in open(repo_id_file))} lines)")
    report.note(f"weight_rules_path: {weight_rules_path}")
    report.note(
        f"dataset.action_mode={cfg.dataset.action_mode}, "
        f"use_external_stats={cfg.dataset.use_external_stats}, "
        f"use_imagenet_stats={cfg.dataset.use_imagenet_stats}, "
        f"dist_loading={cfg.dataset.dist_loading}"
    )

    try:
        repo_ids = resolve_repo_ids(cfg)
    except Exception as exc:
        report.error(f"resolve_repo_ids failed: {exc}")
        print_report(report)
        return 1

    valid_repo_ids, _ = check_repo_paths(cfg, repo_ids, report)
    if not valid_repo_ids:
        print_report(report)
        return 1

    group_to_repos, repo_weights_map, frames_map, episodes_map = summarize_grouping(
        cfg,
        valid_repo_ids,
        weight_rules_path,
        report,
        strict_default=args.strict_default_group,
    )

    if repo_weights_map is not None:
        top = sorted(repo_weights_map.items(), key=lambda item: item[1], reverse=True)[:5]
        report.note("Top repo sampling weights:")
        for rid, weight in top:
            report.note(f"  {weight:.8f}  {rid}")

    summarize_dist_loading(
        cfg,
        valid_repo_ids,
        frames_map,
        weight_rules_path,
        args.world_size,
        report,
    )

    if not args.skip_preprocessor:
        sample_indices = [int(x.strip()) for x in args.sample_indices.split(",") if x.strip()]
        picks = pick_sample_repos(group_to_repos, args.samples_per_group)
        if not picks:
            report.warn("No repos selected for preprocessor sampling.")
        else:
            report.note(f"Running preprocessor checks on {len(picks)} repo(s)...")
            run_preprocessor_samples(
                cfg,
                picks,
                sample_indices,
                report,
                skip_qwen=args.skip_qwen_processor,
            )

    print_report(report)

    if args.json:
        summary = {
            "ok": report.ok,
            "errors": report.errors,
            "warnings": report.warnings,
            "repo_count": len(valid_repo_ids),
            "groups": {name: len(repos) for name, repos in group_to_repos.items() if repos},
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
