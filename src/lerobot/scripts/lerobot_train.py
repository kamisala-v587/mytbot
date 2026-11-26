#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import os
import shutil
import subprocess
import time
import inspect
from importlib import import_module
from math import ceil
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import multiprocessing as mp
from accelerate import Accelerator
from accelerate.state import DistributedType
from accelerate.utils import send_to_device
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import MultiLeRobotHomogeneousBatchSampler, MultiLeRobotWeightedSampler
from lerobot.datasets.transformed_dataset import MultiLeRobotDataset
from lerobot.datasets.utils import cycle, load_json, write_json
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.names import (
    TBOT_SA1_WAN,
    TBOT_SA1_WAN_ALIASES,
    TBOT_SA1_WAN_LEGACY_ALIASES,
    is_bpv5_stageb,
    is_bpv5_stageb_v2,
    is_bpvav3,
    is_tbot_sa1,
    is_tbot_sa1_wan,
)
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.constants import SAMPLE_ACTION_LOSS_MASK
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, append_loss_log, format_time
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
    gather_object, 
)

FASTWAM_TRAINER_STATE_FILE = "fastwam_trainer_state.json"
FASTWAM_POLICY_TYPES = {"fastwam", *TBOT_SA1_WAN_ALIASES}

is_main_process = False


def _is_local_main_process() -> bool:
    """True on rank 0 for both single-process and accelerate/torchrun launches."""
    for key in ("LOCAL_RANK", "RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value) == 0
            except ValueError:
                pass
    return True


def _metric_to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().item())
    return float(value)


def _optimizer_parameters(optimizer: Optimizer):
    for group in optimizer.param_groups:
        yield from group["params"]


def _is_fastwam_policy_type(policy_type: str | None) -> bool:
    return policy_type == "fastwam" or is_tbot_sa1_wan(policy_type)


def _fastwam_policy_module(policy_type: str) -> str:
    return "TBot_SA1_Wan" if is_tbot_sa1_wan(policy_type) else "fastwam"


def _fastwam_family_label(policy_type: str | None) -> str:
    return "TBot_SA1_Wan" if is_tbot_sa1_wan(policy_type) else "FastWAM"


def _fastwam_family_stats_key(policy_type: str | None) -> str:
    return TBOT_SA1_WAN if is_tbot_sa1_wan(policy_type) else "fastwam"


def _fastwam_family_trainer_state_file(policy_type: str | None) -> str:
    return "tbot_sa1_wan_trainer_state.json" if is_tbot_sa1_wan(policy_type) else FASTWAM_TRAINER_STATE_FILE


def _fastwam_family_stats_filename(policy_type: str | None) -> str:
    return "tbot_sa1_wan_dataset_stats.json" if is_tbot_sa1_wan(policy_type) else "fastwam_dataset_stats.json"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _make_training_accelerator(cfg: TrainPipelineConfig) -> Accelerator:
    """Build Accelerator for DDP (default) or opt-in FSDP FULL_SHARD."""
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        FullyShardedDataParallelPlugin,
        InitProcessGroupKwargs,
    )

    timeout = _env_int("LEROBOT_DDP_TIMEOUT_SEC", _env_int("DDP_TIMEOUT_SEC", 1800))
    handlers = [InitProcessGroupKwargs(timeout=timedelta(seconds=timeout))]
    options: dict[str, Any] = {}

    if cfg.distributed_backend == "ddp":
        find_unused_parameters = (
            _env_flag("LEROBOT_DDP_FIND_UNUSED_PARAMETERS", default=True)
            if cfg.ddp_find_unused_parameters is None
            else cfg.ddp_find_unused_parameters
        )
        handlers.insert(0, DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters))
    elif cfg.distributed_backend == "fsdp":
        # Accelerate upcasts FSDP flat parameters when its mixed precision is enabled.
        # Keep constructed BF16 backbone + FP32 IO; train loop already uses autocast.
        options["mixed_precision"] = "no"
        strategy = {
            "full_shard": "FULL_SHARD",
            "shard_grad_op": "SHARD_GRAD_OP",
            "no_shard": "NO_SHARD",
            "hybrid_shard": "HYBRID_SHARD",
            "hybrid_shard_zero2": "_HYBRID_SHARD_ZERO2",
        }[cfg.fsdp_sharding_strategy]
        options["fsdp_plugin"] = FullyShardedDataParallelPlugin(
            fsdp_version=1,
            sharding_strategy=strategy,
            auto_wrap_policy="no_wrap",
            use_orig_params=bool(cfg.fsdp_use_orig_params),
            state_dict_type="SHARDED_STATE_DICT",
            sync_module_states=True,
        )
    else:
        raise ValueError(f"Unsupported distributed_backend={cfg.distributed_backend!r}")

    accelerator = Accelerator(
        step_scheduler_with_optimizer=False,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        kwargs_handlers=handlers,
        **options,
    )
    if cfg.distributed_backend == "ddp":
        expected = {DistributedType.MULTI_GPU, DistributedType.MULTI_CPU, DistributedType.NO}
    else:
        expected = {DistributedType.FSDP}
    if accelerator.num_processes > 1 and accelerator.distributed_type not in expected:
        raise RuntimeError(
            f"Requested distributed_backend={cfg.distributed_backend}, "
            f"but Accelerator landed on {accelerator.distributed_type}"
        )
    if cfg.distributed_backend == "fsdp" and accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            f"Requested fsdp but Accelerator distributed_type is {accelerator.distributed_type} "
            "(refusing silent DDP fallback)"
        )
    if cfg.distributed_backend == "fsdp" and accelerator.num_processes <= 1:
        raise RuntimeError("distributed_backend=fsdp requires a multi-process launch")
    return accelerator


def _isolate_demo_boundary_module(stage_model: torch.nn.Module) -> torch.nn.Module:
    """Re-parent demo_start/demo_end into one module for a single-dtype FSDP unit.

    Attribute access ``model.demo_start`` / ``demo_end`` is preserved. Training
    state-dict keys become ``demo_boundary.demo_*`` under FSDP; document remapping
    when exporting a plain pretrained_model.
    """
    existing = getattr(stage_model, "demo_boundary", None)
    if isinstance(existing, torch.nn.Module) and any(existing.parameters()):
        return existing

    boundary = torch.nn.Module()
    for name in ("demo_start", "demo_end"):
        param = stage_model._parameters.pop(name, None)
        if param is None:
            raise RuntimeError(f"FSDP wrap expected parameter {name!r} on Stage B model")
        boundary.register_parameter(name, param)
        object.__setattr__(stage_model, name, param)
    stage_model.add_module("demo_boundary", boundary)
    return boundary


def _configure_bpv5_fsdp_wrap(
    policy: PreTrainedPolicy,
    accelerator: Accelerator,
    wrap_policy: str = "root",
) -> None:
    """Configure FSDP auto-wrap targets for BPv5 Stage B v2 (IO + optional joint pairs)."""
    if not (is_bpv5_stageb_v2(policy.config.type) or is_bpv5_stageb(policy.config.type)):
        raise ValueError(
            f"FSDP wrap is only implemented for bpv5_stageb_v2/stageb; got {policy.config.type!r}"
        )

    stage_model = policy.model
    core = stage_model.core
    targets: list[torch.nn.Module] = [
        core.state_projection,
        core.action_projection,
        core.action_time_in,
        core.action_time_out,
        core.action_head,
        stage_model.demo_encoder,
        _isolate_demo_boundary_module(stage_model),
    ]

    if wrap_policy == "layer":
        from lerobot.policies.q35.fsdp_joint import enable_fsdp_joint_pairs

        pairs = enable_fsdp_joint_pairs(core.backbone)
        targets.extend(list(pairs))
    elif wrap_policy != "root":
        raise ValueError(f"Unknown fsdp_wrap_policy={wrap_policy!r}")

    target_ids = {id(module) for module in targets}
    covered_ids = {id(parameter) for module in targets for parameter in module.parameters()}
    residual_dtypes = {
        parameter.dtype
        for parameter in policy.parameters()
        if id(parameter) not in covered_ids
    }
    if len(residual_dtypes) > 1:
        raise RuntimeError(
            "FSDP residual parameters have mixed dtypes after BPv5 IO wrapping: "
            f"{sorted(str(dtype) for dtype in residual_dtypes)}"
        )
    for module in targets:
        dtypes = {parameter.dtype for parameter in module.parameters()}
        if len(dtypes) != 1:
            raise RuntimeError(
                f"FSDP wrap target {type(module).__name__} contains mixed parameter dtypes: "
                f"{sorted(str(dtype) for dtype in dtypes)}"
            )

    # Always isolate IO / markers (and joint pairs when layer wrap). Root wrap
    # leaves transformer layers on the root FSDP unit; layer wrap adds pairs.
    if targets:
        accelerator.state.fsdp_plugin.auto_wrap_policy = (
            lambda module, recurse, nonwrapped_numel: True if recurse else id(module) in target_ids
        )


@torch.no_grad()
def _clip_fsdp_mixed_dtype_grad_norm_(
    model,
    accelerator: Accelerator,
    max_norm: float,
) -> torch.Tensor:
    """FSDP1 global L2 clipping for mixed BF16/FP32 gradient dtypes.

    FSDP1 rejects mixed gradient dtypes in its local norm helper; accumulate
    squared norms in FP32 while retaining sharded/nonsharded accounting.
    """
    accelerator.unscale_gradients()
    sharded: list[torch.nn.Parameter] = []
    nonsharded: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for handle in model._all_handles:
        parameters = handle.flat_param._params if handle._use_orig_params else (handle.flat_param,)
        destination = sharded if handle.uses_sharded_strategy else nonsharded
        for parameter in parameters:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                destination.append(parameter)
    for parameter in model.parameters():
        if id(parameter) not in seen:
            seen.add(id(parameter))
            nonsharded.append(parameter)

    def squared_norm(parameters):
        total = torch.zeros((), device=model.compute_device, dtype=torch.float32)
        for parameter in parameters:
            if parameter.grad is not None:
                total += torch.linalg.vector_norm(
                    parameter.grad.detach(),
                    ord=2,
                    dtype=torch.float32,
                ).square()
        return total

    total_squared = squared_norm(sharded)
    if any(handle.uses_sharded_strategy for handle in model._all_handles):
        torch.distributed.all_reduce(total_squared, group=model.process_group)
    total_norm = (total_squared + squared_norm(nonsharded)).sqrt()
    factor = torch.clamp(float(max_norm) / (total_norm + 1e-6), max=1.0)
    for parameter in (*sharded, *nonsharded):
        if parameter.grad is not None:
            parameter.grad.mul_(factor.to(parameter.grad.device, parameter.grad.dtype))
    return total_norm


def _parameter_counts(parameters) -> tuple[str, int, int]:
    params = list(parameters)
    if not params:
        return "", 0, 0
    trainable = sum(param.numel() for param in params if param.requires_grad)
    frozen = sum(param.numel() for param in params if not param.requires_grad)
    if trainable and frozen:
        state = "混合"
    elif trainable:
        state = "更新"
    else:
        state = "冻结"
    return state, trainable, frozen


def _freeze_group_label(path: str) -> str:
    name = path.rsplit(".", 1)[-1]
    if name == "align_queries":
        return "align_queries"
    if name == "visual":
        return "共享 Qwen3.5 ViT（当前观测和 BP 图像）"
    if name == "bp_obs_encoder":
        return "BP encoder（投影、MLP、compressor，不含 ViT）"
    if name == "language_model":
        return "VLM 语言层"
    if name == "expert":
        return "action expert"
    if name in {
        "state_projection",
        "action_projection",
        "action_time_in",
        "action_time_out",
        "action_head",
    }:
        return name
    return path


def _collect_freeze_groups(module: torch.nn.Module, prefix: str) -> list[tuple[str, str, int, int]]:
    direct = list(module.named_parameters(recurse=False))
    children = list(module.named_children())
    all_params = list(module.parameters())
    state, trainable, frozen = _parameter_counts(all_params)
    if not state:
        return []
    if state != "混合" or not children:
        return [(prefix or module.__class__.__name__, state, trainable, frozen)]

    groups = []
    for name, param in direct:
        param_state, param_trainable, param_frozen = _parameter_counts([param])
        param_path = f"{prefix}.{name}" if prefix else name
        groups.append((param_path, param_state, param_trainable, param_frozen))
    for name, child in children:
        child_path = f"{prefix}.{name}" if prefix else name
        groups.extend(_collect_freeze_groups(child, child_path))
    return groups


def _format_freeze_summary(policy: torch.nn.Module) -> str:
    groups = _collect_freeze_groups(policy, "")
    lines = ["参数分组:"]
    for title in ("冻结", "更新", "混合"):
        matched = [group for group in groups if group[1] == title]
        lines.append(f"{title}:")
        if not matched:
            lines.append("  （无）")
            continue
        for path, _state, trainable, frozen in matched:
            count = frozen if title == "冻结" else trainable if title == "更新" else trainable + frozen
            lines.append(
                f"  {_freeze_group_label(path)}  {format_big_number(count)} 参数  ({path})"
            )
    return "\n".join(lines)


def _format_policy_summary(policy: torch.nn.Module) -> str:
    params = list(policy.parameters())
    total_params = sum(param.numel() for param in params)
    trainable_params = sum(param.numel() for param in params if param.requires_grad)
    trainable_pct = (100.0 * trainable_params / total_params) if total_params else 0.0

    first_param = params[0] if params else None
    device = str(first_param.device) if first_param is not None else "n/a"
    dtype = str(first_param.dtype).removeprefix("torch.") if first_param is not None else "n/a"

    return (
        f"{policy.__class__.__name__}("
        f"total_params={format_big_number(total_params)}, "
        f"trainable_params={format_big_number(trainable_params)} "
        f"({trainable_pct:.2f}%), "
        f"device={device}, dtype={dtype}"
        ")"
    )


def _batch_item(value: Any, index: int) -> Any:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        return value[index]
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def _action_sample_enabled(batch: dict[str, Any], index: int) -> bool:
    sample_mask = _batch_item(batch.get(SAMPLE_ACTION_LOSS_MASK), index)
    if sample_mask is None:
        return True
    if isinstance(sample_mask, torch.Tensor):
        if sample_mask.numel() == 0:
            return False
        return bool(sample_mask.detach().flatten()[0].item() > 0.5)
    return bool(float(sample_mask) > 0.5)


def _action_eval_mask(batch: dict[str, Any], index: int, shape: torch.Size | tuple[int, int]) -> torch.Tensor:
    horizon, action_dim = int(shape[0]), int(shape[1])
    valid = torch.ones((horizon, action_dim), dtype=torch.bool)

    action_is_pad = _batch_item(batch.get("action_is_pad"), index)
    if action_is_pad is not None:
        step_valid = ~torch.as_tensor(action_is_pad, dtype=torch.bool).flatten()[:horizon]
        valid = valid & step_valid[:, None]

    action_dim_is_pad = _batch_item(batch.get("action_dim_is_pad"), index)
    if action_dim_is_pad is not None:
        dim_valid = ~torch.as_tensor(action_dim_is_pad, dtype=torch.bool).flatten()[:action_dim]
        valid = valid & dim_valid[None, :]

    return valid


def _action_eval_seed(batch: dict[str, Any], index: int) -> int:
    sample_idx = _batch_item(batch.get("idx"), index)
    if sample_idx is None:
        return 17_123 + index
    if isinstance(sample_idx, torch.Tensor):
        if sample_idx.numel() == 0:
            return 17_123 + index
        sample_idx = int(sample_idx.detach().flatten()[0].item())
    else:
        sample_idx = int(sample_idx)
    return int((17_123 + sample_idx) % (2**31 - 1))


@torch.no_grad()
def evaluate_tbot_sa1_wan_action_policy(
    policy: torch.nn.Module,
    dataloader,
    accelerator: Accelerator,
    *,
    max_batches: int = 2,
) -> dict[str, float]:
    """Run a small open-loop action evaluation pass with inference metrics."""
    del accelerator
    was_training = policy.training
    policy.eval()

    action_mse_per_sample: list[float] = []
    action_l2_per_sample: list[float] = []
    num_batches = 0
    num_valid_samples = 0

    infer_action_params = inspect.signature(policy.model.infer_action).parameters
    infer_needs_num_video_frames = "num_video_frames" in infer_action_params

    try:
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            if batch is None:
                continue

            num_batches += 1
            action_target_batch = batch.get("action")
            if not isinstance(action_target_batch, torch.Tensor):
                continue
            input_images = policy._resolve_input_image(batch)
            batch_size = int(action_target_batch.shape[0])
            num_video_frames = int(batch["video"].shape[2]) if isinstance(batch.get("video"), torch.Tensor) else None

            for index in range(batch_size):
                if not _action_sample_enabled(batch, index):
                    continue

                prompt, context, context_mask = policy._resolve_context_for_inference(batch, index)
                proprio = policy._resolve_proprio(batch, index)
                target_action = action_target_batch[index].detach().cpu().float()

                infer_kwargs = {
                    "prompt": prompt,
                    "input_image": input_images[index : index + 1],
                    "action_horizon": int(target_action.shape[0]),
                    "proprio": proprio,
                    "context": context,
                    "context_mask": context_mask,
                    "num_inference_steps": int(policy.config.num_inference_steps),
                    "seed": _action_eval_seed(batch, index),
                }
                if infer_needs_num_video_frames and num_video_frames is not None:
                    infer_kwargs["num_video_frames"] = num_video_frames

                output = policy.model.infer_action(**infer_kwargs)
                pred_action = output["action"].detach().cpu().float()

                horizon = min(int(pred_action.shape[0]), int(target_action.shape[0]))
                action_dim = min(int(pred_action.shape[1]), int(target_action.shape[1]))
                if horizon <= 0 or action_dim <= 0:
                    continue

                pred_action = pred_action[:horizon, :action_dim]
                target_action = target_action[:horizon, :action_dim]
                valid = _action_eval_mask(batch, index, target_action.shape)
                if not valid.any():
                    continue

                squared_error = (pred_action - target_action).pow(2)[valid]
                action_mse_per_sample.append(float(squared_error.mean().item()))
                action_l2_per_sample.append(float((squared_error.sqrt() / (1 + 1e-3)).mean().item()))
                num_valid_samples += 1
    finally:
        if was_training:
            policy.train()
        else:
            policy.eval()

    eval_metrics: dict[str, float] = {}
    if action_mse_per_sample:
        mse_tensor = torch.tensor(action_mse_per_sample, dtype=torch.float32)
        l2_tensor = torch.tensor(action_l2_per_sample, dtype=torch.float32)
        eval_metrics.update(
            {
                "action_mse_loss": float(mse_tensor.mean().item()),
                "action_l2_error": float(l2_tensor.mean().item()),
                "action_mse_std": float(mse_tensor.std(unbiased=False).item()),
                "action_l2_std": float(l2_tensor.std(unbiased=False).item()),
            }
        )
    eval_metrics["num_batches"] = float(num_batches)
    eval_metrics["num_samples"] = float(num_valid_samples)
    eval_metrics["num_inference_steps"] = float(policy.config.num_inference_steps)
    return eval_metrics


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    *,
    policy_forward_kwargs: dict[str, Any] | None = None,
    use_zero_grad_set_to_none: bool = False,
    skip_scheduler_when_optimizer_step_skipped: bool = False,
    clip_optimizer_params_only: bool = False,
) -> tuple[MetricsTracker, dict, bool, float]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    with accelerator.accumulate(policy):
        # Let accelerator handle mixed precision
        with accelerator.autocast():
            loss, output_dict = policy.forward(batch, **(policy_forward_kwargs or {}))

        # Use accelerator's backward method. This also scales by
        # gradient_accumulation_steps when appropriate.
        accelerator.backward(loss)

        grad_norm = None
        if accelerator.sync_gradients:
            grad_parameters = (
                _optimizer_parameters(optimizer)
                if clip_optimizer_params_only
                else policy.parameters()
            )
            use_fsdp_mixed_clip = accelerator.distributed_type == DistributedType.FSDP and (
                is_bpv5_stageb_v2(accelerator.unwrap_model(policy).config.type)
                or is_bpv5_stageb(accelerator.unwrap_model(policy).config.type)
            )
            if grad_clip_norm > 0:
                if use_fsdp_mixed_clip:
                    grad_norm = _clip_fsdp_mixed_dtype_grad_norm_(policy, accelerator, grad_clip_norm)
                else:
                    grad_norm = accelerator.clip_grad_norm_(grad_parameters, grad_clip_norm)
            elif use_fsdp_mixed_clip:
                grad_norm = _clip_fsdp_mixed_dtype_grad_norm_(policy, accelerator, float("inf"))
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    grad_parameters, float("inf"), error_if_nonfinite=False
                )

        # Optimizer step: Accelerate will turn this into a no-op on
        # non-sync micro-steps when gradient accumulation is enabled.
        with lock if lock is not None else nullcontext():
            optimizer.step()

        # This project initializes Accelerator with
        # step_scheduler_with_optimizer=False, so the prepared scheduler will
        # still advance whenever .step() is called. Gate it on sync steps so
        # gradient accumulation keeps the scheduler aligned with real optimizer
        # updates.
        should_step_scheduler = lr_scheduler is not None and accelerator.sync_gradients
        if should_step_scheduler and skip_scheduler_when_optimizer_step_skipped:
            should_step_scheduler = not bool(getattr(accelerator, "optimizer_step_was_skipped", False))
        if should_step_scheduler:
            lr_scheduler.step()

        if use_zero_grad_set_to_none:
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.zero_grad()

        # Update internal buffers if policy has update method only when an
        # optimizer step actually happened.
        if accelerator.sync_gradients and has_method(
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"
        ):
            accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    for metric_name in (
        "loss_action",
        "loss_align",
        "loss_align_w",
        "loss_video",
        "loss_gen",
        "loss_3d",
        "time_3d_teacher_forward_s",
    ):
        if metric_name in output_dict and metric_name in train_metrics.metrics:
            setattr(train_metrics, metric_name, _metric_to_float(output_dict[metric_name]))
    if accelerator.sync_gradients and grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics, output_dict, accelerator.sync_gradients, time.perf_counter() - start_time


def _sync_scalar_metrics(accelerator: Accelerator, metrics: dict[str, float]) -> dict[str, float]:
    """All-reduce a small set of scalar logging metrics across ranks."""
    if accelerator.num_processes <= 1 or not metrics:
        return metrics

    metric_names = list(metrics.keys())
    metric_tensor = torch.tensor(
        [float(metrics[name]) for name in metric_names],
        device=accelerator.device,
        dtype=torch.float32,
    )
    reduced_metrics = accelerator.reduce(metric_tensor, reduction="mean")
    return {
        name: float(value)
        for name, value in zip(metric_names, reduced_metrics.detach().cpu().tolist(), strict=False)
    }


def _meter_avg_or_val(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else meter.val


GPU_METRIC_KEYS = (
    "gpu_mem_allocated_gb",
    "gpu_mem_reserved_gb",
    "gpu_mem_peak_gb",
    "gpu_util_pct",
    "gpu_mem_used_gb",
)


class _GpuStatsMeter(AverageMeter):
    """Average actual samples; retain the latest value across logging windows."""

    def reset(self) -> None:
        latest = getattr(self, "val", float("nan"))
        super().reset()
        self.val = latest
        self.avg = latest


def _maybe_add_gpu_metrics(train_metrics: dict[str, AverageMeter]) -> None:
    if not torch.cuda.is_available():
        return
    train_metrics["gpu_mem_allocated_gb"] = _GpuStatsMeter("gpu_alloc_gb", ":.2f")
    train_metrics["gpu_mem_reserved_gb"] = _GpuStatsMeter("gpu_res_gb", ":.2f")
    train_metrics["gpu_mem_peak_gb"] = _GpuStatsMeter("gpu_peak_gb", ":.2f")
    train_metrics["gpu_util_pct"] = _GpuStatsMeter("gpu_util", ":.0f")
    train_metrics["gpu_mem_used_gb"] = _GpuStatsMeter("gpu_used_gb", ":.2f")


def _collect_gpu_stats(device: torch.device, local_rank: int) -> dict[str, float]:
    """Snapshot GPU memory (PyTorch) and utilization (nvidia-smi) for logging."""
    if not torch.cuda.is_available():
        return {}

    cuda_device = device if device.type == "cuda" else torch.device(f"cuda:{local_rank}")
    gpu_index = cuda_device.index if cuda_device.index is not None else local_rank
    stats: dict[str, float] = {}

    try:
        torch.cuda.synchronize(cuda_device)
        stats["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated(cuda_device) / (1024 ** 3)
        stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved(cuda_device) / (1024 ** 3)
        stats["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(cuda_device) / (1024 ** 3)
    except Exception:
        pass

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = [part.strip() for part in result.stdout.strip().split(",")]
            if len(parts) >= 1 and parts[0]:
                stats["gpu_util_pct"] = float(parts[0])
            if len(parts) >= 2 and parts[1]:
                stats["gpu_mem_used_gb"] = float(parts[1]) / 1024
    except Exception:
        pass

    return stats


def _update_gpu_metrics(train_tracker: MetricsTracker, device: torch.device, local_rank: int) -> None:
    for key, value in _collect_gpu_stats(device, local_rank).items():
        if key in train_tracker.metrics:
            setattr(train_tracker, key, value)


def _pop_worker_timing(batch: Any) -> dict[str, float] | None:
    """Mean/max of worker __getitem__ timers.

    data_s used to be only the main-process next() wait. With prefetch that wait is ~0
    while cache load, parquet reads and delta/norm still happen in workers.
    """
    if not isinstance(batch, dict) or "_timing" not in batch:
        return None
    timing = batch.pop("_timing")
    if not isinstance(timing, dict) or "getitem_s" not in timing:
        return None
    out: dict[str, float] = {}
    for key, value in timing.items():
        if not torch.is_tensor(value):
            continue
        flat = value.detach().float().reshape(-1)
        if flat.numel() == 0:
            continue
        out[key] = float(flat.mean())
        if key == "getitem_s":
            out["getitem_max_s"] = float(flat.max())
    return out or None


def _format_train_status_line(
    train_tracker: MetricsTracker,
    cfg: TrainPipelineConfig,
    *,
    elapsed_str: str,
    remaining_str: str,
    steps_per_second: float,
    loss_overrides: dict[str, float] | None = None,
) -> str:
    loss_overrides = loss_overrides or {}
    progress_parts = [
        f"step:{format_big_number(train_tracker.steps, precision=1)}",
        f"sample:{format_big_number(train_tracker.samples)}",
        f"episode:{format_big_number(train_tracker.episodes)}",
        f"epoch:{train_tracker.epochs:.2f}",
    ]

    loss_parts = []
    if "loss" in train_tracker.metrics:
        loss_value = loss_overrides.get("loss", _meter_avg_or_val(train_tracker.loss))
        loss_parts.append(f"total:{loss_value:.3f}")
    if "loss_action" in train_tracker.metrics:
        loss_action = loss_overrides.get("loss_action", _meter_avg_or_val(train_tracker.loss_action))
        loss_parts.append(f"action:{loss_action:.3f}")
        if "loss_action_w" in loss_overrides:
            loss_parts.append(f"action_w:{loss_overrides['loss_action_w']:.3f}")
    if "loss_video" in train_tracker.metrics:
        loss_video = loss_overrides.get("loss_video", _meter_avg_or_val(train_tracker.loss_video))
        loss_parts.append(f"video:{loss_video:.3f}")
        if "loss_video_w" in loss_overrides:
            loss_parts.append(f"video_w:{loss_overrides['loss_video_w']:.3f}")
    if "loss_gen" in train_tracker.metrics:
        loss_gen = loss_overrides.get("loss_gen", _meter_avg_or_val(train_tracker.loss_gen))
        lambda_gen = float(getattr(cfg.policy, "lambda_gen", 1.0))
        loss_parts.append(f"gen:{loss_gen:.3f}")
        loss_parts.append(f"gen_w:{lambda_gen * loss_gen:.3f}")
    if "loss_align" in train_tracker.metrics:
        loss_align = loss_overrides.get("loss_align", _meter_avg_or_val(train_tracker.loss_align))
        loss_parts.append(f"align:{loss_align:.3f}")
        loss_align_w = loss_overrides.get(
            "loss_align_w",
            _meter_avg_or_val(train_tracker.loss_align_w) if "loss_align_w" in train_tracker.metrics else loss_align,
        )
        loss_parts.append(f"bp_w:{loss_align_w:.3f}")
    for key in sorted(k for k in loss_overrides if k.startswith(("loss_bp_", "bp_"))):
        loss_parts.append(f"{key.removeprefix('loss_')}:{loss_overrides[key]:.3f}")
    if "loss_3d" in train_tracker.metrics:
        loss_3d = loss_overrides.get("loss_3d", _meter_avg_or_val(train_tracker.loss_3d))
        lambda_3d = float(getattr(cfg.policy, "lambda_3d", 1.0))
        loss_parts.append(f"3d:{loss_3d:.3f}")
        loss_3d_w = loss_overrides.get("loss_3d_w", lambda_3d * loss_3d)
        loss_parts.append(f"3d_w:{loss_3d_w:.3f}")

    optim_parts = []
    if "grad_norm" in train_tracker.metrics:
        optim_parts.append(f"grdn:{_meter_avg_or_val(train_tracker.grad_norm):.3f}")
    if "lr" in train_tracker.metrics:
        optim_parts.append(f"lr:{_meter_avg_or_val(train_tracker.lr):.1e}")

    time_parts = []
    if "update_s" in train_tracker.metrics:
        time_parts.append(f"update:{_meter_avg_or_val(train_tracker.update_s):.3f}s")
    if "dataloading_s" in train_tracker.metrics:
        time_parts.append(f"data:{_meter_avg_or_val(train_tracker.dataloading_s):.3f}s")
    if "time_3d_teacher_forward_s" in train_tracker.metrics:
        time_parts.append(f"da3:{_meter_avg_or_val(train_tracker.time_3d_teacher_forward_s):.3f}s")

    gpu_parts = []
    if "gpu_util_pct" in train_tracker.metrics:
        gpu_parts.append(f"util:{_meter_avg_or_val(train_tracker.gpu_util_pct):.0f}%")
    if "gpu_mem_allocated_gb" in train_tracker.metrics:
        gpu_parts.append(f"alloc:{_meter_avg_or_val(train_tracker.gpu_mem_allocated_gb):.2f}G")
    if "gpu_mem_peak_gb" in train_tracker.metrics:
        gpu_parts.append(f"peak:{_meter_avg_or_val(train_tracker.gpu_mem_peak_gb):.2f}G")
    if "gpu_mem_used_gb" in train_tracker.metrics:
        gpu_parts.append(f"used:{_meter_avg_or_val(train_tracker.gpu_mem_used_gb):.2f}G")

    sections = [
        f"\033[92m\033[1m{elapsed_str} << {remaining_str}\033[0m",
        f"\033[96m\033[1m{steps_per_second:.2f} iters/s\033[0m",
        f"progress[{' | '.join(progress_parts)}]",
    ]
    if loss_parts:
        sections.append(f"loss[{' | '.join(loss_parts)}]")
    if optim_parts:
        sections.append(f"optim[{' | '.join(optim_parts)}]")
    if time_parts:
        sections.append(f"time[{' | '.join(time_parts)}]")
    if gpu_parts:
        sections.append(f"gpu[{' | '.join(gpu_parts)}]")
    return " | ".join(sections)


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    global is_main_process
    # mp.set_start_method("spawn", force=True)
    cfg.validate()

    # Create Accelerator if not provided (DDP by default; FSDP opt-in via cfg.distributed_backend).
    if accelerator is None:
        accelerator = _make_training_accelerator(cfg)

    assert cfg.output_dir is not None
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    if getattr(cfg.policy, "log_sample_loss", False):
        os.environ["BPV5_SAMPLE_LOSS_DIR"] = str(cfg.output_dir)

    train_log_file = cfg.output_dir / "train.log"
    loss_log_file = cfg.output_dir / "loss.log"
    init_logging(
        log_file=train_log_file if accelerator.is_main_process else None,
        accelerator=accelerator,
    )

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info("Training log (log_freq metrics): %s", train_log_file)
        logging.info("Loss CSV: %s", loss_log_file)
        logging.info(
            "distributed_backend=%s wrap_policy=%s sharding=%s distributed_type=%s",
            cfg.distributed_backend,
            cfg.fsdp_wrap_policy,
            cfg.fsdp_sharding_strategy,
            accelerator.distributed_type,
        )
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    fastwam_worker_init_fn = None
    if cfg.seed is not None:
        if cfg.policy is not None and _is_fastwam_policy_type(cfg.policy.type):
            fastwam_module = _fastwam_policy_module(cfg.policy.type)
            set_fastwam_global_seed = import_module(
                f"lerobot.policies.{fastwam_module}.core.utils.pytorch_utils"
            ).set_global_seed

            fastwam_worker_init_fn = set_fastwam_global_seed(cfg.seed, get_worker_init_fn=True)
        else:
            set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    if cfg.policy is not None:
        cfg.policy.device = str(device)
    if _env_flag("LEROBOT_LOG_RANK_DEVICE_MAP", default=False):
        cuda_current_device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        print(
            f"[rank={accelerator.process_index:02d}/{accelerator.num_processes:02d} "
            f"local_rank={accelerator.local_process_index}] "
            f"device={device}, cuda_current_device={cuda_current_device}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
            flush=True,
        )
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    parallel_dataset_load = _env_flag("LEROBOT_PARALLEL_DATASET_LOAD", default=False)
    if parallel_dataset_load:
        if is_main_process:
            logging.info("Creating dataset on all processes in parallel")
        dataset, data_stats = make_dataset(cfg)
        accelerator.wait_for_everyone()
    else:
        # Main process downloads first to avoid race conditions in shared caches.
        if is_main_process:
            logging.info("Creating dataset")
            dataset, data_stats = make_dataset(cfg)

        accelerator.wait_for_everyone()

        if not is_main_process:
            dataset, data_stats = make_dataset(cfg)

        accelerator.wait_for_everyone()

    if accelerator.num_processes>1:
        all_data_stats = gather_object(data_stats, accelerator)
    else:
        all_data_stats = [data_stats]

    if is_main_process:
        merged_data_stats = {}
        for rank_stats in all_data_stats:
            merged_data_stats.update(rank_stats)
        data_stats = merged_data_stats
    else:
        data_stats = None

    if _is_fastwam_policy_type(cfg.policy.type):
        policy_family_label = _fastwam_family_label(cfg.policy.type)
        cfg.policy.action_norm_use_stepwise = bool(getattr(cfg.dataset, "processor_use_stepwise_action_norm", False))
        cfg.policy.action_norm_default_mode = str(getattr(cfg.dataset, "processor_norm_default_mode", "min/max"))
        fastwam_max_steps = getattr(cfg.policy, "train_max_steps", None)
        fastwam_num_epochs = getattr(cfg.policy, "train_num_epochs", None)
        if fastwam_max_steps is not None:
            cfg.steps = max(int(fastwam_max_steps), 1)
            if is_main_process:
                logging.info("%s training steps overridden by policy.train_max_steps=%d", policy_family_label, cfg.steps)
        elif fastwam_num_epochs is not None:
            sampler_processes = 1 if cfg.dataset.dist_loading else accelerator.num_processes
            effective_batch_size_for_epoch = max(cfg.batch_size * sampler_processes, 1)
            micro_steps_per_epoch = max(ceil(len(dataset) / effective_batch_size_for_epoch), 1)
            opt_steps_per_epoch = max(
                ceil(micro_steps_per_epoch / cfg.gradient_accumulation_steps),
                1,
            )
            cfg.steps = max(opt_steps_per_epoch * int(fastwam_num_epochs), 1)
            if is_main_process:
                logging.info(
                    "%s training steps derived from policy.train_num_epochs=%d: "
                    "micro_steps_per_epoch=%d, opt_steps_per_epoch=%d, total_steps=%d",
                    policy_family_label,
                    int(fastwam_num_epochs),
                    micro_steps_per_epoch,
                    opt_steps_per_epoch,
                    cfg.steps,
                )

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
    )

    if _is_fastwam_policy_type(cfg.policy.type) and hasattr(policy, "set_action_postprocess_from_stats"):
        fastwam_stats = getattr(dataset, "dataset_stats", None)
        if fastwam_stats is None and isinstance(data_stats, dict):
            fastwam_stats = data_stats.get(_fastwam_family_stats_key(cfg.policy.type))
            if fastwam_stats is None and is_tbot_sa1_wan(cfg.policy.type):
                for stats_alias in ("tbot_sa1_wan", *TBOT_SA1_WAN_LEGACY_ALIASES):
                    fastwam_stats = data_stats.get(stats_alias)
                    if fastwam_stats is not None:
                        break
            if fastwam_stats is None:
                fastwam_stats = data_stats.get("fastwam")
        if fastwam_stats is not None:
            policy.set_action_postprocess_from_stats(fastwam_stats)

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    if cfg.dataset.dist_loading and accelerator.num_processes<=1:
        raise ValueError("dist_loading is not supported when num_processes is 1")

    if cfg.dataset.dist_loading:
        num_frames = sum(gather_object(dataset.num_frames, accelerator))
        num_episodes = sum(gather_object(dataset.num_episodes, accelerator))
    else:
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes
    num_processes = accelerator.num_processes
    effective_bs = cfg.batch_size * num_processes * cfg.gradient_accumulation_steps

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"\033[91m\033[1mnum_frames={num_frames} ({format_big_number(num_frames)})\033[0m")
        logging.info(f"\033[91m\033[1mnum_episodes={num_episodes} ({format_big_number(num_episodes)})\033[0m")
        logging.info(
            "Effective batch size: "
            f"{cfg.batch_size} x {num_processes} x {cfg.gradient_accumulation_steps} = {effective_bs}"
        )
        if _is_fastwam_policy_type(cfg.policy.type):
            logging.info("policy info: %s", _format_policy_summary(policy))
            startup_summary = getattr(policy, "startup_summary", None)
            if callable(startup_summary):
                logging.info("policy startup summary:\n%s", startup_summary())
            if _env_flag("PRINT_POLICY_STRUCTURE", default=False):
                logging.info("policy structure:\n%s", policy)
        else:
            logging.info("policy info: %s", _format_policy_summary(policy))
            logging.info("\n%s", _format_freeze_summary(policy))
            if _env_flag("PRINT_POLICY_STRUCTURE", default=False):
                logging.info("policy structure:\n%s", policy)

    # create dataloader for offline training
    fastwam_train_sampler = None
    batch_sampler = None
    homogeneous_batch = bool(getattr(cfg.dataset, "homogeneous_batch", False))
    bpv5_batch_sampler = None
    if homogeneous_batch:
        if cfg.dataset.streaming:
            raise ValueError("dataset.homogeneous_batch=True requires streaming=False")
        if _is_fastwam_policy_type(cfg.policy.type):
            raise ValueError(
                "dataset.homogeneous_batch is not supported for FastWAM / TBot_SA1_Wan policies yet"
            )
        if not isinstance(dataset, MultiLeRobotDataset) and callable(
            getattr(dataset, "make_train_batch_sampler", None)
        ):
            bpv5_batch_sampler = dataset.make_train_batch_sampler(
                cfg.batch_size, 0 if cfg.seed is None else int(cfg.seed)
            )
        if not isinstance(dataset, MultiLeRobotDataset) and bpv5_batch_sampler is None:
            raise ValueError(
                "dataset.homogeneous_batch=True requires MultiLeRobotDataset "
                "or a dataset.make_train_batch_sampler()"
            )

    if _is_fastwam_policy_type(cfg.policy.type):
        fastwam_module = _fastwam_policy_module(cfg.policy.type)
        ResumableEpochSampler = import_module(
            f"lerobot.policies.{fastwam_module}.core.utils.samplers"
        ).ResumableEpochSampler

        shuffle = False
        sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=0 if cfg.seed is None else cfg.seed,
            batch_size=cfg.batch_size,
            num_processes=1 if cfg.dataset.dist_loading else accelerator.num_processes,
        )
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = fastwam_worker_init_fn
        fastwam_train_sampler = sampler
    elif homogeneous_batch and bpv5_batch_sampler is None:
        shuffle = False
        sampler = None
        batch_sampler = MultiLeRobotHomogeneousBatchSampler(
            dataset=dataset,
            batch_size=cfg.batch_size,
        )
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None
        logging.info(
            "Using MultiLeRobotHomogeneousBatchSampler (one source per batch), batch_size=%d",
            cfg.batch_size,
        )
    elif bpv5_batch_sampler is not None:
        shuffle = False
        sampler = None
        batch_sampler = bpv5_batch_sampler
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None
        logging.info(
            "Using BPV5HomogeneousBatchSampler (one leaf repo per batch), batch_size=%d",
            cfg.batch_size,
        )
    elif not cfg.dataset.streaming and hasattr(dataset, "dataset_weights") and dataset.dataset_weights is not None:
        shuffle = False
        sampler = MultiLeRobotWeightedSampler(dataset=dataset)
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None
    elif cfg.dataset.streaming:
        shuffle = False
        sampler = None
        num_workers = 1
        prefetch_factor = 4
        worker_init_fn = None
    elif callable(getattr(dataset, "make_train_sampler", None)) and (
        sampler := dataset.make_train_sampler(0 if cfg.seed is None else cfg.seed)
    ) is not None:
        # dataset-defined draw order (e.g. bpv4cdx: length-proportional, task-stratified); same order on every rank
        shuffle = False
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None
        logging.info("Using dataset sampler %s", type(sampler).__name__)
    else:
        shuffle = True
        sampler = None
        num_workers = cfg.num_workers
        prefetch_factor = 2 if cfg.num_workers > 0 else None
        worker_init_fn = None

    if batch_sampler is not None:
        # batch_sampler is mutually exclusive with batch_size / shuffle / sampler / drop_last
        dataloader_kwargs = dict(
            dataset=dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            pin_memory=device.type == "cuda",
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
        )
    else:
        dataloader_kwargs = dict(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle and not cfg.dataset.streaming,
            sampler=sampler,
            pin_memory=device.type == "cuda",
            drop_last=False,
            prefetch_factor=prefetch_factor,
            worker_init_fn=worker_init_fn,
        )
    dataloader_kwargs["persistent_workers"] = cfg.dataloader_persistent_workers and num_workers > 0
    if "in_order" not in inspect.signature(torch.utils.data.DataLoader).parameters:
        if not cfg.dataloader_in_order:
            raise RuntimeError("当前 PyTorch DataLoader 不支持 in_order=False；请升级 PyTorch")
    else:
        dataloader_kwargs["in_order"] = cfg.dataloader_in_order
    logging.info("DataLoader in_order=%s", cfg.dataloader_in_order)
    dataloader = torch.utils.data.DataLoader(**dataloader_kwargs)
    # optional dataset-defined held-out generation monitor, run after every checkpoint save (bpv4cdx HOM)
    heldout_monitor = dataset.make_heldout_monitor() if callable(getattr(dataset, "make_heldout_monitor", None)) else None

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    if cfg.distributed_backend == "fsdp":
        _configure_bpv5_fsdp_wrap(policy, accelerator, cfg.fsdp_wrap_policy)
        if is_main_process:
            logging.info(
                "Configured BPv5 FSDP wrap_policy=%s (joint pairs=%s)",
                cfg.fsdp_wrap_policy,
                cfg.fsdp_wrap_policy == "layer",
            )
    if cfg.dataset.dist_loading:
        policy, optimizer, lr_scheduler = accelerator.prepare(
            policy, optimizer, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )
    if cfg.distributed_backend == "fsdp" and accelerator.distributed_type != DistributedType.FSDP:
        raise RuntimeError(
            f"Requested fsdp but after prepare distributed_type is {accelerator.distributed_type}"
        )

    if getattr(cfg.policy, "preflight_memory_check", False) and torch.cuda.is_available() and not cfg.resume:
        from lerobot.policies.BPV4BPEn.preflight import run_preflight

        run_preflight(
            policy,
            optimizer,
            dataloader,
            accelerator,
            limit_frac=float(getattr(cfg.policy, "preflight_memory_limit_frac", 0.9)),
        )

    eval_dataloader = None
    if is_tbot_sa1_wan(cfg.policy.type) and cfg.eval_freq > 0:
        if is_main_process:
            from lerobot.policies.TBot_SA1_Wan.dataset_tbot_sa1_wan import build_tbot_sa1_wan_dataset

            stats_cache_path = cfg.dataset.normalization_stats_path
            if stats_cache_path is None and cfg.output_dir is not None:
                stats_cache_path = str(Path(cfg.output_dir) / _fastwam_family_stats_filename(cfg.policy.type))

            eval_dataset = build_tbot_sa1_wan_dataset(
                cfg.dataset,
                stats_cache_path=stats_cache_path,
                is_training_set=False,
            )
            eval_num_workers = max(_env_int("LEROBOT_TBOT_SA1_WAN_EVAL_NUM_WORKERS", 0), 0)
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                num_workers=eval_num_workers,
                batch_size=cfg.batch_size,
                shuffle=False,
                sampler=None,
                pin_memory=device.type == "cuda" and eval_num_workers > 0,
                drop_last=False,
                prefetch_factor=2 if eval_num_workers > 0 else None,
                worker_init_fn=None,
            )
            logging.info(
                "Created TBot_SA1_Wan eval dataloader for periodic validation: "
                "eval_freq=%d, max_batches=%d, batch_size=%d, num_workers=%d, metric=action_mse/l2",
                cfg.eval_freq,
                cfg.eval_max_batches,
                cfg.batch_size,
                eval_num_workers,
            )
        accelerator.wait_for_everyone()

    fastwam_epoch = 0
    fastwam_batch_in_epoch = 0
    fastwam_epoch_offset = 0
    if cfg.resume and _is_fastwam_policy_type(cfg.policy.type) and cfg.checkpoint_path is not None:
        fastwam_state_path = Path(cfg.checkpoint_path) / "training_state" / _fastwam_family_trainer_state_file(
            cfg.policy.type
        )
        if fastwam_state_path.is_file():
            fastwam_resume_state = load_json(fastwam_state_path)
            fastwam_epoch = int(fastwam_resume_state.get("epoch", 0))
            fastwam_batch_in_epoch = int(fastwam_resume_state.get("batch_in_epoch", 0))
            fastwam_epoch_offset = fastwam_epoch
            if fastwam_train_sampler is not None:
                fastwam_train_sampler.set_epoch_offset(fastwam_epoch)
                fastwam_train_sampler.set_resume_batch_offset(fastwam_batch_in_epoch)
            if is_main_process:
                logging.info(
                    "Restored %s dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    _fastwam_family_label(cfg.policy.type),
                    fastwam_epoch,
                    fastwam_batch_in_epoch,
                    fastwam_batch_in_epoch
                    * cfg.batch_size
                    * (1 if cfg.dataset.dist_loading else accelerator.num_processes),
                )
    if _is_fastwam_policy_type(cfg.policy.type):
        dl_iter = iter(dataloader)
    else:
        dl_iter = cycle(dataloader)

    policy.train()

    if _is_fastwam_policy_type(cfg.policy.type):
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "loss_video": AverageMeter("loss_video", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        if is_tbot_sa1_wan(cfg.policy.type):
            train_metrics["loss_3d"] = AverageMeter("loss_3d", ":.3f")
            if getattr(cfg.policy, "log_da3_teacher_timing", False):
                train_metrics["time_3d_teacher_forward_s"] = AverageMeter("da3_s", ":.3f")
    elif cfg.policy.type == "qwenaction":
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
    elif cfg.policy.type in ["a1", "qwena1"] or is_tbot_sa1(cfg.policy.type):
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("loss_action", ":.3f"),
            "loss_gen": AverageMeter("loss_gen", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        if is_tbot_sa1(cfg.policy.type):
            train_metrics["loss_3d"] = AverageMeter("loss_3d", ":.3f")
            if getattr(cfg.policy, "log_da3_teacher_timing", False):
                train_metrics["time_3d_teacher_forward_s"] = AverageMeter("da3_s", ":.3f")
    elif is_bpvav3(cfg.policy.type):
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "loss_action": AverageMeter("action", ":.3f"),
            "loss_align": AverageMeter("align", ":.3f"),
            "loss_align_w": AverageMeter("bp_w", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
    else:
        train_metrics = {
            "loss": AverageMeter("loss", ":.3f"),
            "grad_norm": AverageMeter("grdn", ":.3f"),
            "lr": AverageMeter("lr", ":0.1e"),
            "update_s": AverageMeter("updt_s", ":.3f"),
            "dataloading_s": AverageMeter("data_s", ":.3f"),
        }
        if cfg.policy.type == "bpv5_qformer":
            # data_s = mean worker __getitem__ (cache + frames + transform).
            # wait_s = main-process next() stall; the rest are the same sample, split by stage.
            train_metrics["data_wait_s"] = AverageMeter("wait_s", ":.3f")
            train_metrics["data_max_s"] = AverageMeter("data_max_s", ":.3f")
            train_metrics["data_cache_s"] = AverageMeter("cache_s", ":.3f")
            train_metrics["data_frames_s"] = AverageMeter("frames_s", ":.3f")
            train_metrics["data_xform_s"] = AverageMeter("xform_s", ":.3f")

    if cfg.gpu_stats_freq > 0:
        _maybe_add_gpu_metrics(train_metrics)

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    train_tracker = MetricsTracker(
        effective_batch_size,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")
        training_start_time = time.perf_counter()

    train_pbar = None
    if is_main_process:
        train_pbar = tqdm(
            total=cfg.steps,
            initial=step,
            desc="训练",
            unit="step",
            dynamic_ncols=True,
        )
    
    accumulated_update_time = 0.0
    accumulated_dataloading_time = 0.0
    accumulated_wait_time = 0.0
    accumulated_timing: dict[str, float] = {}
    timing_batches = 0
    while step < cfg.steps:
        start_time = time.perf_counter()
        if _is_fastwam_policy_type(cfg.policy.type):
            try:
                batch = next(dl_iter)
                fastwam_batch_in_epoch += 1
            except StopIteration:
                fastwam_epoch += 1
                fastwam_batch_in_epoch = 0
                if fastwam_train_sampler is not None:
                    fastwam_train_sampler.clear_resume_batch_offset()
                    fastwam_train_sampler.set_epoch(fastwam_epoch - fastwam_epoch_offset)
                dl_iter = iter(dataloader)
                batch = next(dl_iter)
                fastwam_batch_in_epoch += 1
        else:
            batch = next(dl_iter)
        wait_s = time.perf_counter() - start_time
        worker_timing = _pop_worker_timing(batch)
        if cfg.dataset.dist_loading:
            h2d_t0 = time.perf_counter()
            batch = send_to_device(batch, accelerator.device, non_blocking=True)
            wait_s += time.perf_counter() - h2d_t0
        accumulated_wait_time += wait_s
        if worker_timing is not None:
            accumulated_dataloading_time += worker_timing["getitem_s"]
            timing_batches += 1
            for key in ("getitem_max_s", "cache_s", "frames_s", "xform_s"):
                accumulated_timing[key] = accumulated_timing.get(key, 0.0) + worker_timing.get(key, 0.0)
        else:
            accumulated_dataloading_time += wait_s

        will_log_after_update = cfg.log_freq > 0 and (step + 1) % cfg.log_freq == 0
        policy_forward_kwargs = {}
        if is_tbot_sa1_wan(cfg.policy.type):
            policy_forward_kwargs["collect_metrics"] = will_log_after_update

        train_tracker, output_dict, did_step, update_time_s = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            policy_forward_kwargs=policy_forward_kwargs,
            use_zero_grad_set_to_none=_is_fastwam_policy_type(cfg.policy.type),
            skip_scheduler_when_optimizer_step_skipped=_is_fastwam_policy_type(cfg.policy.type),
            clip_optimizer_params_only=is_tbot_sa1_wan(cfg.policy.type),
        )
        accumulated_update_time += update_time_s

        if not did_step:
            continue

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        n_micro = max(timing_batches, 1)
        train_tracker.dataloading_s = accumulated_dataloading_time / n_micro
        train_tracker.update_s = accumulated_update_time
        if "data_wait_s" in train_tracker.metrics:
            train_tracker.data_wait_s = accumulated_wait_time
            if timing_batches:
                train_tracker.data_max_s = accumulated_timing.get("getitem_max_s", 0.0) / timing_batches
                train_tracker.data_cache_s = accumulated_timing.get("cache_s", 0.0) / timing_batches
                train_tracker.data_frames_s = accumulated_timing.get("frames_s", 0.0) / timing_batches
                train_tracker.data_xform_s = accumulated_timing.get("xform_s", 0.0) / timing_batches
        accumulated_dataloading_time = 0.0
        accumulated_update_time = 0.0
        accumulated_wait_time = 0.0
        accumulated_timing = {}
        timing_batches = 0
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        should_log = is_log_step and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if (
            is_main_process
            and cfg.gpu_stats_freq > 0
            and step % cfg.gpu_stats_freq == 0
            and torch.cuda.is_available()
        ):
            _update_gpu_metrics(train_tracker, accelerator.device, accelerator.local_process_index)
        if is_main_process and train_pbar is not None:
            train_pbar.update(1)
            train_pbar.set_postfix(train_tracker.to_postfix(use_avg=is_log_step), refresh=False)

        synced_log_loss_dict = None
        if is_log_step:
            log_scalar_metrics = {
                name: value
                for name, value in {
                    "loss": _meter_avg_or_val(train_tracker.loss) if "loss" in train_tracker.metrics else None,
                    "loss_action": _meter_avg_or_val(train_tracker.loss_action) if "loss_action" in train_tracker.metrics else None,
                    "loss_align": _meter_avg_or_val(train_tracker.loss_align) if "loss_align" in train_tracker.metrics else None,
                    "loss_align_w": _meter_avg_or_val(train_tracker.loss_align_w) if "loss_align_w" in train_tracker.metrics else None,
                    "loss_video": _meter_avg_or_val(train_tracker.loss_video) if "loss_video" in train_tracker.metrics else None,
                    "loss_gen": _meter_avg_or_val(train_tracker.loss_gen) if "loss_gen" in train_tracker.metrics else None,
                    "loss_3d": _meter_avg_or_val(train_tracker.loss_3d) if "loss_3d" in train_tracker.metrics else None,
                }.items()
                if value is not None
            }
            if is_tbot_sa1_wan(cfg.policy.type):
                lambda_action = float(getattr(cfg.policy, "lambda_action", 1.0))
                lambda_video = float(getattr(cfg.policy, "lambda_video", 1.0))
                lambda_3d = float(getattr(cfg.policy, "lambda_3d", 1.0))
                if "loss_action" in log_scalar_metrics:
                    log_scalar_metrics["loss_action_w"] = lambda_action * log_scalar_metrics["loss_action"]
                if "loss_video" in log_scalar_metrics:
                    log_scalar_metrics["loss_video_w"] = lambda_video * log_scalar_metrics["loss_video"]
                if "loss_3d" in log_scalar_metrics:
                    log_scalar_metrics["loss_3d_w"] = lambda_3d * log_scalar_metrics["loss_3d"]
            log_scalar_metrics.update(
                {
                    key: value
                    for key, value in output_dict.items()
                    if key.startswith(("loss_action_dim", "loss_bp_", "bp_"))
                    and isinstance(value, (int, float))
                }
            )
            synced_log_loss_dict = _sync_scalar_metrics(
                accelerator,
                log_scalar_metrics,
            )

        if should_log:
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            avg_update_time = train_tracker.update_s.avg if hasattr(train_tracker.update_s, "avg") else train_tracker.update_s.val
            steps_per_second = 1.0 / avg_update_time if avg_update_time > 0 else 0
            elapsed_time = time.perf_counter() - training_start_time if training_start_time else 0
            remaining_steps = cfg.steps - step
            estimated_remaining_time = remaining_steps * avg_update_time if avg_update_time > 0 else 0
            elapsed_str = format_time(elapsed_time)
            remaining_str = format_time(estimated_remaining_time)
            status_line = _format_train_status_line(
                train_tracker,
                cfg,
                elapsed_str=elapsed_str,
                remaining_str=remaining_str,
                steps_per_second=steps_per_second,
                loss_overrides=synced_log_loss_dict,
            )
            with open(train_log_file, "a", encoding="utf-8") as log_f:
                log_f.write(f"INFO {dt} lerobot_train.py:step {status_line}\n")
                log_f.flush()
            append_loss_log(loss_log_file, train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if synced_log_loss_dict:
                    wandb_log_dict.update(synced_log_loss_dict)
                if output_dict:
                    for key, value in output_dict.items():
                        if key in wandb_log_dict:
                            continue
                        wandb_log_dict[key] = value
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        should_eval = is_tbot_sa1_wan(cfg.policy.type) and cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        if should_eval:
            accelerator.wait_for_everyone()
            if is_main_process and eval_dataloader is not None:
                eval_policy = accelerator.unwrap_model(policy)
                eval_metrics = evaluate_tbot_sa1_wan_action_policy(
                    eval_policy,
                    eval_dataloader,
                    accelerator,
                    max_batches=max(int(cfg.eval_max_batches), 1),
                )
                summary_keys = ["action_mse_loss", "action_l2_error", "action_mse_std", "action_l2_std"]
                summary_parts = [
                    f"{key}={eval_metrics[key]:.4f}"
                    for key in summary_keys
                    if key in eval_metrics
                ]
                summary_parts.append(f"batches={int(eval_metrics['num_batches'])}")
                summary_parts.append(f"samples={int(eval_metrics['num_samples'])}")
                logging.info("TBot_SA1_Wan eval @ step %d: %s", step, ", ".join(summary_parts))
                if wandb_logger:
                    wandb_logger.log_dict(eval_metrics, step, mode="eval")
            accelerator.wait_for_everyone()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                msg = f"Checkpoint saved after step {step}"
                if train_pbar is not None:
                    tqdm.write(msg)
                else:
                    logging.info(msg)
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                logging.info(colored("Checkpoint saved at:", "cyan", attrs=["bold"]) + f" {checkpoint_dir}")
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    data_stats=data_stats, 
                )
                if _is_fastwam_policy_type(cfg.policy.type):
                    write_json(
                        {
                            "epoch": fastwam_epoch,
                            "batch_in_epoch": fastwam_batch_in_epoch,
                        },
                        checkpoint_dir / "training_state" / _fastwam_family_trainer_state_file(cfg.policy.type),
                    )
                update_last_checkpoint(checkpoint_dir)
                keep_states = int(os.environ.get("BPV4CDX_KEEP_TRAINING_STATES", "0"))
                if keep_states > 0:
                    older = sorted(p for p in checkpoint_dir.parent.iterdir() if p.name.isdigit())[:-keep_states]
                    for p in older:
                        shutil.rmtree(p / "training_state", ignore_errors=True)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if is_saving_step and heldout_monitor is not None:
            # dataset-defined held-out generation error (BP / BPv5 HOM); shard or main-only depending on monitor.
            # Under FSDP, pass the wrapped policy so all ranks can participate / summon params.
            hom_policy = (
                policy
                if cfg.distributed_backend == "fsdp"
                else accelerator.unwrap_model(policy)
            )
            hom = heldout_monitor(hom_policy, accelerator, step)
            if is_main_process and hom:
                mod_name = type(heldout_monitor).__module__
                if mod_name.endswith(("BPV5QFormer.heldout_monitor", "BPV5StageB.heldout_monitor")):
                    import importlib

                    _hom = importlib.import_module(mod_name)
                    log_line, trend_advice, write_g9_json = _hom.log_line, _hom.trend_advice, _hom.write_g9_json
                else:
                    from lerobot.policies.BPV4CDX.heldout_monitor import log_line, trend_advice

                    write_g9_json = None

                hom_file = cfg.output_dir / "heldout_monitor.jsonl"
                history = [json.loads(line) for line in hom_file.read_text().splitlines()] if hom_file.is_file() else []
                decay = int(getattr(cfg.policy, "scheduler_decay_steps", 0) or 0)
                hom["advice"] = trend_advice(history, hom, lr_at_floor=bool(decay and step >= decay))
                logging.info("HOM @ step %d: %s", step, log_line(hom))
                with open(hom_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(hom) + "\n")
                if write_g9_json is not None and bool(getattr(cfg.dataset, "heldout_monitor_write_g9_json", False)):
                    ckpt_dir = cfg.output_dir / "checkpoints" / f"{step:06d}" / "pretrained_model"
                    write_g9_json(cfg.output_dir, hom, step, ckpt=str(ckpt_dir.resolve()))
                if wandb_logger:
                    flat = {}
                    for k, v in hom.items():
                        if isinstance(v, (int, float)):
                            flat[k] = v
                        elif isinstance(v, dict) and isinstance(v.get("ratio"), (int, float)):
                            flat[k] = v["ratio"]
                    wandb_logger.log_dict(flat, step, mode="eval")
            # Stage B gated C1: arm when blank/match is below threshold;
            # fix m = max(mse_blank, scale * mse_matched) so hinge is active on this ckpt.
            gate_thr = float(getattr(cfg.policy, "blank_margin_gate_blank_over_match", 0.0) or 0.0)
            m_scale = float(getattr(cfg.policy, "blank_margin_m_hom_matched_scale", 0.0) or 0.0)
            if gate_thr > 0 or m_scale > 0:
                armed_f = 1.0 if gate_thr <= 0 else 0.0
                m_fixed_f = float(getattr(cfg.policy, "blank_margin_m", 0.05) or 0.05)
                if is_main_process and hom:
                    bom = hom.get("blank_over_match")
                    blank_r = bom.get("ratio") if isinstance(bom, dict) else bom
                    mse_m = hom.get("mse_matched")
                    mse_b = hom.get("mse_blank")
                    if gate_thr > 0:
                        armed_f = (
                            1.0
                            if blank_r is not None and float(blank_r) < gate_thr
                            else 0.0
                        )
                    if m_scale > 0 and mse_m is not None:
                        m_from_match = float(m_scale) * float(mse_m)
                        m_from_blank = float(mse_b) if mse_b is not None else m_from_match
                        m_fixed_f = max(m_from_blank, m_from_match)
                    logging.info(
                        "C1 gate @ step %d: blank/match=%s blank=%.4f match=%.4f thr=%s armed=%s "
                        "m_fixed=%.4f (=max(blank, %.3g*match))",
                        step,
                        blank_r,
                        float(mse_b) if mse_b is not None else float("nan"),
                        float(mse_m) if mse_m is not None else float("nan"),
                        gate_thr,
                        bool(armed_f > 0.5),
                        m_fixed_f,
                        m_scale,
                    )
                payload = torch.tensor(
                    [armed_f, m_fixed_f],
                    device=accelerator.device,
                    dtype=torch.float64,
                )
                if accelerator.num_processes > 1:
                    import torch.distributed as dist

                    dist.broadcast(payload, src=0)
                unwrapped = accelerator.unwrap_model(policy)
                if gate_thr > 0:
                    unwrapped._blank_margin_armed = bool(payload[0].item() > 0.5)
                if m_scale > 0:
                    unwrapped._blank_margin_m_fixed = float(payload[1].item())
            accelerator.wait_for_everyone()
    if is_main_process and train_pbar is not None:
        train_pbar.close()

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()

import traceback

from lerobot.scripts.send_error_to_dingtalk import send_error_to_dingtalk

def main():
    try:
        register_third_party_plugins()
        train()
    except Exception as e:
        if _is_local_main_process():
            error_info = traceback.format_exc()
            error_info = "代码累了，它想歇会：\n" + error_info
            # send_error_to_dingtalk(error_info)
        else:
            # logging.error("非主进程，不发送错误信息 + 1")
            pass
        raise e


if __name__ == "__main__":
    main()
