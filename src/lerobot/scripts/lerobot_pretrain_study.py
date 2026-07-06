#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
lerobot_pretrain_study.py — TBot-SA1 **预训练**训练循环学习版
============================================================

本文件由 ``lerobot_train.py``（约 1200 行）精简而来，**仅保留 TBot-SA1 预训练主路径**，
并附中文注释，便于代码分享与阅读。

## 如何启动（与正式训练相同，仅换模块名）

.. code-block:: bash

    cd /home/jovyan/vla/workspace/mytbot
    source /home/jovyan/.conda/envs/tbot/bin/activate
    export PYTHONPATH="${PWD}/src"

    accelerate launch --num_processes=4 \\
      -m lerobot.scripts.lerobot_pretrain_study \\
      --config_path=.config/pretrain_config.jsonc

## 预训练主路径（本文件保留）

::

    pretrain_config.jsonc
        → TrainPipelineConfig.validate()
        → make_dataset()          # 多数据集 + weight_rules + dist_loading
        → make_policy(TBot_SA1)
        → DataLoader + MultiLeRobotWeightedSampler
        → while step < steps:
              batch → update_policy() → policy.forward()  # 见 modeling_tbot_sa1.py
              log / save_checkpoint

## 已从 lerobot_train.py **刻意删除**的分支（学习时可跳过原文对应段落）

+---------------------------+------------------------------------------+
| 分支                      | 原文大致位置 / 用途                       |
+===========================+==========================================+
| FastWAM / TBot_SA1_Wan    | L71–104, L641–651, L701–729, L794–812…   |
|                           | 6B 视频 WSA 变体，独立 sampler / eval     |
| qwenaction / a1 / qwena1  | L937–945                                 |
|                           | 其他对比策略的 metrics 定义               |
| streaming 数据集          | L819–824                                 |
|                           | 流式读取，预训练不用                      |
| shuffle=True 普通 DL      | L825–830                                 |
|                           | 单数据集微调常用；预训练用加权采样器      |
| evaluate_tbot_sa1_wan_*   | L200–296, L855–889, L1129–1151           |
|                           | Wan 训练中 periodic action eval           |
| FastWAM epoch 迭代器      | L1000–1012, L916–919                     |
|                           | 按 epoch 手动 StopIteration；TBot 用 cycle |
| push_to_hub               | L1191–1193                               |
|                           | 训练结束上传 HF，预训练本地为主           |
| LEROBOT_PARALLEL_DATASET  | L669–674                                 |
|                           | 并行建 dataset，默认主进程先下载即可      |
| LEROBOT_LOG_RANK_DEVICE   | L657–665                                 |
|                           | 调试多卡 device 映射                      |
+---------------------------+------------------------------------------+

## 下一步阅读（按顺序）

1. ``lerobot/configs/train.py`` — ``TrainPipelineConfig`` 字段含义
2. ``lerobot/datasets/factory.py`` — ``make_dataset()`` 如何把 LeRobot v3 变成 Dataset
3. ``lerobot/policies/TBot_SA1/configuration_tbot_sa1.py`` — 预训练超参
4. ``lerobot/policies/TBot_SA1/modeling_tbot_sa1.py`` — ``TBotSA1Policy.forward`` / 三 loss

对应正式源码行号对照见各函数 docstring 中的「原文」标注。
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from datetime import datetime, timedelta
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from accelerate.utils import send_to_device
from termcolor import colored
from torch.optim import Optimizer
from tqdm import tqdm

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import MultiLeRobotWeightedSampler
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, append_loss_log, format_time
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import gather_object, init_logging


# ---------------------------------------------------------------------------
# 单步训练：forward → backward → clip → optimizer.step → scheduler.step
# 原文：lerobot_train.py L299–397
# ---------------------------------------------------------------------------
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: dict[str, Any],
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict, bool, float]:
    """执行一次梯度更新（TBot-SA1 预训练标准路径）。"""
    start_time = time.perf_counter()
    policy.train()

    with accelerator.accumulate(policy):
        # 混合精度由 Accelerate 自动处理；loss 在 TBotSA1Policy.forward 内汇总
        with accelerator.autocast():
            loss, output_dict = policy.forward(batch)

        accelerator.backward(loss)

        grad_norm = None
        if accelerator.sync_gradients:
            if grad_clip_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), float("inf"), error_if_nonfinite=False
                )

        with nullcontext():
            optimizer.step()

        # step_scheduler_with_optimizer=False，仅在真实 optimizer step 时推进 scheduler
        if lr_scheduler is not None and accelerator.sync_gradients:
            lr_scheduler.step()

        optimizer.zero_grad()

    # 记录标量，供 log_freq 打印；TBot 预训练关注 action / gen / 3d 分项
    train_metrics.loss = loss.item()
    for name in ("loss_action", "loss_gen", "loss_3d", "time_3d_teacher_forward_s"):
        if name in output_dict and name in train_metrics.metrics:
            val = output_dict[name]
            setattr(train_metrics, name, float(val.detach().item()) if isinstance(val, torch.Tensor) else float(val))

    if accelerator.sync_gradients and grad_norm is not None:
        train_metrics.grad_norm = grad_norm.item()
        train_metrics.lr = optimizer.param_groups[0]["lr"]

    return train_metrics, output_dict, accelerator.sync_gradients, time.perf_counter() - start_time


def _meter_avg(meter: AverageMeter) -> float:
    return meter.avg if meter.count > 0 else meter.val


def _sync_log_metrics(accelerator: Accelerator, metrics: dict[str, float]) -> dict[str, float]:
    """多卡训练时对 loss 标量做 all-reduce 平均。原文 L400–415。"""
    if accelerator.num_processes <= 1:
        return metrics
    names = list(metrics.keys())
    tensor = torch.tensor([metrics[n] for n in names], device=accelerator.device, dtype=torch.float32)
    reduced = accelerator.reduce(tensor, reduction="mean")
    return {n: float(v) for n, v in zip(names, reduced.detach().cpu().tolist(), strict=False)}


def _format_status_line(
    tracker: MetricsTracker,
    cfg: TrainPipelineConfig,
    *,
    elapsed_str: str,
    remaining_str: str,
    steps_per_second: float,
) -> str:
    """精简版训练状态行（仅 TBot-SA1 三 loss）。原文 L490–570 的子集。"""
    parts = [
        f"\033[92m\033[1m{elapsed_str} << {remaining_str}\033[0m",
        f"\033[96m\033[1m{steps_per_second:.2f} iters/s\033[0m",
        f"step:{tracker.steps}",
    ]
    if "loss" in tracker.metrics:
        parts.append(f"total:{_meter_avg(tracker.loss):.3f}")
    if "loss_action" in tracker.metrics:
        parts.append(f"action:{_meter_avg(tracker.loss_action):.3f}")
    if "loss_gen" in tracker.metrics:
        lg = _meter_avg(tracker.loss_gen)
        parts.append(f"gen:{lg:.3f}(w:{float(cfg.policy.lambda_gen) * lg:.3f})")
    if "loss_3d" in tracker.metrics:
        l3 = _meter_avg(tracker.loss_3d)
        parts.append(f"3d:{l3:.3f}(w:{float(cfg.policy.lambda_3d) * l3:.3f})")
    if "grad_norm" in tracker.metrics:
        parts.append(f"grdn:{_meter_avg(tracker.grad_norm):.3f}")
    if "lr" in tracker.metrics:
        parts.append(f"lr:{_meter_avg(tracker.lr):.1e}")
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# 预训练主函数
# 原文：lerobot_train.py L574–1197（已删 Wan/FastWAM/对比策略/streaming 等分支）
# ---------------------------------------------------------------------------
@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    TBot-SA1 预训练编排入口。

    职责划分（分享时可画白板）：
    - **本文件**：数据加载、分布式、优化器、日志、checkpoint
    - **TBotSA1Policy.forward**：batch → 三 loss（见 modeling_tbot_sa1.py）
    """
    cfg.validate()

    # --- 1. Accelerate：多卡 DDP + 梯度累积 ---
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=1800))
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs, init_pg_kwargs],
        )

    assert cfg.output_dir is not None
    if accelerator.is_main_process:
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    is_main = accelerator.is_main_process
    train_log = cfg.output_dir / "train.log"
    loss_log = cfg.output_dir / "loss.log"
    init_logging(log_file=train_log if is_main else None, accelerator=accelerator)

    if is_main:
        logging.info("【学习版】lerobot_pretrain_study.py — 仅 TBot-SA1 预训练路径")
        logging.info("训练日志: %s", train_log)
        logging.info("Loss CSV: %s", loss_log)
        logging.info(pformat(cfg.to_dict()))

    # --- 2. 随机种子（预训练：标准 set_seed，非 FastWAM 专用 worker_init）---
    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    if cfg.policy is not None:
        cfg.policy.device = str(device)

    # --- 3. 构建数据集 ---
    # 预训练典型配置：repo_id_file + weight_rules_path + dist_loading=true
    # 主进程先 make_dataset，避免 HF cache / 本地路径竞态（原文 L676–686）
    if is_main:
        logging.info("Creating dataset (main process first)")
        dataset, data_stats = make_dataset(cfg)
    accelerator.wait_for_everyone()
    if not is_main:
        dataset, data_stats = make_dataset(cfg)
    accelerator.wait_for_everyone()

    # 多卡各自有一份 stats，主进程 merge（原文 L688–699）
    if accelerator.num_processes > 1:
        all_stats = gather_object(data_stats, accelerator)
    else:
        all_stats = [data_stats]
    if is_main:
        merged = {}
        for rank_stats in all_stats:
            merged.update(rank_stats)
        data_stats = merged
    else:
        data_stats = None

    # --- 4. 构建策略 TBot_SA1 ---
    if is_main:
        logging.info("Creating policy (TBot_SA1)")
    policy = make_policy(cfg=cfg.policy)
    accelerator.wait_for_everyone()

    # --- 5. 优化器 & 学习率调度（来自 TBotSA1Config preset）---
    if is_main:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # dist_loading：每个 rank 自己迭代 dataset，batch 需 send_to_device（原文 L763–764）
    if cfg.dataset.dist_loading and accelerator.num_processes <= 1:
        raise ValueError("dist_loading 需要 num_processes > 1")

    if cfg.dataset.dist_loading:
        num_frames = sum(gather_object(dataset.num_frames, accelerator))
        num_episodes = sum(gather_object(dataset.num_episodes, accelerator))
    else:
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes

    effective_bs = cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    if is_main:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info("steps=%s, num_frames=%s, effective_batch_size=%s", cfg.steps, num_frames, effective_bs)
        logging.info("policy:\n%s", policy)

    # --- 6. DataLoader：预训练用 MultiLeRobotWeightedSampler ---
    # 【已删分支】streaming / shuffle=True / FastWAM ResumableEpochSampler
    if hasattr(dataset, "dataset_weights") and dataset.dataset_weights is not None:
        sampler = MultiLeRobotWeightedSampler(dataset=dataset)
        shuffle = False
    else:
        # 单数据集预训练实验时 fallback
        sampler = None
        shuffle = True

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # dist_loading 时不 prepare dataloader（各 rank 独立读）
    accelerator.wait_for_everyone()
    if cfg.dataset.dist_loading:
        policy, optimizer, lr_scheduler = accelerator.prepare(policy, optimizer, lr_scheduler)
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    dl_iter = cycle(dataloader)  # TBot 预训练：无限循环迭代（原文 L919）
    policy.train()

    # --- 7. TBot-SA1 专用 metrics（原文 L946–959 分支）---
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "loss_action": AverageMeter("loss_action", ":.3f"),
        "loss_gen": AverageMeter("loss_gen", ":.3f"),
        "loss_3d": AverageMeter("loss_3d", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    if getattr(cfg.policy, "log_da3_teacher_timing", False):
        train_metrics["time_3d_teacher_forward_s"] = AverageMeter("da3_s", ":.3f")

    tracker = MetricsTracker(
        effective_bs,
        num_frames,
        num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main:
        logging.info("Start TBot-SA1 pretraining loop")
        t0 = time.perf_counter()

    pbar = tqdm(total=cfg.steps, initial=step, desc="预训练", unit="step") if is_main else None
    acc_data_time = 0.0
    acc_update_time = 0.0

    # --- 8. 主训练循环 ---
    while step < cfg.steps:
        t_data = time.perf_counter()
        batch = next(dl_iter)
        if cfg.dataset.dist_loading:
            batch = send_to_device(batch, accelerator.device, non_blocking=True)
        acc_data_time += time.perf_counter() - t_data

        tracker, output_dict, did_step, update_s = update_policy(
            tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
        )
        acc_update_time += update_s

        if not did_step:
            continue  # 梯度累积中间 micro-step，不 increment step

        step += 1
        tracker.dataloading_s = acc_data_time
        tracker.update_s = acc_update_time
        acc_data_time = 0.0
        acc_update_time = 0.0
        tracker.step()

        is_log = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_save = cfg.save_checkpoint and (step % cfg.save_freq == 0 or step == cfg.steps)

        if is_main and pbar is not None:
            pbar.update(1)
            if is_log:
                pbar.set_postfix(tracker.to_postfix(use_avg=True), refresh=False)

        if is_log and is_main:
            avg_upd = _meter_avg(tracker.update_s)
            sps = 1.0 / avg_upd if avg_upd > 0 else 0.0
            elapsed = time.perf_counter() - t0
            remaining = (cfg.steps - step) * avg_upd if avg_upd > 0 else 0.0
            line = _format_status_line(
                tracker,
                cfg,
                elapsed_str=format_time(elapsed),
                remaining_str=format_time(remaining),
                steps_per_second=sps,
            )
            logging.info(line)
            with open(train_log, "a", encoding="utf-8") as f:
                f.write(f"INFO {datetime.now():%Y-%m-%d %H:%M:%S} {line}\n")
            append_loss_log(loss_log, tracker)
            tracker.reset_averages()

        if is_save:
            if is_main:
                ckpt_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                logging.info(colored("Checkpoint saved:", "cyan", attrs=["bold"]) + f" {ckpt_dir}")
                save_checkpoint(
                    checkpoint_dir=ckpt_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    data_stats=data_stats,
                )
                update_last_checkpoint(ckpt_dir)
            accelerator.wait_for_everyone()

    if is_main and pbar is not None:
        pbar.close()
        logging.info("End of pretraining")

    accelerator.wait_for_everyone()
    accelerator.end_training()

import traceback
from lerobot.scripts.send_error_to_dingtalk import send_error_to_dingtalk

def main():
    try:
        register_third_party_plugins()
        train()
    except Exception as e:
        error_info = traceback.format_exc()
        error_info = "别搞啊，又双叒叕报错了：\n" + error_info
        send_error_to_dingtalk(error_info)
        raise e


if __name__ == "__main__":
    main()
