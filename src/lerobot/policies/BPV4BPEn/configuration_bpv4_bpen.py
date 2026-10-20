"""BPv4 第一阶段：以整条轨迹的关键帧图像为 BP，给定进度 p 和 p 时刻 state，预测之后的 action。

这不是完整的 v4 模型，只用于训练和检验 BP encoder。模块命名（bp_obs_encoder / bp_resampler）与将来的完整 v4 保持一致，
以便直接加载权重。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import ClassVar

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from lerobot.policies.q35.configuration_q35 import Q35Config

BPV4_BPEN = "bpv4_bpen"
_DEFAULT_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


@DatasetConfig.register_subclass(BPV4_BPEN)
@dataclass
class BPV4BPEnDatasetConfig(DatasetConfig):
    """每个数据项是一条轨迹：缓存的关键帧 ViT 特征 + 该轨迹上的 S 个查询（p、state、action chunk、遮挡方案）。"""

    _canonical_type: ClassVar[str] = BPV4_BPEN
    action_mode: str = "delta"
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_5_processor_path: str = "/vla/workspace/models/q35/base_800m"

    use_language: bool = True
    language_drop_prob: float = 0.2
    language_max_length: int = 48

    # 关键帧采样 v2（见 keyframes.py），与特征缓存一一对应。
    bp_num_keyframes: int = 32
    bp_keyframe_alpha: float = 0.5
    bp_camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERAS))
    # Qwen3.5-0.8B base 原始 ViT 提取（不含任何 VLA 权重）。
    bp_feature_cache_dir: str = "/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2"

    queries_per_trajectory: int = 4
    # 0：不丢弃。主实验 0.1，对照组 0。
    progress_drop_prob: float = 0.1

    # 遮挡方案在数据侧生成（事件整段 + 比例窗口各一份），是否遮挡、用哪一种由模型按训练步数决定。
    mask_window_ratio: tuple[float, float] = (0.10, 0.30)
    mask_min_visible_keyframes: int = 8

    # 每个 repo 末尾留出的验证 episode 数：按 repo 路径中的子串匹配。
    val_holdout_last: dict[str, int] = field(
        default_factory=lambda: {"clean_50": 5, "randomized_100": 10}
    )
    # train / val / all：离线验证工具用 val。
    split: str = "train"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if not 0.0 <= self.progress_drop_prob <= 1.0 or not 0.0 <= self.language_drop_prob <= 1.0:
            raise ValueError("drop probabilities must be in [0, 1]")
        lo, hi = self.mask_window_ratio
        if not 0.0 < lo <= hi <= 1.0:
            raise ValueError("mask_window_ratio must satisfy 0 < lo <= hi <= 1")
        if self.queries_per_trajectory <= 0 or self.bp_num_keyframes <= 0:
            raise ValueError("queries_per_trajectory and bp_num_keyframes must be positive")


@LRSchedulerConfig.register_subclass("bpv4_bpen_grouped")
@dataclass
class BPV4GroupedSchedulerConfig(LRSchedulerConfig):
    """所有参数组共用 warmup + cosine；参数组可带 lr_zero_steps / lr_ramp_steps（先为 0 再线性升温）。

    带 reset_state_at_unfreeze 的参数组，在升温开始那一步清零 AdamW 状态（exp_avg / exp_avg_sq / step）。
    清零是原地操作，显存不变；断点续训时由 last_epoch 判断，只会发生一次。
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        warmup, decay = self.num_warmup_steps, self.num_decay_steps
        if num_training_steps < decay:
            scale = num_training_steps / decay
            warmup, decay = int(warmup * scale), num_training_steps
        floor = self.decay_lr / self.peak_lr

        def base(step: int) -> float:
            if step < warmup:
                return (step + 1) / (warmup + 1)
            s = min(step, decay)
            cosine = 0.5 * (1 + math.cos(math.pi * s / decay))
            return floor + (1 - floor) * cosine

        def make(group: dict):
            zero, ramp = int(group.get("lr_zero_steps", 0)), int(group.get("lr_ramp_steps", 0))

            def fn(step: int) -> float:
                if step <= zero and zero > 0:
                    return 0.0
                mult = 1.0 if ramp <= 0 else min(1.0, (step - zero) / ramp)
                return base(step) * mult

            return fn

        return _GroupedLambdaLR(optimizer, [make(g) for g in optimizer.param_groups])


class _GroupedLambdaLR(LambdaLR):
    def step(self, epoch=None):  # noqa: D401 - signature follows LambdaLR
        super().step(epoch)
        for group in self.optimizer.param_groups:
            zero = int(group.get("lr_zero_steps", 0))
            if zero > 0 and group.get("reset_state_at_unfreeze", False) and self.last_epoch == zero:
                for p in group["params"]:
                    state = self.optimizer.state.get(p)
                    if not state:
                        continue
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key in state:
                            state[key].zero_()
                    if "step" in state:
                        if hasattr(state["step"], "zero_"):
                            state["step"].zero_()
                        else:
                            state["step"] = 0.0
                logging.info("bpv4_bpen: reset optimizer state of group %r at step %d", group.get("name"), zero)


@PreTrainedConfig.register_subclass(BPV4_BPEN)
@dataclass
class BPV4BPEnConfig(Q35Config):
    """Q35 主干 + 只用图像的 BP encoder；prefix 为 语言 + p token + 重采样后的 BP token，没有当前图像。"""

    _canonical_type: ClassVar[str] = BPV4_BPEN
    use_language: bool = True

    # 与数据集缓存一致。
    bp_num_keyframes: int = 32
    bp_num_cameras: int = 3
    bp_visual_dim: int = 1024
    # chunk compressor：每个关键帧 3 路 × 64 个视觉 token → Q 个 token。
    bp_num_query_tokens: int = 8
    bp_compressor_dim: int = 512
    bp_compressor_num_layers: int = 2
    bp_compressor_num_heads: int = 8
    bp_compressor_ff_mult: int = 4
    # 轨迹重采样器：可见关键帧的 K×Q 个 token → M 个 token。
    bp_resampler_num_tokens: int = 128
    bp_resampler_num_layers: int = 2
    bp_resampler_num_heads: int = 16

    # 遮挡课程：遮挡概率从 start 线性升到 end（前 ramp_steps 步）；遮挡时 event_prob 的概率用事件整段。
    mask_prob_start: float = 0.5
    mask_prob_end: float = 0.9
    mask_prob_ramp_steps: int = 3000
    mask_event_segment_prob: float = 0.7

    # 参数组学习率（方案 C）。optimizer_lr 为 act expert 的峰值；其余组按倍数或绝对值。
    bp_new_modules_lr: float = 1e-4
    # 不加载 VLA 权重时 act expert 是随机初始化；None 时用 optimizer_lr。
    act_expert_lr: float | None = None
    vlm_tail_layers: list[int] = field(default_factory=lambda: [20, 21, 22, 23])
    vlm_tail_lr_mult: float = 0.3
    vlm_lr_mult: float = 0.1
    vlm_zero_steps: int = 2500
    vlm_ramp_steps: int = 1000
    vlm_reset_optimizer_state_at_unfreeze: bool = True

    preflight_memory_check: bool = True
    preflight_memory_limit_frac: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.mask_prob_start <= 1.0 or not 0.0 <= self.mask_prob_end <= 1.0:
            raise ValueError("mask_prob_start / mask_prob_end must be in [0, 1]")
        if not 0.0 <= self.mask_event_segment_prob <= 1.0:
            raise ValueError("mask_event_segment_prob must be in [0, 1]")
        if self.bp_compressor_dim % self.bp_compressor_num_heads:
            raise ValueError("bp_compressor_dim must be divisible by bp_compressor_num_heads")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
