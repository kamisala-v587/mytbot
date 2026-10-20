"""BPv4 第二阶段（跨示教）：示教轨迹 D 的 BP + 当前观测 Q（三路图像 + state）→ 之后 50 步 action。

BP 编码器、VLM、act expert 从第一阶段（bpv4_bpen）checkpoint 初始化；不输入语言，没有进度 token。
设计见 code/conclusion/bpv4/bpv4 第二阶段详细方案.md。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4GroupedSchedulerConfig
from lerobot.policies.q35.configuration_q35 import Q35Config

BPV4_XDEMO = "bpv4_xdemo"
_DEFAULT_CAMERAS = [
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
]


@DatasetConfig.register_subclass(BPV4_XDEMO)
@dataclass
class BPV4XDemoDatasetConfig(DatasetConfig):
    """每个数据项是一对轨迹：当前轨迹 Q 上的 S 个查询（三路图像、state、action chunk）+ 同任务另一条轨迹 D 的 BP。"""

    _canonical_type: ClassVar[str] = BPV4_XDEMO
    action_mode: str = "delta"
    max_state_dim: int = 32
    max_action_dim: int = 32
    qwen3_5_processor_path: str = "/vla/workspace/models/q35/base_800m"

    # BP：与第一阶段的特征缓存一一对应。
    bp_num_keyframes: int = 32
    bp_keyframe_alpha: float = 0.5
    bp_camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_CAMERAS))
    bp_feature_cache_dir: str = "/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2"
    # 训练时每个示教关键帧独立丢弃的概率（至少保留 min_visible 帧）；验证 / 评测不丢弃。
    bp_keyframe_drop_prob: float = 0.15
    bp_min_visible_keyframes: int = 8

    # 当前观测：与 BP 缓存相同的预处理（resize_with_pad 到 image_size，再走 Qwen 图像处理器）。
    image_size: int = 224
    queries_per_trajectory: int = 4

    # 示教与当前取同一 episode 的概率（重新打开抄轨迹的捷径，默认 0）。
    same_episode_prob: float = 0.0
    # 按指令文本里的关键词划分类别、只在同类别之间配对的任务（物体类别轻度变化的 A2* 任务）。
    category_pairing: dict[str, list[str]] = field(
        default_factory=lambda: {"place_container_plate": ["bowl", "cup"]}
    )

    val_holdout_last: dict[str, int] = field(
        default_factory=lambda: {"clean_50": 5, "randomized_100": 10}
    )
    # train / val / all。示教只在同一 split 内选取。
    split: str = "train"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.split not in {"train", "val", "all"}:
            raise ValueError("split must be train, val or all")
        if not 0.0 <= self.bp_keyframe_drop_prob < 1.0 or not 0.0 <= self.same_episode_prob <= 1.0:
            raise ValueError("bp_keyframe_drop_prob must be in [0, 1) and same_episode_prob in [0, 1]")
        if self.queries_per_trajectory <= 0 or self.bp_num_keyframes <= 0:
            raise ValueError("queries_per_trajectory and bp_num_keyframes must be positive")


@PreTrainedConfig.register_subclass(BPV4_XDEMO)
@dataclass
class BPV4XDemoConfig(Q35Config):
    """Q35 主干；prefix = 当前三路图像 token（mrope 位置）+ 示教的 M 个 BP token（顺序位置），没有语言和进度 token。"""

    _canonical_type: ClassVar[str] = BPV4_XDEMO

    # BP 编码器结构：必须与第一阶段 checkpoint 一致。
    bp_num_keyframes: int = 32
    bp_num_cameras: int = 3
    bp_visual_dim: int = 1024
    bp_num_query_tokens: int = 8
    bp_compressor_dim: int = 512
    bp_compressor_num_layers: int = 2
    bp_compressor_num_heads: int = 8
    bp_compressor_ff_mult: int = 4
    bp_resampler_num_tokens: int = 128
    bp_resampler_num_layers: int = 2
    bp_resampler_num_heads: int = 16

    # 参数组学习率。optimizer_lr 为 VLM 倍率的基准。
    new_modules_lr: float = 1e-4
    act_expert_lr: float = 1e-4
    bp_encoder_lr: float = 3e-5
    bp_encoder_zero_steps: int = 1500
    bp_encoder_ramp_steps: int = 500
    bp_encoder_reset_optimizer_state_at_unfreeze: bool = True
    vlm_tail_layers: list[int] = field(default_factory=lambda: [20, 21, 22, 23])
    vlm_tail_lr_mult: float = 0.3
    vlm_lr_mult: float = 0.1
    vlm_ramp_steps: int = 1000

    preflight_memory_check: bool = True
    preflight_memory_limit_frac: float = 0.9

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bp_compressor_dim % self.bp_compressor_num_heads:
            raise ValueError("bp_compressor_dim must be divisible by bp_compressor_num_heads")

    def get_scheduler_preset(self) -> BPV4GroupedSchedulerConfig:
        return BPV4GroupedSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )
