from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.q35.configuration_q35 import Q35Config
from lerobot.utils.constants import OBS_IMAGES


BPVAV3 = "bpvav3"
_DEFAULT_BP_CAMERAS = [
    f"{OBS_IMAGES}.image0",
    f"{OBS_IMAGES}.image1",
    f"{OBS_IMAGES}.image2",
]


def _validate_bp_camera_keys(bp_camera_keys: list[str]) -> None:
    if not bp_camera_keys:
        raise ValueError("bp_camera_keys must contain 1 or more canonical image keys")
    if len(set(bp_camera_keys)) != len(bp_camera_keys):
        raise ValueError("bp_camera_keys must not contain duplicates")
    invalid = [key for key in bp_camera_keys if not key.startswith(f"{OBS_IMAGES}.")]
    if invalid:
        raise ValueError(f"bp_camera_keys must be canonical observation image keys: {invalid}")


@DatasetConfig.register_subclass(BPVAV3)
@dataclass
class BPVAv3DatasetConfig(DatasetConfig):
    """Q35 当前观测加上一条 BP 示教轨迹。

    变换链由 `make_bpvav3_behavior_dataset` 按机器人特征补全。
    配置里只保留采样和处理器参数，避免和运行时 hydrate 后的变换分叉。
    """

    _canonical_type: ClassVar[str] = BPVAV3
    action_mode: str = "delta"
    height: int = 224
    width: int = 224
    max_state_dim: int = 32
    max_action_dim: int = 32
    current_image_index: int = 0
    qwen3_5_processor_path: str = "Qwen/Qwen3.5-0.8B-Base"
    qwen3_5_processor_revision: str | None = None
    use_language: bool = True
    bp_num_chunks: int = 8
    bp_seed: int = 0
    # random 可抽到当前 episode；origin 用当前整段轨迹做示教；avoid 排除当前 episode。
    bp_sampling_mode: str = "random"
    # 为 true 时额外挂 future_prompt。它只做 BP encoder 对齐监督，不进入 VLM 条件。
    align_bp_encoder: bool = False
    # 只作用于 future_prompt：丢掉锚点早于当前帧的 chunk，至少保留最后一个有效 chunk。
    future_filter: bool = True
    bp_camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_BP_CAMERAS))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.current_image_index < 0:
            raise ValueError("current_image_index must be non-negative")
        if self.bp_sampling_mode not in {"random", "origin", "avoid"}:
            raise ValueError(
                "bp_sampling_mode must be 'random', 'origin', or 'avoid', "
                f"got {self.bp_sampling_mode!r}"
            )
        _validate_bp_camera_keys(self.bp_camera_keys)


@PreTrainedConfig.register_subclass(BPVAV3)
@dataclass
class BPVAv3Config(Q35Config):
    """Q35 action expert。BP query 接在当前 prefix 后面，一起进入 Qwen3.5 VLM。"""

    _canonical_type: ClassVar[str] = BPVAV3
    use_language: bool = True

    bp_num_chunks: int = 8
    bp_action_chunk_size: int = 50
    bp_camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_BP_CAMERAS))
    bp_compressor_dim: int = 512
    bp_state_action_hidden_dim: int = 256
    bp_num_query_tokens: int = 8
    bp_compressor_num_layers: int = 2
    bp_compressor_num_heads: int = 8
    bp_compressor_ff_mult: int = 4
    # 0 时只算 action loss，也不把 align query 送进 VLM。
    lambda_bp_encoder: float = 0.0
    # true 时冻结 bp_obs_encoder 里 ViT 以外的部分：投影、state/action MLP 和 compressor。
    # 不碰共享 Qwen3.5 ViT。
    frozen_bp_encoder: bool = False
    # true 时冻结共享 Qwen3.5 ViT。当前观测和 BP 图像走同一份参数，两边都不再更新。
    # 默认 false，已有配置不写这个字段时 ViT 仍会训练。
    frozen_vit: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.bp_action_chunk_size <= 0:
            raise ValueError("bp_action_chunk_size must be positive")
        for name in (
            "bp_compressor_dim",
            "bp_state_action_hidden_dim",
            "bp_num_query_tokens",
            "bp_compressor_num_layers",
            "bp_compressor_num_heads",
            "bp_compressor_ff_mult",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.bp_compressor_dim % self.bp_compressor_num_heads:
            raise ValueError("bp_compressor_dim must be divisible by bp_compressor_num_heads")
        if float(self.lambda_bp_encoder) < 0:
            raise ValueError("lambda_bp_encoder must be non-negative")
        _validate_bp_camera_keys(self.bp_camera_keys)


def validate_bpvav3_stage_pair(dataset_cfg: BPVAv3DatasetConfig, policy_cfg: BPVAv3Config) -> None:
    """阶段 1 用 origin 且 lambda 为 0；阶段 2 的对齐不能把当前轨迹同时放进条件。"""
    lam = float(policy_cfg.lambda_bp_encoder)
    if lam != 0.0 and not bool(dataset_cfg.align_bp_encoder):
        raise ValueError(
            "lambda_bp_encoder != 0 requires dataset.align_bp_encoder=true so future_prompt is built"
        )
    if lam != 0.0 and dataset_cfg.bp_sampling_mode == "origin":
        raise ValueError(
            "bp_sampling_mode='origin' puts the current trajectory into the VLM condition "
            "and cannot be combined with lambda_bp_encoder != 0"
        )
