from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

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


@dataclass(frozen=True)
class BPDimUse:
    """示教某一模态要保留哪些最后一维。

    mode 为 all 时保留全部；none 时整段丢掉；keep 时只保留 pieces 展开后的下标。
    """

    mode: str
    pieces: tuple[tuple[Any, ...], ...] = ()

    def resolve(self, dim: int) -> list[int] | None:
        """返回要保留的下标。None 表示全部保留，空列表表示整段丢掉。"""
        if dim <= 0:
            raise ValueError(f"tensor dim must be positive, got {dim}")
        if self.mode == "all":
            return None
        if self.mode == "none":
            return []
        kept: list[int] = []
        for piece in self.pieces:
            kind = piece[0]
            if kind == "index":
                index = int(piece[1])
                if index >= dim:
                    raise ValueError(f"dimension index {index} is outside 0:{dim}")
                kept.append(index)
                continue
            start = int(piece[1])
            end = dim if piece[2] is None else int(piece[2])
            if start >= dim or end > dim or start >= end:
                raise ValueError(f"dimension slice {start}:{end} is invalid for dim {dim}")
            kept.extend(range(start, end))
        unique = list(dict.fromkeys(kept))
        if len(unique) == dim:
            return None
        return unique


def parse_bp_dim_use(spec: Any, *, name: str) -> BPDimUse:
    """解析 bp_action_use / bp_state_use。

    true / "all" 保留全部。false / "none" 整段丢掉。
    "0:6"、"0:6,7:13"、[0, 1, 2] 只保留这些下标，其余置零。
    区间右端不含，和 Python slice 一致。
    """
    if isinstance(spec, bool):
        return BPDimUse("none" if spec is False else "all")
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError(f"{name} index must be non-negative, got {spec}")
        return BPDimUse("keep", (("index", spec),))
    if isinstance(spec, (list, tuple)):
        if len(spec) == 0:
            return BPDimUse("none")
        pieces = []
        for item in spec:
            if isinstance(item, bool) or not isinstance(item, int):
                raise ValueError(f"{name} list items must be non-negative ints, got {item!r}")
            if item < 0:
                raise ValueError(f"{name} index must be non-negative, got {item}")
            pieces.append(("index", int(item)))
        return BPDimUse("keep", tuple(pieces))
    if isinstance(spec, str):
        text = spec.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1].strip()
        lowered = text.lower()
        if lowered in {"true", "all", "yes", "*"}:
            return BPDimUse("all")
        if lowered in {"false", "none", "no"} or text == "":
            if spec.strip() == "":
                raise ValueError(f"{name} must not be an empty string")
            return BPDimUse("none")
        pieces = []
        for part in text.split(","):
            part = part.strip()
            if not part:
                raise ValueError(f"{name} contains an empty piece: {spec!r}")
            if ":" in part:
                left, right = part.split(":", 1)
                if ":" in right:
                    raise ValueError(f"{name} slice must be start:end, got {part!r}")
                try:
                    start = 0 if left.strip() == "" else int(left)
                    end = None if right.strip() == "" else int(right)
                except ValueError as exc:
                    raise ValueError(f"{name} slice must use integers, got {part!r}") from exc
                if start < 0 or (end is not None and end < 0):
                    raise ValueError(f"{name} slice must be non-negative, got {part!r}")
                pieces.append(("range", start, end))
            else:
                try:
                    index = int(part)
                except ValueError as exc:
                    raise ValueError(f"{name} must be true, false, a slice, or indices, got {spec!r}") from exc
                if index < 0:
                    raise ValueError(f"{name} index must be non-negative, got {index}")
                pieces.append(("index", index))
        return BPDimUse("keep", tuple(pieces))
    raise TypeError(f"{name} must be bool, str, or a list of ints, got {type(spec).__name__}")


def validate_bp_dim_use(spec: Any, *, name: str, max_dim: int) -> BPDimUse:
    parsed = parse_bp_dim_use(spec, name=name)
    parsed.resolve(max_dim)
    return parsed


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
    # random/avoid 默认同 repo；*_in_task 按完整任务描述匹配；avoid_in_task 无候选时 mask BP。
    bp_sampling_mode: str = "random"
    # 为 true 时额外挂 future_prompt。它只做 BP encoder 对齐监督，不进入 VLM 条件。
    align_bp_encoder: bool = False
    # 只作用于 future_prompt：丢掉锚点早于当前帧的 chunk，至少保留最后一个有效 chunk。
    future_filter: bool = True
    # true 使用示教 action 的全部维度。false 整段置零并把 action_is_pad 置为 true。
    # 也可以只保留部分维度："0:6"、"0:6,7:13"，或 [0, 1, 2, 3, 4, 5]。
    # 下标是 pad 到 max_action_dim 之后的最后一维，Aloha 左右夹爪是 6 和 13。
    bp_action_use: Any = True
    # 与 bp_action_use 相同的写法。false 时 state 置零，并把 state_is_available 置为 false。
    bp_state_use: Any = True
    bp_camera_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_BP_CAMERAS))

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.bp_num_chunks <= 0:
            raise ValueError("bp_num_chunks must be positive")
        if self.current_image_index < 0:
            raise ValueError("current_image_index must be non-negative")
        if self.bp_sampling_mode not in {"random", "origin", "avoid", "random_in_task", "random_in_repo", "avoid_in_task", "avoid_in_repo"}:
            raise ValueError(
                "Invalid bp_sampling_mode; expected origin, random[_in_task|_in_repo], or avoid[_in_task|_in_repo], "
                f"got {self.bp_sampling_mode!r}"
            )
        _validate_bp_camera_keys(self.bp_camera_keys)
        validate_bp_dim_use(self.bp_action_use, name="bp_action_use", max_dim=self.max_action_dim)
        validate_bp_dim_use(self.bp_state_use, name="bp_state_use", max_dim=self.max_state_dim)


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
    # cosine compares directions; mse compares raw feature values (including magnitude).
    bp_align_loss_type: str = "cosine"
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
        if self.bp_align_loss_type not in {"cosine", "mse"}:
            raise ValueError("bp_align_loss_type must be cosine or mse")
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
