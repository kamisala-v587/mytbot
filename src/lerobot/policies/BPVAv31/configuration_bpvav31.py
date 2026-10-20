from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from lerobot.configs.default import DatasetConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.q35.configuration_q35 import Q35Config
from lerobot.utils.constants import OBS_IMAGES


BPVAV31 = "bpvav31"
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


@DatasetConfig.register_subclass(BPVAV31)
@dataclass
class BPVAv31DatasetConfig(DatasetConfig):
    """Q35 当前观测加上一条 BP 示教轨迹。

    变换链由 `make_bpvav31_behavior_dataset` 按机器人特征补全。
    配置里只保留采样和处理器参数，避免和运行时 hydrate 后的变换分叉。
    """

    _canonical_type: ClassVar[str] = BPVAV31
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
    # true 时所有轨迹都取 bp_num_chunks 个关键帧；false 时 K=min(ceil(T/bp_action_chunk_size), bp_num_chunks)。
    # 必须与 policy.bp_fixed_num_chunks 一致，评测侧 BP 缓存读 policy 上的值。
    bp_fixed_num_chunks: bool = False
    # true 时样本带 bp_task_id（repo 上级目录名，即任务名的 hash），policy.lambda_bp_task>0 时必须开启。
    bp_attach_task_id: bool = False
    # true 时额外挂当前 episode 的关键帧 state / 进度，供检索 CE 与下一关键帧 state MSE 使用。
    bp_attach_retrieval: bool = True
    # false 时不再解码当前 episode 的 origin_prompt（只有增量 / 绝对 next-state 监督需要它）。
    bp_attach_origin: bool = True
    # None：不按手臂约束。p：以概率 p 抽同手示教、1-p 抽异手示教（无对应示教时退回全部候选）。
    bp_arm_match_prob: float | None = None
    # 单臂判定：一只手关节总运动量超过另一只手的倍数。
    bp_arm_ratio: float = 3.0
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


@PreTrainedConfig.register_subclass(BPVAV31)
@dataclass
class BPVAv31Config(Q35Config):
    """Q35 action expert。BP query 接在当前 prefix 后面，一起进入 Qwen3.5 VLM。"""

    _canonical_type: ClassVar[str] = BPVAV31
    use_language: bool = True

    bp_num_chunks: int = 8
    # 与 dataset.bp_fixed_num_chunks 相同，保存在 checkpoint 里供评测侧构造同样的 BP。
    bp_fixed_num_chunks: bool = False
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
    # >0 时对 BP 全局池化（8 chunk 平均）经投影头加同任务 SupCon；不再逐 chunk 压成 task id。
    # 需要 batch 内有多个任务：dataset.homogeneous_batch=false，dataset.bp_attach_task_id=true。
    lambda_bp_task: float = 0.0
    bp_task_tau: float = 0.1
    bp_task_proj_dim: int = 256
    # >0：当前观测 → 8 个 BP chunk 的检索 CE（目标=归一化时间上“下一个关键帧”）。
    lambda_bp_retrieve: float = 0.0
    bp_retrieve_tau: float = 0.1
    # >0：检索加权后的 BP 表征预测当前轨迹下一关键帧 state（MSE）。
    lambda_bp_next_state: float = 0.0
    # >0：逐 chunk 时序对比（同任务且进度接近为正，同示教不同进度为负）。
    lambda_bp_temporal: float = 0.0
    bp_temporal_tau: float = 0.2
    bp_temporal_progress_tol: float = 0.15
    # true：chunk 位置用 source_time_ratio 的正弦编码，替代可学习 Embedding。
    bp_use_progress_encoding: bool = True
    # timestamp：query 带当前进度、key 用 VLM 后带时间编码的 BP（可只靠时间戳解）。
    # content：query = 当前 ViT 特征 + state，key = 加时间编码前的 BP chunk 内容，目标是以当前进度为中心的软标签。
    bp_retrieve_mode: str = "timestamp"
    bp_retrieve_sigma: float = 0.1
    # content 模式下只在「同手 / 非当前轨迹」样本上算对齐类损失（检索、时序、next-state）。
    bp_align_require_arm_match: bool = True
    bp_align_skip_self: bool = True
    # post_vlm：旧时序对比（VLM 后、带时间编码）；content：在加时间编码前的 chunk 内容上做跨示教 TCN。
    bp_temporal_mode: str = "post_vlm"
    # absolute：旧的绝对 state 预测；delta：用检索到的内容 + 当前 state 预测「下一关键帧 state - 当前 state」。
    bp_next_state_mode: str = "absolute"
    # true 时冻结 bp_obs_encoder 里 ViT 以外的部分：投影、state/action MLP 和 compressor。
    # 不碰共享 Qwen3.5 ViT。
    frozen_bp_encoder: bool = False
    # true 时冻结共享 Qwen3.5 ViT。当前观测和 BP 图像走同一份参数，两边都不再更新。
    # 默认 false，已有配置不写这个字段时 ViT 仍会训练。
    frozen_vit: bool = False
    # true 时只让 BP 图像在 no_grad 下过共享 ViT，ViT 仅由当前观测那一路的梯度更新。
    # 当前观测和 BP 图像会分两次过 ViT。frozen_vit=true 时本开关无额外作用。
    frozen_bp_vit: bool = False

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
        for name in ("lambda_bp_encoder", "lambda_bp_task", "lambda_bp_retrieve", "lambda_bp_next_state", "lambda_bp_temporal"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if float(self.bp_task_tau) <= 0 or int(self.bp_task_proj_dim) <= 0:
            raise ValueError("bp_task_tau and bp_task_proj_dim must be positive")
        if float(self.bp_retrieve_tau) <= 0 or float(self.bp_temporal_tau) <= 0:
            raise ValueError("bp_retrieve_tau and bp_temporal_tau must be positive")
        if not (0.0 < float(self.bp_temporal_progress_tol) <= 1.0):
            raise ValueError("bp_temporal_progress_tol must be in (0, 1]")
        if self.bp_retrieve_mode not in {"timestamp", "content"}:
            raise ValueError("bp_retrieve_mode must be timestamp or content")
        if self.bp_temporal_mode not in {"post_vlm", "content"}:
            raise ValueError("bp_temporal_mode must be post_vlm or content")
        if self.bp_next_state_mode not in {"absolute", "delta"}:
            raise ValueError("bp_next_state_mode must be absolute or delta")
        if float(self.bp_retrieve_sigma) <= 0:
            raise ValueError("bp_retrieve_sigma must be positive")
        if self.bp_next_state_mode == "delta" and self.bp_retrieve_mode != "content":
            raise ValueError("bp_next_state_mode=delta requires bp_retrieve_mode=content")
        _validate_bp_camera_keys(self.bp_camera_keys)


def validate_bpvav31_stage_pair(dataset_cfg: BPVAv31DatasetConfig, policy_cfg: BPVAv31Config) -> None:
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
    needs_task_id = any(
        float(getattr(policy_cfg, name)) != 0.0
        for name in ("lambda_bp_task", "lambda_bp_temporal")
    )
    if needs_task_id:
        if not bool(dataset_cfg.bp_attach_task_id):
            raise ValueError("lambda_bp_task / lambda_bp_temporal != 0 requires dataset.bp_attach_task_id=true")
        if bool(dataset_cfg.homogeneous_batch) and float(policy_cfg.lambda_bp_task) != 0.0:
            raise ValueError("lambda_bp_task != 0 requires dataset.homogeneous_batch=false (single-task batches have no negatives)")
    if float(policy_cfg.lambda_bp_retrieve) != 0.0 or float(policy_cfg.lambda_bp_next_state) != 0.0:
        if not bool(getattr(dataset_cfg, "bp_attach_retrieval", True)):
            raise ValueError("retrieval / next-state losses require dataset.bp_attach_retrieval=true")
    if float(policy_cfg.lambda_bp_next_state) != 0.0 and not bool(getattr(dataset_cfg, "bp_attach_origin", True)):
        raise ValueError("lambda_bp_next_state != 0 requires dataset.bp_attach_origin=true")
    if policy_cfg.bp_temporal_mode == "content" and float(policy_cfg.lambda_bp_temporal) != 0.0:
        if policy_cfg.bp_retrieve_mode != "content":
            raise ValueError("bp_temporal_mode=content requires bp_retrieve_mode=content")
