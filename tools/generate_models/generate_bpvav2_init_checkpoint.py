"""从 TBot 权重生成 BPVAv2 初始化 checkpoint。

仅需编辑本文件 import 后的顶部大写常量，然后直接运行：

    cd /home/jovyan/workspace/mytbot
    python tools/generate_bpvav2_init_checkpoint.py

脚本会按真实加载规则报告权重来源，使用项目现有 transforms 构造随机 batch，
执行一次有限值 forward 门禁，并且只在 forward 成功后保存 checkpoint。

conda activate bptbot
cd /home/jovyan/workspace/mytbot
python tools/generate_models/generate_bpvav2_init_checkpoint.py
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPVAv2.configuration_bpva import BPVAv2Config
from lerobot.policies.BPVAv2.modeling_bpva import BPVAv2Policy
from lerobot.transforms.core_bp import BPVAv2QwenImageTransformFn, ImgOnlyQwen3VLTransformFn
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE, OBS_STR, SAMPLE_ACTION_LOSS_MASK


# =============================================================================
# 用户配置：只需修改本区，不需要命令行参数
# =============================================================================
TBOT_CHECKPOINT_DIR = Path("/home/jovyan/workspace/models/tbot-pretrain-v2/30w")
OUTPUT_ROOT = Path("/home/jovyan/workspace/models/bpvas")
QWEN3_VL_DIR = Path("/home/jovyan/workspace/models/Qwen3-vl-2b-instruct")
COSMOS_DIR = Path("/home/jovyan/workspace/models/nvidia-cosmos-tokwnizer-ci8x8")
DEVICE = "cuda"
DTYPE = "bfloat16"
SEED = 0
BPVAV2_DIRNAME = "bpvav2"

CONFIG: dict[str, Any] = {
    # BPVAv2 相比 TBot 新增或决定 Query Compressor 结构的配置。
    "bp_num_chunks": 8,
    "bp_action_chunk_size": 50,
    "bp_camera_keys": [
        f"{OBS_IMAGES}.image0",
       # f"{OBS_IMAGES}.image1",
       # f"{OBS_IMAGES}.image2",
    ],
    "bp_encoder_version": "query_compressor_v1",
    "bp_freeze_shared_visual": True,
    "bp_compressor_dim": 512,
    "bp_state_action_hidden_dim": 256,
    "bp_num_query_tokens": 8,
    "bp_compressor_num_layers": 2,
    "bp_compressor_num_heads": 8,
    "bp_compressor_ff_mult": 4,
    "bp_use_modality_type_embedding": True,
    "bp_use_camera_embedding": True,
    "bp_use_chunk_position_embedding": True,
}

# 随机 forward 实际提供的 BP 相机；必须是 CONFIG["bp_camera_keys"] 的非空子集。
ACTIVE_BP_CAMERA_KEYS = [f"{OBS_IMAGES}.image0"]

# 路径、设备及训练期辅助项属于运行时覆盖，不伪装成 BPVAv2 新增结构配置。
RUNTIME_OVERRIDES: dict[str, Any] = {
    "pretrained_path": None,
    "qwen3_vl_pretrained_path": str(QWEN3_VL_DIR),
    "cosmos_tokenizer_path_or_name": str(COSMOS_DIR),
    "device": DEVICE,
    "dtype": DTYPE,
    "lambda_3d": 0.0,
    "lambda_gen": 0.0,
    "gradient_checkpointing": False,
}

ALLOW_OVERWRITE = False
PRINT_ALL_WEIGHT_NAMES = True
WEIGHT_NAME_PREVIEW_LIMIT = 30
PRINT_UNUSED_SOURCE_KEYS = True
UNUSED_SOURCE_KEY_PREVIEW_LIMIT = 50
IMAGE_TIMESTEPS = 3
BP_ENCODER_PREFIX = "model.bp_obs_encoder."


def resolve_save_dir(output_root: Path) -> Path:
    """保证最终目录只有一个尾部 bpvav2。"""
    root = output_root.expanduser()
    return root if root.name.lower() == BPVAV2_DIRNAME.lower() else root / BPVAV2_DIRNAME


def validate_paths(save_dir: Path) -> None:
    for label, path in (
        ("TBot checkpoint", TBOT_CHECKPOINT_DIR),
        ("Qwen3-VL directory", QWEN3_VL_DIR),
        ("Cosmos directory", COSMOS_DIR),
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"{label} does not exist or is not a directory: {path}")
    for filename in ("config.json", SAFETENSORS_SINGLE_FILE):
        if not (TBOT_CHECKPOINT_DIR / filename).is_file():
            raise FileNotFoundError(f"TBot checkpoint has no {filename}: {TBOT_CHECKPOINT_DIR}")
    if save_dir.exists() and not save_dir.is_dir():
        raise NotADirectoryError(f"Resolved output is not a directory: {save_dir}")
    if save_dir.exists() and any(save_dir.iterdir()) and not ALLOW_OVERWRITE:
        raise FileExistsError(
            f"Output directory is non-empty: {save_dir}. Set ALLOW_OVERWRITE=True to let "
            "save_pretrained replace its own files; this script never recursively deletes the directory."
        )


def validate_user_config() -> None:
    required = {
        "bp_num_chunks", "bp_action_chunk_size", "bp_camera_keys", "bp_encoder_version",
        "bp_freeze_shared_visual", "bp_compressor_dim", "bp_state_action_hidden_dim",
        "bp_num_query_tokens", "bp_compressor_num_layers", "bp_compressor_num_heads",
        "bp_compressor_ff_mult", "bp_use_modality_type_embedding",
        "bp_use_camera_embedding", "bp_use_chunk_position_embedding",
    }
    missing = sorted(required.difference(CONFIG))
    if missing:
        raise ValueError(f"CONFIG is missing explicit BPVAv2 fields: {missing}")
    policy_keys = list(CONFIG["bp_camera_keys"])
    active_keys = list(ACTIVE_BP_CAMERA_KEYS)
    if not active_keys:
        raise ValueError("ACTIVE_BP_CAMERA_KEYS must be non-empty")
    if len(set(active_keys)) != len(active_keys):
        raise ValueError("ACTIVE_BP_CAMERA_KEYS must not contain duplicates")
    unknown = sorted(set(active_keys).difference(policy_keys))
    if unknown:
        raise ValueError(
            "ACTIVE_BP_CAMERA_KEYS must be a subset of CONFIG['bp_camera_keys']; "
            f"unknown keys: {unknown}"
        )
    for name in ("bp_num_chunks", "bp_action_chunk_size"):
        if int(CONFIG[name]) <= 0:
            raise ValueError(f"CONFIG[{name!r}] must be positive")
    if int(CONFIG["bp_action_chunk_size"]) > 0 and int(CONFIG["bp_num_chunks"]) <= 0:
        raise ValueError("bp_action_chunk_size requires a positive bp_num_chunks")


def build_bpvav2_config() -> tuple[PreTrainedConfig, BPVAv2Config]:
    """深拷贝源 config 与 BPVAv2Config 的共有字段，再应用明确覆盖。"""
    source = PreTrainedConfig.from_pretrained(TBOT_CHECKPOINT_DIR, local_files_only=True)
    if getattr(source, "type", None) == "bpvav2":
        raise ValueError(
            "TBOT_CHECKPOINT_DIR points to a bpvav2 checkpoint; this script expects a TBot source "
            "whose BP encoder must not be loaded."
        )
    destination_fields = {field.name for field in fields(BPVAv2Config) if field.init}
    copied: dict[str, Any] = {
        name: copy.deepcopy(getattr(source, name))
        for name in destination_fields
        if hasattr(source, name)
    }
    copied.update(copy.deepcopy(RUNTIME_OVERRIDES))
    copied.update(copy.deepcopy(CONFIG))
    config = BPVAv2Config(**copied)
    config.validate_features()
    if ACTION not in config.output_features or OBS_STATE not in config.input_features:
        raise ValueError("Resolved config must preserve source action and observation.state features")
    if not config.output_features[ACTION].shape or not config.input_features[OBS_STATE].shape:
        raise ValueError("Source action/state feature shapes must be non-empty")
    if config.bp_action_chunk_size != int(CONFIG["bp_action_chunk_size"]):
        raise ValueError("Resolved bp_action_chunk_size differs from CONFIG")
    if config.bp_num_chunks != int(CONFIG["bp_num_chunks"]):
        raise ValueError("Resolved bp_num_chunks differs from CONFIG")
    return source, config


def _shape_numel(shape: tuple[int, ...]) -> int:
    return math.prod(shape)


def read_source_shapes(weight_path: Path) -> dict[str, tuple[int, ...]]:
    """仅读取 safetensors 元数据，不把整个源 state dict 载入内存。"""
    shapes: dict[str, tuple[int, ...]] = {}
    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            shapes[key] = tuple(checkpoint.get_slice(key).get_shape())
    return shapes


def classify_weight_sources(
    policy: BPVAv2Policy, source_shapes: dict[str, tuple[int, ...]]
) -> tuple[dict[str, list[tuple[str, tuple[int, ...]]]], list[tuple[str, tuple[int, ...], str]]]:
    """严格复现 TBot 源下 BP encoder 跳过及同名同 shape 加载条件。"""
    categories = {
        "TBOT_LOADED": [],
        "BPVAV2_RANDOM_INIT": [],
        "OTHER_NOT_LOADED_FROM_TBOT": [],
    }
    parameter_shapes = {name: tuple(parameter.shape) for name, parameter in policy.named_parameters()}
    for name, shape in parameter_shapes.items():
        source_shape = source_shapes.get(name)
        if name.startswith(BP_ENCODER_PREFIX):
            categories["BPVAV2_RANDOM_INIT"].append((name, shape))
        elif source_shape == shape:
            categories["TBOT_LOADED"].append((name, shape))
        else:
            categories["OTHER_NOT_LOADED_FROM_TBOT"].append((name, shape))

    # 源未使用键按真实 state_dict 加载面判断，包含源 buffer，不误把可加载 buffer 算作未使用。
    target_shapes = {name: tuple(value.shape) for name, value in policy.state_dict().items()}
    unused: list[tuple[str, tuple[int, ...], str]] = []
    for name, source_shape in source_shapes.items():
        target_shape = target_shapes.get(name)
        if name.startswith(BP_ENCODER_PREFIX):
            reason = "BP encoder skipped because source config type is not bpvav2 (legacy/incompatible BP key)"
        elif target_shape is None:
            reason = "no target state_dict key"
        elif target_shape != source_shape:
            reason = f"shape mismatch, target={target_shape}"
        else:
            continue
        unused.append((name, source_shape, reason))
    return categories, unused


def _selected_entries(entries: list[Any], print_all: bool, limit: int) -> list[Any]:
    return entries if print_all else entries[:limit]


def print_weight_report(
    stage: str,
    categories: dict[str, list[tuple[str, tuple[int, ...]]]],
    unused_source: list[tuple[str, tuple[int, ...], str]],
) -> None:
    print(f"\n===== WEIGHT SOURCE REPORT ({stage}) =====")
    descriptions = {
        "TBOT_LOADED": "source has same name/shape and loader does not skip it",
        "BPVAV2_RANDOM_INIT": "all Query Compressor parameters under model.bp_obs_encoder.*",
        "OTHER_NOT_LOADED_FROM_TBOT": "may be initialized by Qwen/Cosmos/current model; not claimed random",
    }
    for category, entries in categories.items():
        numel = sum(_shape_numel(shape) for _, shape in entries)
        print(f"{category}: tensors={len(entries)}, parameter_numel={numel}")
        print(f"  rule: {descriptions[category]}")
        shown = _selected_entries(entries, PRINT_ALL_WEIGHT_NAMES, WEIGHT_NAME_PREVIEW_LIMIT)
        for name, shape in shown:
            print(f"  - {name} shape={shape}")
        if len(shown) < len(entries):
            print(f"  ... {len(entries) - len(shown)} more (set PRINT_ALL_WEIGHT_NAMES=True)")

    print(f"UNUSED_SOURCE_KEYS: tensors={len(unused_source)}")
    if PRINT_UNUSED_SOURCE_KEYS:
        shown_unused = _selected_entries(
            unused_source, PRINT_ALL_WEIGHT_NAMES, UNUSED_SOURCE_KEY_PREVIEW_LIMIT
        )
        for name, shape, reason in shown_unused:
            print(f"  - {name} shape={shape}; reason={reason}")
        if len(shown_unused) < len(unused_source):
            print(f"  ... {len(unused_source) - len(shown_unused)} more")
    print("===== END WEIGHT SOURCE REPORT =====\n")


def add_batch_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, dict):
        return {key: add_batch_dim(item) for key, item in value.items()}
    if isinstance(value, list):
        return [add_batch_dim(item) for item in value]
    if isinstance(value, tuple):
        return tuple(add_batch_dim(item) for item in value)
    return value


def move_tensors(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def build_random_forward_batch(
    config: BPVAv2Config, generator: torch.Generator
) -> dict[str, Any]:
    """先构造未 batch 的随机 sample，再走项目真实 Qwen/BPVAv2 transforms。"""
    height, width = (int(value) for value in config.image_resolution)
    state_dim = int(config.input_features[OBS_STATE].shape[0])
    action_dim = int(config.output_features[ACTION].shape[0])
    sample: dict[str, Any] = {
        OBS_STATE: torch.rand(state_dim, generator=generator) * 2 - 1,
        ACTION: torch.rand(config.chunk_size, action_dim, generator=generator) * 2 - 1,
        SAMPLE_ACTION_LOSS_MASK: torch.tensor(1.0, dtype=torch.float32),
    }
    for camera_idx in range(3):
        key = f"{OBS_IMAGES}.image{camera_idx}"
        sample[key] = torch.rand(IMAGE_TIMESTEPS, 3, height, width, generator=generator)
        sample[f"{key}_mask"] = torch.tensor(True)

    num_chunks = int(config.bp_num_chunks)
    active_keys = list(ACTIVE_BP_CAMERA_KEYS)
    sample["behavior_prompt"] = {
        "images": {
            key: torch.rand(num_chunks, 3, height, width, generator=generator)
            for key in active_keys
        },
        "state": torch.rand(num_chunks, config.max_state_dim, generator=generator) * 2 - 1,
        "action": torch.rand(
            num_chunks, config.bp_action_chunk_size, config.max_action_dim, generator=generator
        ) * 2 - 1,
        "action_is_pad": torch.zeros(num_chunks, config.bp_action_chunk_size, dtype=torch.bool),
        "image_masks": {key: torch.ones(num_chunks, dtype=torch.bool) for key in active_keys},
        "mask": torch.ones(num_chunks, dtype=torch.bool),
        "chunk_indices": torch.arange(num_chunks, dtype=torch.long),
        "state_is_available": torch.ones(num_chunks, dtype=torch.bool),
    }
    sample = ImgOnlyQwen3VLTransformFn(
        pretrained_model_name_or_path=str(QWEN3_VL_DIR)
    )(sample)
    sample = BPVAv2QwenImageTransformFn(
        pretrained_model_name_or_path=str(QWEN3_VL_DIR)
    )(sample)
    return add_batch_dim(sample)


def print_batch_summary(batch: dict[str, Any], config: BPVAv2Config) -> None:
    prompt = batch["behavior_prompt"]
    print(f"BP policy slots ({len(config.bp_camera_keys)}): {config.bp_camera_keys}")
    print(f"BP active cameras ({len(ACTIVE_BP_CAMERA_KEYS)}): {ACTIVE_BP_CAMERA_KEYS}")
    print(f"current image shape: {tuple(batch[f'{OBS_IMAGES}.image0'].shape)}")
    print(f"current pixel_values shape: {tuple(batch[f'{OBS_STR}.pixel_values'].shape)}")
    print(f"current image_grid_thw shape: {tuple(batch[f'{OBS_STR}.image_grid_thw'].shape)}")
    print(f"state/action shapes: {tuple(batch[OBS_STATE].shape)} / {tuple(batch[ACTION].shape)}")
    for key in ACTIVE_BP_CAMERA_KEYS:
        print(
            f"{key} BP pixels/grid: {tuple(prompt['bp_pixel_values'][key].shape)} / "
            f"{tuple(prompt['bp_image_grid_thw'][key].shape)}"
        )


def run_forward(policy: BPVAv2Policy, batch: dict[str, Any]) -> dict[str, float]:
    policy.eval()
    batch = move_tensors(batch, next(policy.parameters()).device)
    with torch.no_grad():
        loss, loss_dict = policy.forward(batch)
    required = {"loss", "loss_action", "loss_gen", "loss_3d"}
    missing = required.difference(loss_dict)
    if missing:
        raise RuntimeError(f"policy.forward loss_dict is missing keys: {sorted(missing)}")
    if not bool(torch.isfinite(loss).all().item()):
        raise RuntimeError(f"policy.forward returned non-finite loss: {loss}")
    for key, value in loss_dict.items():
        if not math.isfinite(float(value)):
            raise RuntimeError(f"policy.forward returned non-finite {key}: {value}")
    print("forward losses: " + ", ".join(
        f"{key}={float(value):.6g}" for key, value in sorted(loss_dict.items())
    ))
    return {key: float(value) for key, value in loss_dict.items()}


def save_and_verify(policy: BPVAv2Policy, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(save_dir)
    config_path = save_dir / "config.json"
    weight_path = save_dir / SAFETENSORS_SINGLE_FILE
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError(f"Checkpoint save is incomplete: {save_dir}")
    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
    bp_keys = [key for key in keys if key.startswith(BP_ENCODER_PREFIX)]
    if not bp_keys:
        raise RuntimeError(f"Saved checkpoint contains no {BP_ENCODER_PREFIX}* tensors")
    required_fragments = (
        "query_tokens", "state_mlp", "action_mlp", "action_position_embedding",
        "compressor_layers", "output_projection",
    )
    missing_fragments = [
        fragment for fragment in required_fragments if not any(fragment in key for key in bp_keys)
    ]
    if policy.config.bp_use_modality_type_embedding and not any("type_embedding" in key for key in bp_keys):
        missing_fragments.append("type_embedding")
    if policy.config.bp_use_camera_embedding and not any("camera_embedding" in key for key in bp_keys):
        missing_fragments.append("camera_embedding")
    if policy.config.bp_use_chunk_position_embedding and not any(
        "chunk_position_embedding" in key for key in bp_keys
    ):
        missing_fragments.append("chunk_position_embedding")
    if missing_fragments:
        raise RuntimeError(f"Saved query compressor is incomplete: {missing_fragments}")
    duplicate_visual_keys = [key for key in bp_keys if ".visual." in key.lower()]
    if duplicate_visual_keys:
        raise RuntimeError(
            "BP encoder contains a duplicated nested .visual. module: "
            + ", ".join(duplicate_visual_keys[:8])
        )
    print(f"saved checkpoint: {save_dir}")
    print(f"verified config.json/model.safetensors; total/BP tensors={len(keys)}/{len(bp_keys)}")
    print("verified query-compressor fragments and no duplicated model.bp_obs_encoder.*.visual.*")


def main() -> None:
    validate_user_config()
    save_dir = resolve_save_dir(OUTPUT_ROOT)
    validate_paths(save_dir)
    random.seed(SEED)
    torch.manual_seed(SEED)
    generator = torch.Generator(device="cpu").manual_seed(SEED)

    source_config, config = build_bpvav2_config()
    print(f"source checkpoint: {TBOT_CHECKPOINT_DIR}")
    print(f"source config type: {getattr(source_config, 'type', None)!r}")
    print(f"resolved output: {save_dir}")
    print(f"device/dtype: {config.device}/{config.dtype}")

    # 只构造一次目标模型；随后直接调用与 from_pretrained 相同的 policy 专用加载器。
    policy = BPVAv2Policy(config)
    setattr(policy, "_loaded_pretrained_source_config", source_config)
    source_weight_path = TBOT_CHECKPOINT_DIR / SAFETENSORS_SINGLE_FILE
    source_shapes = read_source_shapes(source_weight_path)
    categories, unused_source = classify_weight_sources(policy, source_shapes)
    print_weight_report("BEFORE LOAD", categories, unused_source)

    policy = BPVAv2Policy._load_as_safetensor(
        policy, str(source_weight_path), config.device, strict=False
    )
    policy.to(config.device)
    policy.eval()
    print_weight_report("AFTER LOAD (CONFIRMED BY LOADER RULES)", categories, unused_source)

    batch = build_random_forward_batch(config, generator)
    print_batch_summary(batch, config)
    run_forward(policy, batch)
    # 严格保证 forward 成功后才创建目录和保存。
    save_and_verify(policy, save_dir)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
