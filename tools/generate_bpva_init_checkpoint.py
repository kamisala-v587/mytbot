"""从 TBot 基础 checkpoint 生成包含 BPObsEncoder 的 BPVA 初始化权重。"""

from pathlib import Path

import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open

from lerobot.policies.BPVA.configuration_bpva import BPVAConfig
from lerobot.policies.BPVA.modeling_bpva import BPVAPolicy


# =============================================================================
# 用户配置：通常只需要改 BP_NUM_CHUNKS、BP_ACTION_CHUNK_SIZE 和 SAVE_DIR。
# =============================================================================

LOAD_MODE = "TBOT_BASE"  # 可选："TBOT_BASE" / "BPVA" / "SCRATCH"
TBOT_BASE_DIR = Path("/vla/workspace/models/tbot_base")
BPVA_SOURCE_DIR = Path("/vla/workspace/models/bpva_init_c10_a50")
SAVE_DIR = Path("/vla/workspace/models/bpva_init_c10_a50")

QWEN3_VL_DIR = Path("/vla/workspace/models/Qwen3-VL-2B-Instruct")
COSMOS_DIR = Path("/vla/workspace/models/Cosmos-Tokenizer-CI8x8")
DEVICE = "cuda"
DTYPE = "bfloat16"
STRICT_LOAD = False
ALLOW_OVERWRITE = False

# K：每个样本包含的 behavior prompt chunk 数量。
BP_NUM_CHUNKS = 10
# 每个 behavior prompt chunk 中包含的 action 时间步数。
BP_ACTION_CHUNK_SIZE = 50

# 下列 BPObsEncoder 结构参数一般保持不变。改动后同样需要重新生成初始化权重。
BP_VISION_MODEL_NAME = "vit_base_patch16_clip_224.openai"
BP_VISION_PRETRAINED = True
BP_FREEZE_VISION_ENCODER = True
BP_TOKEN_DIM = 768
BP_IMAGE_FEATURE_AGGREGATION = "cls"

# 初始化 checkpoint 不加载 DA3 teacher，训练时可通过训练配置重新启用 3D loss。
ENABLE_3D_QUERIES = True
LAMBDA_3D = 0.0


def validate_paths() -> None:
    if LOAD_MODE == "TBOT_BASE" and not TBOT_BASE_DIR.is_dir():
        raise FileNotFoundError(f"TBot 基础 checkpoint 不存在：{TBOT_BASE_DIR}")
    if LOAD_MODE == "BPVA" and not BPVA_SOURCE_DIR.is_dir():
        raise FileNotFoundError(f"BPVA 来源 checkpoint 不存在：{BPVA_SOURCE_DIR}")
    for external_dir in (QWEN3_VL_DIR, COSMOS_DIR):
        if not external_dir.is_dir():
            raise FileNotFoundError(f"外部模型目录不存在：{external_dir}")
    if SAVE_DIR.exists() and any(SAVE_DIR.iterdir()) and not ALLOW_OVERWRITE:
        raise FileExistsError(
            f"输出目录非空：{SAVE_DIR}。请修改 SAVE_DIR，或明确设置 ALLOW_OVERWRITE=True。"
        )


def build_config() -> BPVAConfig:
    return BPVAConfig(
        pretrained_path=None,
        device=DEVICE,
        dtype=DTYPE,
        qwen3_vl_pretrained_path=str(QWEN3_VL_DIR),
        cosmos_tokenizer_path_or_name=str(COSMOS_DIR),
        bp_num_chunks=BP_NUM_CHUNKS,
        bp_action_chunk_size=BP_ACTION_CHUNK_SIZE,
        bp_vision_model_name=BP_VISION_MODEL_NAME,
        bp_vision_pretrained=BP_VISION_PRETRAINED,
        bp_freeze_vision_encoder=BP_FREEZE_VISION_ENCODER,
        bp_token_dim=BP_TOKEN_DIM,
        bp_image_feature_aggregation=BP_IMAGE_FEATURE_AGGREGATION,
        enable_3d_queries=ENABLE_3D_QUERIES,
        lambda_3d=LAMBDA_3D,
    )


def build_policy(config: BPVAConfig) -> tuple[BPVAPolicy, Path | None]:
    if LOAD_MODE == "TBOT_BASE":
        load_dir: Path | None = TBOT_BASE_DIR
        policy = BPVAPolicy.from_pretrained(load_dir, config=config, strict=STRICT_LOAD)
    elif LOAD_MODE == "BPVA":
        load_dir = BPVA_SOURCE_DIR
        policy = BPVAPolicy.from_pretrained(load_dir, config=config, strict=STRICT_LOAD)
    elif LOAD_MODE == "SCRATCH":
        load_dir = None
        policy = BPVAPolicy(config)
    else:
        raise ValueError(f"不支持的 LOAD_MODE：{LOAD_MODE}")
    return policy.eval(), load_dir


def save_and_verify(policy: BPVAPolicy) -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(SAVE_DIR)

    config_path = SAVE_DIR / "config.json"
    weight_path = SAVE_DIR / SAFETENSORS_SINGLE_FILE
    if not config_path.is_file() or not weight_path.is_file():
        raise RuntimeError(f"checkpoint 保存不完整：{SAVE_DIR}")

    with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
    bp_keys = [key for key in keys if key.startswith("model.bp_obs_encoder.")]
    if not bp_keys:
        raise RuntimeError("保存的 checkpoint 中没有 model.bp_obs_encoder.* 权重")

    print(f"保存目录：{SAVE_DIR}")
    print(f"配置类型：{policy.config.type}")
    print(f"BP chunk 数：{policy.config.bp_num_chunks}")
    print(f"BP action chunk size：{policy.config.bp_action_chunk_size}")
    print(f"总参数张量数：{len(keys)}")
    print(f"BPObsEncoder 参数张量数：{len(bp_keys)}")
    print(f"权重文件大小：{weight_path.stat().st_size / 1024**3:.2f} GiB")


def main() -> None:
    validate_paths()
    config = build_config()
    policy, load_dir = build_policy(config)
    print(f"加载模式：{LOAD_MODE}")
    print(f"来源目录：{load_dir}")
    print(f"输出目录：{SAVE_DIR}")
    print(f"设备：{config.device}，dtype：{config.dtype}")
    print(
        "BPObsEncoder 已注册：",
        any(key.startswith("model.bp_obs_encoder.") for key in policy.state_dict()),
    )
    save_and_verify(policy)


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
