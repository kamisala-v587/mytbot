#!/usr/bin/env bash
# =============================================================================
# tbot_sa1_finetune_correct.commented.sh
# -----------------------------------------------------------------------------
# 【总体说明】
#
# 你的理解「脚本在封装 TrainPipelineConfig」—— 大方向正确，但不完整：
#
#   1. 核心训练配置段（ARGS 数组里的 --xxx）确实会被 draccus 解析为
#      `lerobot.configs.train.TrainPipelineConfig` 实例，再传入
#      `lerobot.scripts.lerobot_train.train(cfg)`。
#
#   2. 本脚本还额外做了 TrainPipelineConfig 之外的事情：
#      - conda / CUDA / 分布式环境变量
#      - accelerate launch 的多机多卡参数（不属于 TrainPipelineConfig）
#      - 启动前校验（stats 文件是否存在、ACTION_TYPE 合法性等）
#      - 由 DATASET_REPO_ID 推导 DATASET_NAME、external_stats_path 等 shell 逻辑
#
#   3. 本脚本微调的是 TBot_SA1，不是 TBot_SA1_Wan：
#      - Policy 配置类:  TBotSA1Config      (configuration_tbot_sa1.py)
#      - Dataset 配置类: TBotSA1DatasetConfig (同上)
#      - Policy 模型类:  TBotSA1Policy      (modeling_tbot_sa1.py)
#      三者通过 draccus ChoiceRegistry 的 type="TBot_SA1" 关联。
#
#   4. CLI 参数命名规则（draccus 嵌套字段）：
#      --batch_size=8          → TrainPipelineConfig.batch_size
#      --policy.chunk_size=50  → TrainPipelineConfig.policy.chunk_size
#                              → 运行时 policy 对象是 TBotSA1Config 实例
#      --dataset.repo_id=...   → TrainPipelineConfig.dataset.repo_id
#                              → 运行时 dataset 对象是 TBotSA1DatasetConfig 实例
#
#   5. 常见误解纠正：
#      - BATCH_SIZE  → TrainPipelineConfig.batch_size   （不是 steps）
#      - STEPS       → TrainPipelineConfig.steps
#      - policy.push_to_hub → PreTrainedConfig.push_to_hub
#        （定义在 policies.py 基类；TBotSA1Config 继承它，不是 TBotSA1WanPolicy）
#      - policy.optimizer_lr → TBotSA1Config.optimizer_lr
#        若 use_policy_training_preset=true（默认），validate() 会调用
#        TBotSA1Config.get_optimizer_preset() / get_scheduler_preset()
#        生成 TrainPipelineConfig.optimizer / .scheduler
#
# 配置解析入口：
#   src/lerobot/configs/parser.py  @parser.wrap()
#   src/lerobot/scripts/lerobot_train.py  train(cfg: TrainPipelineConfig)
# =============================================================================

set -euo pipefail

###############################################################################
################################# ENV config ##################################
# 以下段落：运行环境配置，不进入 TrainPipelineConfig
###############################################################################

# HuggingFace 缓存目录（datasets / hub 下载），与 TrainPipelineConfig 无关
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"

# WandB / Conda：shell 层工具配置，不进入 TrainPipelineConfig
WANDB_TOKEN="${WANDB_TOKEN:-}"
CONDA_ROOT="${_CONDA_ROOT:-${CONDA_ROOT:-}}"
CONDA_ENV="${CONDA_ENV:-tbot_sa1}"

if [[ -n "${CONDA_ROOT}" && -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV}"
fi

if [[ -n "${WANDB_TOKEN}" ]]; then
    wandb login "${WANDB_TOKEN}"
fi

###############################################################################
# 分布式训练环境变量（PyTorch DDP / accelerate），不进入 TrainPipelineConfig
###############################################################################

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"   # DDP master 地址
export MASTER_PORT="${MASTER_PORT:-6379}"        # DDP master 端口
echo "MASTER_ADDR=${MASTER_ADDR}, MASTER_PORT=${MASTER_PORT}"

PROC_PER_NODE="${PROC_PER_NODE:-1}"              # 每节点 GPU 进程数 → 传给 accelerate --num_processes
NODE_COUNT="${NODE_COUNT:-1}"                    # 节点总数 → accelerate --num_machines
NODE_RANK="${NODE_RANK:-0}"                      # 当前节点 rank → accelerate --machine_rank
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))    # 总进程数 = 节点数 × 每节点进程数

# CUDA / 动态库路径：运行时依赖，不进入 TrainPipelineConfig
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export CUDA_HOME
export LD_LIBRARY_PATH="${CUDA_HOME:+${CUDA_HOME}/lib64:}${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CONDA_PREFIX:+${CONDA_PREFIX}/lib:}${LD_LIBRARY_PATH}"

export PYTHONUNBUFFERED=1                        # Python 日志实时输出
export OMP_NUM_THREADS=1                         # 避免 DataLoader 多线程与 OpenMP 争抢 CPU
export MKL_NUM_THREADS=1

# WANDB_MODE 最终会传给 TrainPipelineConfig.wandb.mode（见 ARGS 末尾）
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"     # HuggingFace Hub 是否离线
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

###############################################################################
############################## TRAINING config ################################
# 以下段落：准备 TrainPipelineConfig 的输入（shell 变量 → draccus CLI 参数）
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
echo "SCRIPT_DIR = ${SCRIPT_DIR}"
echo "PROJ_ROOT  = ${PROJ_ROOT}"

# 让 Python 能 import lerobot.*；不进入 TrainPipelineConfig
export PYTHONPATH="${PROJ_ROOT}/src:${PYTHONPATH:-}"
cd "${PROJ_ROOT}"

if (( $# > 3 )); then
  echo "Usage:"
  echo "  bash launch/tbot_sa1_finetune.sh DATASET_REPO_ID [ACTION_TYPE] [USE_EXTERNAL_STATS]"
  echo "  DATASET_REPO_ID=/path/or/hf_repo POLICY_INIT_PATH=/path/to/bootstrap bash launch/tbot_sa1_finetune.sh"
  exit 1
fi

# ---------------------------------------------------------------------------
# Policy 类型选择
# POLICY → draccus 解析时用于：
#   --policy.type="${POLICY}"   → TBotSA1Config.type == "TBot_SA1"
#   --dataset.type="${POLICY}"  → TBotSA1DatasetConfig.type == "TBot_SA1"
# 决定 factory.make_policy() 实例化 TBotSA1Policy，
#         factory.make_dataset() 实例化 TBotSA1 数据管线
# ---------------------------------------------------------------------------
POLICY="${POLICY:-TBot_SA1}"

# POLICY_INIT_PATH → --policy.pretrained_path
# 对应 TBotSA1Config.pretrained_path (PreTrainedConfig 基类字段)
# make_policy() 会调用 TBotSA1Policy.from_pretrained(pretrained_name_or_path=...)
# 从 bootstrap checkpoint 加载权重（如 tbot_base / zaleni/TBot-SA1-Base）
POLICY_INIT_PATH="${POLICY_INIT_PATH:-${PRETRAINED_PATH:-}}"

# 以下路径字段均属于 TBotSA1Config（policy 侧），用于构建 TBotSA1Policy 内部子模块
QWEN3_VL_PRETRAINED_PATH="${QWEN3_VL_PRETRAINED_PATH:-Qwen/Qwen3-VL-2B-Instruct}"
# → --policy.qwen3_vl_pretrained_path → TBotSA1Config.qwen3_vl_pretrained_path
#   加载 Qwen3-VL 理解专家 (und_expert)

QWEN3_VL_PROCESSOR_PATH="${QWEN3_VL_PROCESSOR_PATH:-${QWEN3_VL_PRETRAINED_PATH}}"
# → --dataset.qwen3_vl_processor_path → TBotSA1DatasetConfig.qwen3_vl_processor_path
#   数据 transform 中 Qwen3_VLProcessorTransformFn 使用的 processor 路径

COSMOS_TOKENIZER_PATH_OR_NAME="${COSMOS_TOKENIZER_PATH_OR_NAME:-nvidia/Cosmos-Tokenizer-CI8x8}"
# → --policy.cosmos_tokenizer_path_or_name → TBotSA1Config.cosmos_tokenizer_path_or_name
#   Cosmos 离散图像 tokenizer，用于生成 (gen) 分支

DA3_MODEL_PATH_OR_NAME="${DA3_MODEL_PATH_OR_NAME:-depth-anything/DA3-LARGE-1.1}"
# → --policy.da3_model_path_or_name → TBotSA1Config.da3_model_path_or_name
#   Depth-Anything-3 teacher，用于 3D 对齐蒸馏

DA3_VARIANT="${DA3_VARIANT:-auto}"
# → --policy.da3_variant → TBotSA1Config.da3_variant

DA3_ALIGNMENT_MODE="${DA3_ALIGNMENT_MODE:-query_decoder}"
# → --policy.da3_alignment_mode → TBotSA1Config.da3_alignment_mode

DA3_CODE_ROOT="${DA3_CODE_ROOT:-}"
# → --policy.da3_code_root → TBotSA1Config.da3_code_root（可选）
#   本地 Depth-Anything-3 源码根目录；为空则不传该 CLI 参数

# ---------------------------------------------------------------------------
# Dataset 输入（最终会进入 TBotSA1DatasetConfig）
# ---------------------------------------------------------------------------
DATASET_REPO_ID="${1:-${DATASET_REPO_ID:-}}"
# → --dataset.repo_id → TBotSA1DatasetConfig.repo_id
#   LeRobot 数据集本地路径或 HuggingFace repo id

ACTION_TYPE="${2:-${ACTION_TYPE:-delta}}"
# → --dataset.action_mode → TBotSA1DatasetConfig.action_mode ("delta" | "abs")
#   控制是否在 data_transforms 中插入 DeltaActionTransformFn

POSITIONAL_USE_EXTERNAL_STATS="${3:-}"

if [[ -z "${DATASET_REPO_ID}" ]]; then
  echo "Please provide DATASET_REPO_ID as the first argument or environment variable."
  exit 1
fi

if [[ "${ACTION_TYPE}" != "delta" && "${ACTION_TYPE}" != "abs" ]]; then
  echo "ACTION_TYPE must be abs or delta, got ${ACTION_TYPE}"
  exit 1
fi

# USE_EXTERNAL_STATS → --dataset.use_external_stats → TBotSA1DatasetConfig.use_external_stats
# 为 true 时，make_dataset() 会从 external_stats_path 或 external_stats_root 加载归一化统计量
if [[ -n "${POSITIONAL_USE_EXTERNAL_STATS}" ]]; then
  USE_EXTERNAL_STATS="${POSITIONAL_USE_EXTERNAL_STATS}"
elif [[ -z "${USE_EXTERNAL_STATS+x}" ]]; then
  if [[ "${ACTION_TYPE}" == "delta" ]]; then
    USE_EXTERNAL_STATS=true
  else
    USE_EXTERNAL_STATS=false
  fi
fi

case "${USE_EXTERNAL_STATS}" in
  true|false)
    ;;
  *)
    echo "USE_EXTERNAL_STATS must be true or false, got ${USE_EXTERNAL_STATS}"
    exit 1
    ;;
esac

if [[ -z "${POLICY_INIT_PATH}" ]]; then
  echo "Please set POLICY_INIT_PATH to the TBot_SA1 bootstrap checkpoint."
  echo "For backward compatibility, PRETRAINED_PATH is also accepted."
  exit 1
fi

# DATASET_NAME：纯 shell 辅助变量，用于拼 job_name 和 external_stats_path
# 不直接进入 TrainPipelineConfig（除非间接体现在 output_dir / external_stats_path 中）
if [[ -z "${DATASET_NAME:-}" ]]; then
  if [[ -e "${DATASET_REPO_ID}" ]]; then
    DATASET_NAME="$(basename "${DATASET_REPO_ID}")"
  else
    DATASET_NAME="${DATASET_REPO_ID//[\/ ]/_}"
  fi
fi

# ---------------------------------------------------------------------------
# TBotSA1Config 字段（policy 侧模型/训练超参）
# ---------------------------------------------------------------------------
CHUNK_SIZE="${CHUNK_SIZE:-50}"
# → --policy.chunk_size → TBotSA1Config.chunk_size
#   动作 chunk 长度（flow matching 预测 horizon）

N_ACTION_STEPS="${N_ACTION_STEPS:-${CHUNK_SIZE}}"
# → --policy.n_action_steps → TBotSA1Config.n_action_steps
#   推理/执行时使用的 action 步数，需 <= chunk_size

TBOT_SA1_ATTENTION_MASK_MODE="${TBOT_SA1_ATTENTION_MASK_MODE:-causal}"
# → --policy.attention_mask_mode → TBotSA1Config.attention_mask_mode
#   "causal" 要求 enable_3d_queries=true（见 TBotSA1Config.__post_init__）

ENABLE_3D_QUERIES="${ENABLE_3D_QUERIES:-true}"
# → --policy.enable_3d_queries → TBotSA1Config.enable_3d_queries

NUM_3D_QUERY_TOKENS="${NUM_3D_QUERY_TOKENS:-432}"
# → --policy.num_3d_query_tokens → TBotSA1Config.num_3d_query_tokens

LAMBDA_3D="${LAMBDA_3D:-0.01}"
# → --policy.lambda_3d → TBotSA1Config.lambda_3d
#   3D 蒸馏 loss 权重

# ---------------------------------------------------------------------------
# TBotSA1DatasetConfig 字段（dataset 侧归一化 stats 路径）
# ---------------------------------------------------------------------------
NORM_STATS_ROOT="${NORM_STATS_ROOT:-norm_stats}"
DATASET_EXTERNAL_STATS_ROOT="${DATASET_EXTERNAL_STATS_ROOT:-}"
# → --dataset.external_stats_root → TBotSA1DatasetConfig.external_stats_root（可选）

DATASET_EXTERNAL_STATS_PATH="${DATASET_EXTERNAL_STATS_PATH:-}"
# → --dataset.external_stats_path → TBotSA1DatasetConfig.external_stats_path（可选）
# shell 自动推导路径（非 TrainPipelineConfig 逻辑，是 launch 脚本便利功能）：
if [[ "${USE_EXTERNAL_STATS}" == "true" && -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
  DATASET_EXTERNAL_STATS_PATH="${NORM_STATS_ROOT}/${ACTION_TYPE}/${DATASET_NAME}/stats.json"
fi

# 启动前校验：确保 stats 文件存在（shell 层 guard，TrainPipelineConfig 不负责此检查）
if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
  if [[ -z "${DATASET_EXTERNAL_STATS_PATH}" && -z "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    echo "USE_EXTERNAL_STATS=true but neither DATASET_EXTERNAL_STATS_PATH nor DATASET_EXTERNAL_STATS_ROOT is set."
    exit 1
  fi
  if [[ -n "${DATASET_EXTERNAL_STATS_PATH}" && ! -f "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    echo "Missing external stats: ${DATASET_EXTERNAL_STATS_PATH}"
    echo "Compute them first with tools/compute_norm_stats_single.py or set DATASET_EXTERNAL_STATS_PATH."
    exit 1
  fi
fi

# ---------------------------------------------------------------------------
# TBotSA1DatasetConfig.image_transforms（ImageTransformsConfig 嵌套字段）
# ---------------------------------------------------------------------------
ENABLE_IMAGE_AUG="${ENABLE_IMAGE_AUG:-false}"
# → --dataset.image_transforms.enable → ImageTransformsConfig.enable

IMAGE_AUG_PRESET="${IMAGE_AUG_PRESET:-pi05}"
# → --dataset.image_transforms.preset → ImageTransformsConfig.preset
#   为 "pi05" 时，TBotSA1DatasetConfig.__post_init__ 会插入 Pi05ImageAugmentFn

# ---------------------------------------------------------------------------
# TrainPipelineConfig 顶层训练超参
# ---------------------------------------------------------------------------
BATCH_SIZE="${BATCH_SIZE:-8}"
# → --batch_size → TrainPipelineConfig.batch_size

STEPS="${STEPS:-200}"
# → --steps → TrainPipelineConfig.steps
# 注意：本 _correct 脚本默认 200 步，适合做 smoke test；正式微调常用 30000

SAVE_FREQ="${SAVE_FREQ:-100}"
# → --save_freq → TrainPipelineConfig.save_freq

LOG_FREQ="${LOG_FREQ:-50}"
# → --log_freq → TrainPipelineConfig.log_freq

NUM_WORKERS="${NUM_WORKERS:-12}"
# → --num_workers → TrainPipelineConfig.num_workers
#   PyTorch DataLoader worker 数量

# ---------------------------------------------------------------------------
# TBotSA1Config 中的 optimizer/scheduler 预设参数
# （validate() 后复制到 TrainPipelineConfig.optimizer / .scheduler）
# ---------------------------------------------------------------------------
OPTIMIZER_LR="${OPTIMIZER_LR:-5.0e-5}"
# → --policy.optimizer_lr → TBotSA1Config.optimizer_lr
#   → get_optimizer_preset() → AdamWConfig.lr → TrainPipelineConfig.optimizer.lr

SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-20}"
# → --policy.scheduler_warmup_steps → TBotSA1Config.scheduler_warmup_steps
#   → get_scheduler_preset() → CosineDecayWithWarmupSchedulerConfig.num_warmup_steps

SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-${STEPS}}"
# → --policy.scheduler_decay_steps → TBotSA1Config.scheduler_decay_steps

SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-5.0e-6}"
# → --policy.scheduler_decay_lr → TBotSA1Config.scheduler_decay_lr

# ---------------------------------------------------------------------------
# TrainPipelineConfig 输出目录相关
# ---------------------------------------------------------------------------
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-outputs/${POLICY}}"
JOB_NAME="${JOB_NAME:-${POLICY}-${DATASET_NAME}-${ACTION_TYPE}-chunk${CHUNK_SIZE}-attn-${TBOT_SA1_ATTENTION_MASK_MODE}-finetune-$(date +'%Y_%m_%d_%H_%M_%S')}"
# → --job_name → TrainPipelineConfig.job_name

OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"
# → --output_dir → TrainPipelineConfig.output_dir
#   训练 checkpoint / train_config.json / wandb 日志的保存根目录

WANDB_PROJECT="${WANDB_PROJECT:-lerobot_lab_${POLICY}}"
# → --wandb.project → WandBConfig.project（TrainPipelineConfig.wandb 子对象）

echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "ACTION_TYPE=${ACTION_TYPE}"
echo "CHUNK_SIZE=${CHUNK_SIZE}"
echo "N_ACTION_STEPS=${N_ACTION_STEPS}"
echo "TBOT_SA1_ATTENTION_MASK_MODE=${TBOT_SA1_ATTENTION_MASK_MODE}"
echo "USE_EXTERNAL_STATS=${USE_EXTERNAL_STATS}"
echo "DATASET_EXTERNAL_STATS_PATH=${DATASET_EXTERNAL_STATS_PATH:-<unset>}"
echo "DATASET_EXTERNAL_STATS_ROOT=${DATASET_EXTERNAL_STATS_ROOT:-<unset>}"
echo "ENABLE_IMAGE_AUG=${ENABLE_IMAGE_AUG}"
echo "IMAGE_AUG_PRESET=${IMAGE_AUG_PRESET}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "STEPS=${STEPS}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# accelerate launch 参数：分布式启动配置，不属于 TrainPipelineConfig
# ---------------------------------------------------------------------------
ACCELERATE_ARGS=()
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_ARGS=(--multi_gpu)
fi

# =============================================================================
# ARGS：draccus CLI → TrainPipelineConfig 的完整映射
#
# 解析后对象结构：
#   cfg: TrainPipelineConfig
#     ├── dataset: TBotSA1DatasetConfig   (--dataset.*)
#     ├── policy:  TBotSA1Config          (--policy.*)
#     ├── wandb:   WandBConfig            (--wandb.*)
#     ├── optimizer: AdamWConfig          (由 policy preset 自动生成，默认 use_policy_training_preset=true)
#     ├── scheduler: CosineDecayWithWarmupSchedulerConfig (同上)
#     └── 顶层 scalar 字段                 (--batch_size, --steps, ...)
#
# 训练时：
#   make_dataset(cfg)  → LeRobotDataset + TBotSA1 transforms
#   make_policy(cfg.policy) → TBotSA1Policy(config=TBotSA1Config, weights from pretrained_path)
# =============================================================================
ARGS=(
    # --- accelerate 分布式参数（传给 accelerate，不是 TrainPipelineConfig）---
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"

    # --- 训练入口脚本 ---
    src/lerobot/scripts/lerobot_train.py

    # --- TrainPipelineConfig 顶层字段 ---
    --output_dir="${OUTPUT_DIR}"           # TrainPipelineConfig.output_dir: Path
    --num_workers="${NUM_WORKERS}"         # TrainPipelineConfig.num_workers: int
    --job_name="${JOB_NAME}"               # TrainPipelineConfig.job_name: str

    # --- TBotSA1Config (--policy.*) ---
    --policy.type="${POLICY}"              # TBotSA1Config.type → "TBot_SA1"（ChoiceRegistry 键）
    --policy.repo_id="lerobot_lab/${POLICY}"  # PreTrainedConfig.repo_id（Hub 推送用；push_to_hub=false 时不生效）
    --policy.pretrained_path="${POLICY_INIT_PATH}"  # PreTrainedConfig.pretrained_path → 加载 bootstrap 权重
    --policy.qwen3_vl_pretrained_path="${QWEN3_VL_PRETRAINED_PATH}"  # TBotSA1Config.qwen3_vl_pretrained_path
    --policy.cosmos_tokenizer_path_or_name="${COSMOS_TOKENIZER_PATH_OR_NAME}"  # TBotSA1Config.cosmos_tokenizer_path_or_name
    --policy.push_to_hub=false             # PreTrainedConfig.push_to_hub（基类字段，TBotSA1Config 继承）
    --policy.gradient_checkpointing=false  # TBotSA1Config.gradient_checkpointing
    --policy.dtype=bfloat16                # TBotSA1Config.dtype
    --policy.optimizer_lr="${OPTIMIZER_LR}"               # TBotSA1Config.optimizer_lr
    --policy.scheduler_warmup_steps="${SCHEDULER_WARMUP_STEPS}"  # TBotSA1Config.scheduler_warmup_steps
    --policy.scheduler_decay_steps="${SCHEDULER_DECAY_STEPS}"    # TBotSA1Config.scheduler_decay_steps
    --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"          # TBotSA1Config.scheduler_decay_lr
    --policy.freeze_vision_encoder=false   # TBotSA1Config.freeze_vision_encoder
    --policy.train_expert_only=false       # TBotSA1Config.train_expert_only
    --policy.train_vlm_only=false          # TBotSA1Config.train_vlm_only
    --policy.qwen3_vl_variant=qwen3_vl_28l       # TBotSA1Config.qwen3_vl_variant
    --policy.action_expert_variant=qwen3_28l       # TBotSA1Config.action_expert_variant
    --policy.chunk_size="${CHUNK_SIZE}"            # TBotSA1Config.chunk_size
    --policy.n_action_steps="${N_ACTION_STEPS}"  # TBotSA1Config.n_action_steps
    --policy.attention_mask_mode="${TBOT_SA1_ATTENTION_MASK_MODE}"  # TBotSA1Config.attention_mask_mode
    --policy.enable_3d_queries="${ENABLE_3D_QUERIES}"               # TBotSA1Config.enable_3d_queries
    --policy.num_3d_query_tokens="${NUM_3D_QUERY_TOKENS}"           # TBotSA1Config.num_3d_query_tokens
    --policy.lambda_3d="${LAMBDA_3D}"                               # TBotSA1Config.lambda_3d
    --policy.da3_model_path_or_name="${DA3_MODEL_PATH_OR_NAME}"     # TBotSA1Config.da3_model_path_or_name
    --policy.da3_variant="${DA3_VARIANT}"                           # TBotSA1Config.da3_variant
    --policy.da3_alignment_mode="${DA3_ALIGNMENT_MODE}"             # TBotSA1Config.da3_alignment_mode
    --policy.log_da3_teacher_timing=true   # TBotSA1Config.log_da3_teacher_timing

    # --- TBotSA1DatasetConfig (--dataset.*) ---
    --dataset.type="${POLICY}"             # TBotSA1DatasetConfig.type → "TBot_SA1"
    --dataset.repo_id="${DATASET_REPO_ID}" # TBotSA1DatasetConfig.repo_id
    --dataset.qwen3_vl_processor_path="${QWEN3_VL_PROCESSOR_PATH}"  # TBotSA1DatasetConfig.qwen3_vl_processor_path
    --dataset.action_mode="${ACTION_TYPE}" # TBotSA1DatasetConfig.action_mode
    --dataset.use_external_stats="${USE_EXTERNAL_STATS}"  # TBotSA1DatasetConfig.use_external_stats

    # --- TrainPipelineConfig 顶层字段（续）---
    --seed=42                              # TrainPipelineConfig.seed
    --batch_size="${BATCH_SIZE}"           # TrainPipelineConfig.batch_size
    --steps="${STEPS}"                     # TrainPipelineConfig.steps
    --save_freq="${SAVE_FREQ}"             # TrainPipelineConfig.save_freq
    --log_freq="${LOG_FREQ}"               # TrainPipelineConfig.log_freq

    # --- WandBConfig (--wandb.*) → TrainPipelineConfig.wandb ---
    --wandb.enable=true                    # WandBConfig.enable
    --wandb.project="${WANDB_PROJECT}"     # WandBConfig.project
    --wandb.mode="${WANDB_MODE}"           # WandBConfig.mode
)

# 可选：TBotSA1Config.da3_code_root
if [[ -n "${DA3_CODE_ROOT}" ]]; then
    ARGS+=(--policy.da3_code_root="${DA3_CODE_ROOT}")
fi

# 可选：TBotSA1DatasetConfig.external_stats_path
if [[ "${USE_EXTERNAL_STATS}" == "true" && -n "${DATASET_EXTERNAL_STATS_PATH}" ]]; then
    ARGS+=(--dataset.external_stats_path="${DATASET_EXTERNAL_STATS_PATH}")
fi

# 可选：TBotSA1DatasetConfig.external_stats_root
if [[ -n "${DATASET_EXTERNAL_STATS_ROOT}" ]]; then
    ARGS+=(--dataset.external_stats_root="${DATASET_EXTERNAL_STATS_ROOT}")
fi

# 可选：ImageTransformsConfig（TBotSA1DatasetConfig.image_transforms 子对象）
if [[ "${ENABLE_IMAGE_AUG}" == "true" ]]; then
    ARGS+=(
        --dataset.image_transforms.enable=true       # ImageTransformsConfig.enable
        --dataset.image_transforms.preset="${IMAGE_AUG_PRESET}"  # ImageTransformsConfig.preset
    )
fi

# 最终启动：
#   accelerate launch [accelerate_args] lerobot_train.py [draccus_args]
# draccus 在 @parser.wrap() 处把 ARGS 中的 --xxx 解析为 TrainPipelineConfig
accelerate launch "${ACCELERATE_ARGS[@]}" "${ARGS[@]}"
