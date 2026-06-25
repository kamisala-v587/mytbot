#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TBot-SA1 RoboTwin 多数据集微调启动脚本（JSON / JSONC config_path）
#
# 与 launch/tbot_sa1_finetune_robotwin.sh 等效；训练超参写在 JSONC 中。
#
# 用法：
#   cd /home/jovyan/vla/workspace/mytbot
#   bash launch/tbot_sa1_finetune_robotwin_json.sh
#   bash launch/tbot_sa1_finetune_robotwin_json.sh .配置/finetune_robotwin_config_official.jsonc
#
# 或直接 accelerate（支持 .json 与 .jsonc）：
#   accelerate launch --multi_gpu --num_processes=8 \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=/home/jovyan/vla/workspace/mytbot/.配置/finetune_robotwin_config_official.jsonc
###############################################################################

CONFIG_PATH="${1:-.配置/finetune_robotwin_config_official.jsonc}"

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6379}"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PROC_PER_NODE="${PROC_PER_NODE:-8}"
NODE_COUNT="${NODE_COUNT:-1}"
NODE_RANK="${NODE_RANK:-0}"
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

ACCELERATE_ARGS=()
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_ARGS=(--multi_gpu)
fi

exec accelerate launch "${ACCELERATE_ARGS[@]}" \
    --num_processes="${NUM_PROCESSES}" \
    --num_machines="${NODE_COUNT}" \
    --machine_rank="${NODE_RANK}" \
    --main_process_ip="${MASTER_ADDR}" \
    --main_process_port="${MASTER_PORT}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="${CONFIG_PATH}"
