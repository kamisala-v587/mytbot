#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TBot-SA1 多数据集预训练启动脚本（JSON / JSONC config_path）
#
# 用法：
#   cd /vla/my_tbot
#   export PYTHONPATH=/vla/my_tbot/src:${PYTHONPATH:-}
#   bash launch/tbot_sa1_pretrain_json.sh .配置/pretrain_config.jsonc
#
# 或直接 accelerate（支持 .json 与 .jsonc）：
#   accelerate launch --multi_gpu --num_processes=2 \
#     -m lerobot.scripts.lerobot_train \
#     --config_path=/vla/my_tbot/.配置/pretrain_config.jsonc
###############################################################################

CONFIG_PATH="${1:-.配置/pretrain_config.jsonc}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"
export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

PROC_PER_NODE="${PROC_PER_NODE:-2}"
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
    --main_process_ip="${MASTER_ADDR:-127.0.0.1}" \
    --main_process_port="${MASTER_PORT:-6379}" \
    -m lerobot.scripts.lerobot_train \
    --config_path="${CONFIG_PATH}"
