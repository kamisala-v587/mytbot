#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# TBot-SA1 微调启动脚本（可选跳板，不是必须）
#
# 【最简单做法】本脚本可以完全不用，在 shell 里 cd + 配好环境后直接跑：
#
#   cd /vla/my_tbot
#   conda activate tbot_sa1
#   export HF_HUB_OFFLINE=1
#   export TRANSFORMERS_OFFLINE=1
#   export TOKENIZERS_PARALLELISM=false
#   export PYTHONPATH=/vla/my_tbot/src:$PYTHONPATH   # 若未 pip install -e . 则需要
#   CUDA_VISIBLE_DEVICES=0,1 \
#   accelerate launch -m lerobot.scripts.lerobot_train \
#     --config_path=/vla/my_tbot/.配置/train_config.jsonc
#
# 上面才是「真正启动训练」的核心；本脚本只是把最后一行包一层，方便传 JSON 路径。
#
# 【本脚本用法】
#   cd /vla/my_tbot          # 需自行 cd，脚本内不再 cd
#   ...（环境变量同上）...
#   bash launch/tbot_sa1_finetune_json.sh .配置/train_config.json
#
# 【cd / PYTHONPATH 是否有必要】
#   - cd：若你已在 my_tbot 下运行，且 JSON 里相对路径（如 norm_stats/...）相对 cwd，则必须 cd；
#         脚本内默认不 cd，由你手动 cd。
#   - PYTHONPATH：仅当 conda 环境里未 pip install -e /vla/my_tbot 时需要，手动 export 即可；
#         脚本内默认不设置。
#
# 所有 TrainPipelineConfig 参数只改 JSON，不在此脚本配置。
###############################################################################

CONFIG_PATH="$1"

# ---------------------------------------------------------------------------
# 可选：由脚本自动 cd 并设置 PYTHONPATH（默认关闭；你已在 shell 里手动处理时可忽略）
#
# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# cd "${PROJ_ROOT}"
# export PYTHONPATH="${PROJ_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 多机训练：Accelerate / DDP 参数（单机多卡一般不需要，见注释）
#
# 单机多卡：CUDA_VISIBLE_DEVICES=0,1 + 下方默认 accelerate launch 即可。
# 多机多卡：才需要 MASTER_ADDR、NODE_COUNT、--num_machines 等。
#
# export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
# export MASTER_PORT="${MASTER_PORT:-6379}"
# PROC_PER_NODE="${PROC_PER_NODE:-1}"
# NODE_COUNT="${NODE_COUNT:-1}"
# NODE_RANK="${NODE_RANK:-0}"
# NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))
# ACCELERATE_ARGS=()
# if (( NUM_PROCESSES > 1 )); then ACCELERATE_ARGS=(--multi_gpu); fi
# exec accelerate launch "${ACCELERATE_ARGS[@]}" \
#     --num_processes="${NUM_PROCESSES}" \
#     --num_machines="${NODE_COUNT}" \
#     --machine_rank="${NODE_RANK}" \
#     --main_process_ip="${MASTER_ADDR}" \
#     --main_process_port="${MASTER_PORT}" \
#     -m lerobot.scripts.lerobot_train \
#     --config_path="${CONFIG_PATH}"
# ---------------------------------------------------------------------------

exec accelerate launch -m lerobot.scripts.lerobot_train \
    --config_path="${CONFIG_PATH}"
