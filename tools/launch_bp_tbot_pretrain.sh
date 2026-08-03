#!/usr/bin/env bash
# Launch BP_TBot pretraining from a JSONC config.
#
# Usage:
#   cd /vla/workspace/my_tbot
#   bash tools/launch_bp_tbot_pretrain.sh
#
# Optional env:
#   CONFIG_PATH=/vla/workspace/my_tbot/configs/bp_tbot_pretrain_config.jsonc
#   NUM_PROCESSES=1
#   CUDA_VISIBLE_DEVICES=0
#   PARALLEL_DATASET_LOAD=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYTBOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${MYTBOT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${MYTBOT_ROOT}/configs/bp_tbot_pretrain_config.jsonc}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD:-0}"

export PYTHONPATH="${MYTBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export LEROBOT_PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD}"
export LEROBOT_DDP_TIMEOUT_SEC="${LEROBOT_DDP_TIMEOUT_SEC:-7200}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[ERROR] CONFIG_PATH does not exist: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "/vla/workspace/models/bp_tbot_init" ]]; then
  echo "[ERROR] /vla/workspace/models/bp_tbot_init does not exist." >&2
  echo "Run the BP_change_verify.ipynb init/save cells first, or change policy.pretrained_path." >&2
  exit 1
fi

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
echo "LEROBOT_PARALLEL_DATASET_LOAD=${LEROBOT_PARALLEL_DATASET_LOAD}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
accelerate launch --num_processes="${NUM_PROCESSES}" \
  -m lerobot.scripts.lerobot_train \
  --config_path="${CONFIG_PATH}"
