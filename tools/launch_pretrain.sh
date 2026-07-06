#!/usr/bin/env bash
# Memory-safe launcher for TBot-SA1 multi-source pretraining (2016 repos).
#
# Why: 8 ranks × ~252 LeRobotDataset + 128 DataLoader workers can spike host RAM
# and take down the IDE. Run this inside tmux, not the Cursor integrated terminal.
#
# Usage:
#   tmux new -s tbot
#   cd /home/jovyan/vla/workspace/mytbot
#   bash tools/launch_pretrain.sh
#
# Optional env:
#   NUM_PROCESSES=8
#   CONFIG_PATH=.config/pretrain_config.jsonc
#   PARALLEL_DATASET_LOAD=0   # 0=safer startup (default), 1=faster but higher RAM peak

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYTBOT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${MYTBOT_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${MYTBOT_ROOT}/.config/pretrain_config.jsonc}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD:-0}"

export PYTHONPATH="${MYTBOT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Safer defaults for full pretrain_data.txt (2016 repos)
export LEROBOT_PARALLEL_DATASET_LOAD="${PARALLEL_DATASET_LOAD}"
export LEROBOT_DDP_TIMEOUT_SEC="${LEROBOT_DDP_TIMEOUT_SEC:-7200}"

# Limit CPU thread explosion (each rank also spawns num_workers)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

if [[ "${PARALLEL_DATASET_LOAD}" == "1" ]]; then
  echo "[WARN] LEROBOT_PARALLEL_DATASET_LOAD=1: faster dataset init, higher RAM peak."
else
  echo "[INFO] LEROBOT_PARALLEL_DATASET_LOAD=0: rank0 loads first, lower RAM peak."
fi

echo "CONFIG_PATH=${CONFIG_PATH}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
accelerate launch --num_processes="${NUM_PROCESSES}" \
  -m lerobot.scripts.lerobot_train \
  --config_path="${CONFIG_PATH}"
