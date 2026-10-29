#!/bin/bash
# Resume a Stage B run from a saved checkpoint (weights + optimizer + scheduler + step). Runs in the background.
# usage: bash tools/bpv4cdxb/resume_b.sh <checkpoint dir, e.g. .../checkpoints/0105000> <log name> [extra draccus overrides ...]
# log: outputs/BPv4CDX/stage_b/logs/train_<log name>_<time>.log
set -u
if [ "${CUDA_VISIBLE_DEVICES-unset}" = "" ]; then
  echo "CUDA_VISIBLE_DEVICES is set but empty: no GPU would be visible; unset it first" >&2
  exit 1
fi
CKPT=$(realpath "$1")
NAME=$2
shift 2
cd /home/jovyan/workspace/mytbot || exit 1
source tools/bpv4cdx/env.b200.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=outputs/BPv4CDX/stage_b/logs
mkdir -p "$LOGS"
LOG=$LOGS/train_${NAME}_$(date +%m%d_%H%M).log
setsid nohup accelerate launch --num_processes=8 --main_process_port 29810 -m lerobot.scripts.lerobot_train \
  --config_path="$CKPT/pretrained_model/train_config.json" --resume=true "$@" > "$LOG" 2>&1 < /dev/null &
echo "resumed $CKPT -> $LOG"
