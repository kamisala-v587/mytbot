#!/bin/bash
# B-2 training on the B200 main path (StageB实施.md S5). Runs in the background; log: outputs/BPv4CDX/stage_b/logs/train_<name>_<time>.log
# usage: bash tools/bpv4cdxb/train_b2.sh [config name, default b2_main] [extra draccus overrides ...]
# MIXED_PRECISION=bf16 (accelerate autocast) is required when part of the model is fp32, e.g. B-5 with vlm_trainable
set -u
NAME=${1:-b2_main}
[ $# -gt 0 ] && shift
cd /home/jovyan/workspace/mytbot || exit 1
source tools/bpv4cdx/env.b200.sh
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=outputs/BPv4CDX/stage_b/logs
mkdir -p "$LOGS"
LOG=$LOGS/train_${NAME}_$(date +%m%d_%H%M).log
setsid nohup accelerate launch --num_processes=8 --main_process_port 29810 --mixed_precision="${MIXED_PRECISION:-no}" \
  -m lerobot.scripts.lerobot_train \
  --config_path=configs/Pro6k/bpv4cdxb_plan/generated_b200/$NAME.json "$@" > "$LOG" 2>&1 < /dev/null &
echo "started $NAME -> $LOG"
