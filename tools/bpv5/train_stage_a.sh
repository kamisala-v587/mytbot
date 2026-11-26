#!/bin/bash
# BPv5 Stage A single-node train launcher (local / smoke).
# usage:
#   CUDA_VISIBLE_DEVICES=0,1 bash tools/bpv5/train_stage_a.sh local2
#   CUDA_VISIBLE_DEVICES=0,1 bash tools/bpv5/train_stage_a.sh main
#   CUDA_VISIBLE_DEVICES=0,1 bash tools/bpv5/train_stage_a.sh smoke
#   bash tools/bpv5/train_stage_a.sh local2 --batch_size=2
# env: NUM_GPUS / NPROC, PORT, MIXED_PRECISION, FOREGROUND=1, CONFIG override
#
# Multi-node: use tools/bpv5/train_stage_a_dist.sh instead.
#
# Logs: outputs/BPv5/stage_a/logs/train_<name>_<stamp>.log
#   tail -f $(ls -t outputs/BPv5/stage_a/logs/train_local2_*.log | head -1)
#   # or:
#   tail -f outputs/BPv5/stage_a/logs/train_local2_latest.log
set -euo pipefail
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va
cd "$REPO"
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

VARIANT=${1:-local2}
shift || true
case "$VARIANT" in
  # 正式主训在 configs/bpv5/stage_a/；历史探针见 configs/bpv5/historyOrTestOrEx/<日期>/
  smoke)  CONFIG=${CONFIG:-configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_smoke.jsonc}; NAME=smoke ;;
  local2) CONFIG=${CONFIG:-configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_local2.jsonc}; NAME=local2 ;;
  main)   CONFIG=${CONFIG:-configs/bpv5/stage_a/stage_a_main.jsonc}; NAME=main ;;
  dist8)  CONFIG=${CONFIG:-configs/bpv5/stage_a/stage_a_main_dist8.jsonc}; NAME=dist8; PORT=${PORT:-29700} ;;
  e1)     CONFIG=${CONFIG:-configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_e1.jsonc}; NAME=e1; PORT=${PORT:-29670} ;;
  e2)     CONFIG=${CONFIG:-configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_e2.jsonc}; NAME=e2; PORT=${PORT:-29671} ;;
  e3)     CONFIG=${CONFIG:-configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_e3.jsonc}; NAME=e3; PORT=${PORT:-29672} ;;
  *)      CONFIG=$VARIANT; NAME=$(basename "$CONFIG" .jsonc) ;;
esac
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG"; exit 1; }
# Keep env aligned with config.data_root when unset
if [ "$NAME" = "e1" ]; then
  export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_rt20_clean_root}
elif [ "$NAME" = "e2" ]; then
  export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_rt80_clean_root}
  export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/physical-ai-data/norm_stats/action80_new/delta}
elif [ "$NAME" = "e3" ]; then
  export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_rt80_clean_root}
elif [ "$NAME" = "main" ] || [ "$NAME" = "stage_a_e9_main" ] || [ "$NAME" = "e9_main" ]; then
  export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
  export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}
fi

if [ -n "${NUM_GPUS:-}" ]; then
  NPROC=$NUM_GPUS
elif [ -n "${NPROC:-}" ]; then
  :
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
  NPROC=${#_gpus[@]}
else
  NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
[ "${NPROC:-0}" -ge 1 ] || { echo "NPROC/NUM_GPUS empty and no GPU"; exit 1; }
PORT=${PORT:-29660}
MIXED_PRECISION=${MIXED_PRECISION:-no}

LOG_DIR=$REPO/outputs/BPv5/stage_a/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/train_${NAME}_${STAMP}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/train_${NAME}_latest.log"

echo "config=$CONFIG  num_processes=$NPROC  port=$PORT  log=$LOG  extras=$*"

CMD=(
  accelerate launch
  --num_processes="$NPROC"
  --main_process_port "$PORT"
  --mixed_precision "$MIXED_PRECISION"
  -m lerobot.scripts.lerobot_train
  --config_path="$CONFIG"
)

if [ "${FOREGROUND:-0}" = "1" ]; then
  "${CMD[@]}" "$@" 2>&1 | tee "$LOG"
else
  setsid nohup "${CMD[@]}" "$@" >"$LOG" 2>&1 < /dev/null &
  echo "started pid=$! -> $LOG"
  echo "tail:  tail -f $LOG"
  echo "tail latest:  tail -f $LOG_DIR/train_${NAME}_latest.log"
fi
