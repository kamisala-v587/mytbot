#!/bin/bash
# BPv5 Stage B launcher. Does not touch a running Stage A job.
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh b1
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh b2
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh b3
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh v2_short
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh v2_full
# B-2 / B-3 need policy.pretrained_path in the config (previous phase pretrained_model).
# B-1 needs policy.stage_a_checkpoint.
set -euo pipefail
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va
cd "$REPO"
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}

VARIANT=${1:-b1}
shift || true
case "$VARIANT" in
  b1) CONFIG=${CONFIG:-configs/bpv5/stage_b/v1/stage_b1.jsonc}; NAME=b1; PORT=${PORT:-29710} ;;
  b1_lora_l0) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc}; NAME=b1_lora_l0; PORT=${PORT:-29713} ;;
  pb2) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb2.jsonc}; NAME=pb2; PORT=${PORT:-29714} ;;
  pb2_1) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb2_1.jsonc}; NAME=pb2_1; PORT=${PORT:-29715} ;;
  pb3) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb3.jsonc}; NAME=pb3; PORT=${PORT:-29716} ;;
  pb3_4src) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb3_4src.jsonc}; NAME=pb3_4src; PORT=${PORT:-29717} ;;
  pb3_4src_c1) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb3_4src_c1.jsonc}; NAME=pb3_4src_c1; PORT=${PORT:-29718} ;;
  pb_rt_vobs) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb_rt_vobs.jsonc}; NAME=pb_rt_vobs; PORT=${PORT:-29719} ;;
  pb_rt_mix50k) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb_rt_mix50k.jsonc}; NAME=pb_rt_mix50k; PORT=${PORT:-29720} ;;
  pb_fp_notext30k) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_pb_fp_notext30k.jsonc}; NAME=pb_fp_notext30k; PORT=${PORT:-29721} ;;
  ae2k) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_ae2k.jsonc}; NAME=ae2k; PORT=${PORT:-29722} ;;
  ae2k_then_b3) CONFIG=${CONFIG:-configs/bpv5/stage_b/probes/stage_b_ae2k_then_b3.jsonc}; NAME=ae2k_then_b3; PORT=${PORT:-29723} ;;
  b2) CONFIG=${CONFIG:-configs/bpv5/stage_b/v1/stage_b2.jsonc}; NAME=b2; PORT=${PORT:-29711} ;;
  b3) CONFIG=${CONFIG:-configs/bpv5/stage_b/v1/stage_b3.jsonc}; NAME=b3; PORT=${PORT:-29712} ;;
  v2_short) CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/stage_b_v2_short.jsonc}; NAME=v2_short; PORT=${PORT:-29730} ;;
  v2_full) CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/stage_b_v2_full.jsonc}; NAME=v2_full; PORT=${PORT:-29731} ;;
  *)  CONFIG=$VARIANT; NAME=$(basename "$CONFIG" .jsonc); PORT=${PORT:-29710} ;;
esac
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG"; exit 1; }

if [ -n "${NUM_GPUS:-}" ]; then
  NPROC=$NUM_GPUS
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
  NPROC=${#_gpus[@]}
else
  NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
[ "${NPROC:-0}" -ge 1 ] || { echo "NPROC/NUM_GPUS empty and no GPU"; exit 1; }
MIXED_PRECISION=${MIXED_PRECISION:-no}

case "$VARIANT" in
  v2_short|v2_full) LOG_DIR=$REPO/outputs/BPv5/bv2/logs ;;
  *) LOG_DIR=$REPO/outputs/BPv5/stage_b/logs ;;
esac
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
fi
