#!/bin/bash
# Compact G9 on every new checkpoint of a Stage B training run (StageB实施.md). Polls every 5 minutes; a checkpoint is
# evaluated once its train_config.json exists and has not changed for 2 minutes; exits when no training process is
# left and every checkpoint has a result. Same query sample (seed 0) as the B-2 evaluations, so results compare.
# usage: bash tools/bpv4cdxb/eval_watch.sh <run output dir> <result prefix> [gpu=7] [queries=300] [every=5000]
# every: only checkpoints whose step is a multiple of it are evaluated
# results: outputs/BPv4CDX/stage_b/g9/<prefix>_<step>.json   log: outputs/BPv4CDX/stage_b/logs/eval_watch_<prefix>.log
set -u
RUN=$1
PREFIX=$2
GPU=${3:-7}
QUERIES=${4:-300}
EVERY=${5:-5000}
cd /home/jovyan/workspace/mytbot || exit 1
source tools/bpv4cdx/env.b200.sh
export OMP_NUM_THREADS=4
OUT=outputs/BPv4CDX/stage_b/g9
LOGS=outputs/BPv4CDX/stage_b/logs
mkdir -p "$OUT" "$LOGS"
say() { echo "$(date '+%F %T') $*"; }
say "watching $RUN (prefix $PREFIX, GPU $GPU, $QUERIES queries x 2 steps, every $EVERY steps)"
while true; do
  pending=0
  for ck in $(ls -d "$RUN"/*/checkpoints/[0-9]*/pretrained_model 2>/dev/null | sort); do
    step=$(basename "$(dirname "$ck")")
    name=${PREFIX}_${step}
    [ $(( 10#$step % EVERY )) -ne 0 ] && continue  # 10#: step dirs have leading zeros (else read as octal)
    [ -f "$OUT/$name.json" ] && continue
    if [ ! -f "$ck/train_config.json" ] || [ $(( $(date +%s) - $(date -r "$ck/train_config.json" +%s) )) -lt 120 ]; then
      pending=1
      continue
    fi
    say "G9 $name"
    CUDA_VISIBLE_DEVICES=$GPU python -m lerobot.policies.BPV4CDXB.gate_g9_main --ckpt "$ck" --out "$OUT" --name "$name" \
      --queries "$QUERIES" --workers 16 --batch 16 > "$LOGS/g9_$name.log" 2>&1
    say "G9 $name exit=$? -> $(tail -1 "$LOGS/g9_$name.log" | cut -c1-200)"
  done
  if ! pgrep -f "lerobot_trai[n]" > /dev/null && [ $pending -eq 0 ]; then
    say "no training process and nothing pending: done"
    exit 0
  fi
  sleep 300
done
