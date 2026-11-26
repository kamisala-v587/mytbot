#!/bin/bash
# Resilient launcher for pb_fp_notext30k (auto-resume on crash).
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/run_pb_fp_notext30k_resilient.sh
set -uo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

OUT_BASE=/vla/workspace/my_tbot/outputs/BPv5/stage_b/pb_fp_notext30k
LOG_DIR=$REPO/outputs/BPv5/stage_b/logs
mkdir -p "$LOG_DIR" "$OUT_BASE"
STAMP=$(date +%Y%m%d_%H%M%S)
WATCH_LOG=$LOG_DIR/watch_pb_fp_notext30k_${STAMP}.log
ln -sfn "$(basename "$WATCH_LOG")" "$LOG_DIR/watch_pb_fp_notext30k_latest.log"
RUN_DIR_FILE=$OUT_BASE/.active_run_dir
TARGET_STEPS=${TARGET_STEPS:-30000}
MAX_RESTARTS=${MAX_RESTARTS:-30}
PORT=${PORT:-29721}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

log() { echo "[$(date -Is)] $*" | tee -a "$WATCH_LOG"; }

latest_ckpt_train_config() {
  local base=$1 best="" best_n=-1
  [[ -d "$base/checkpoints" ]] || return 1
  for d in "$base"/checkpoints/*/; do
    [[ -d "$d" ]] || continue
    local name step tc
    name=$(basename "$d")
    [[ "$name" =~ ^[0-9]+$ ]] || continue
    step=$((10#$name))
    tc="$d/pretrained_model/train_config.json"
    [[ -f "$tc" ]] || continue
    if (( step > best_n )); then best_n=$step; best=$tc; fi
  done
  [[ -n "$best" ]] || return 1
  echo "$best"
}

training_finished() {
  local base=$1 tc step
  tc=$(latest_ckpt_train_config "$base" 2>/dev/null) || return 1
  step=$(basename "$(dirname "$(dirname "$tc")")")
  step=$((10#$step))
  (( step >= TARGET_STEPS ))
}

log "watch start CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES TARGET_STEPS=$TARGET_STEPS"
attempt=0
started_fresh=0
while (( attempt <= MAX_RESTARTS )); do
  RUN_DIR=$(cat "$RUN_DIR_FILE" 2>/dev/null || true)
  if [[ -n "$RUN_DIR" ]] && training_finished "$RUN_DIR"; then
    log "already finished: $RUN_DIR"; exit 0
  fi
  if [[ -n "$RUN_DIR" ]] && tc=$(latest_ckpt_train_config "$RUN_DIR"); then
    log "resume attempt=$attempt from $tc"
    CONFIG="$tc" FOREGROUND=1 PORT=$PORT \
      bash tools/bpv5/train_stage_b.sh pb_fp_notext30k --resume=true
    rc=$?
  else
    [[ $started_fresh -eq 1 ]] && log "no ckpt yet; retry fresh (attempt=$attempt)"
    log "fresh start attempt=$attempt"
    before=$(ls -1d "$OUT_BASE"/20* 2>/dev/null | sort | tail -1 || true)
    FOREGROUND=1 PORT=$PORT bash tools/bpv5/train_stage_b.sh pb_fp_notext30k
    rc=$?
    after=$(ls -1d "$OUT_BASE"/20* 2>/dev/null | sort | tail -1 || true)
    if [[ -n "$after" ]]; then echo "$after" > "$RUN_DIR_FILE"; log "recorded run dir $after"; fi
    started_fresh=1
  fi
  RUN_DIR=$(cat "$RUN_DIR_FILE" 2>/dev/null || true)
  if [[ -n "$RUN_DIR" ]] && training_finished "$RUN_DIR"; then
    log "training finished ok rc=$rc"; exit 0
  fi
  log "train exited rc=$rc; sleep 30s then restart (attempt=$((attempt+1))/$MAX_RESTARTS)"
  sleep 30
  attempt=$((attempt + 1))
done
log "gave up after $MAX_RESTARTS restarts"; exit 1
