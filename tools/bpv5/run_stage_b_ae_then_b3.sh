#!/bin/bash
# Curriculum: AE-only 2k (freeze BP/VLM/LoRA) → b3 train BP+LoRA+AE (~13k more, ~15k total).
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/run_stage_b_ae_then_b3.sh
set -uo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

OUT_BASE=/vla/workspace/my_tbot/outputs/BPv5/stage_b/ae_then_b3
LOG_DIR=$REPO/outputs/BPv5/stage_b/logs
mkdir -p "$LOG_DIR" "$OUT_BASE"
STAMP=$(date +%Y%m%d_%H%M%S)
WATCH=$LOG_DIR/watch_ae_then_b3_${STAMP}.log
ln -sfn "$(basename "$WATCH")" "$LOG_DIR/watch_ae_then_b3_latest.log"
AE_PORT=${AE_PORT:-29722}
B3_PORT=${B3_PORT:-29723}
AE_STEPS=${AE_STEPS:-2000}

log() { echo "[$(date -Is)] $*" | tee -a "$WATCH"; }

find_ckpt() {
  local base=$1 step=$2
  local d="$base/checkpoints/$(printf '%06d' "$step")/pretrained_model"
  [[ -f "$d/model.safetensors" || -f "$d/model.safetensors.index.json" ]] || return 1
  echo "$d"
}

latest_run_under() {
  ls -1d "$OUT_BASE"/20* 2>/dev/null | sort | tail -1
}

log "curriculum start AE_STEPS=$AE_STEPS then b3 (+13k) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ----- phase 1: AE only -----
before=$(latest_run_under || true)
log "phase1 ae2k launch"
FOREGROUND=1 PORT=$AE_PORT bash tools/bpv5/train_stage_b.sh ae2k
rc=$?
after=$(latest_run_under || true)
if [[ -z "$after" ]]; then
  log "ERROR: no run dir after ae2k (rc=$rc)"; exit 1
fi
echo "$after" > "$OUT_BASE/.ae_run_dir"
log "ae run dir=$after rc=$rc"

AE_CKPT=$(find_ckpt "$after" "$AE_STEPS") || {
  # tolerate last save naming
  AE_CKPT=$(find_ckpt "$after" "$AE_STEPS") || true
  if [[ -z "${AE_CKPT:-}" ]]; then
    log "ERROR: missing ae ckpt @ $AE_STEPS under $after"
    ls -la "$after/checkpoints" 2>/dev/null | tee -a "$WATCH"
    exit 1
  fi
}
log "ae ckpt=$AE_CKPT"

# ----- phase 2: b3 from ae weights (new timestamped run under same OUT_BASE) -----
log "phase2 b3 launch pretrained=$AE_CKPT"
FOREGROUND=1 PORT=$B3_PORT bash tools/bpv5/train_stage_b.sh ae2k_then_b3 \
  --policy.pretrained_path="$AE_CKPT" \
  --policy.stage_a_checkpoint=""
rc2=$?
b3_run=$(latest_run_under || true)
echo "$b3_run" > "$OUT_BASE/.b3_run_dir"
log "b3 finished rc=$rc2 run=$b3_run"
exit "$rc2"
