#!/bin/bash
# Wait until b1_lora_l0 reaches TARGET_STEP, then write formal comparison report.
#   nohup bash tools/bpv5/wait_lora_l0_10k_then_report.sh &
set -euo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"
Q35=/vla/workspace/shiliangchao/env/q35va
TARGET_STEP=${TARGET_STEP:-10000}
POLL_SEC=${POLL_SEC:-60}
BASELINE=${BASELINE:-$REPO/outputs/BPv5/stage_b/b1/2026-11-19_02-10}
LOG_DIR=$REPO/outputs/BPv5/stage_b/logs
mkdir -p "$LOG_DIR"
WAIT_LOG=$LOG_DIR/wait_lora_l0_report_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$WAIT_LOG") 2>&1
ln -sfn "$(basename "$WAIT_LOG")" "$LOG_DIR/wait_lora_l0_report_latest.log"

log() { echo "[$(date '+%F %T')] $*"; }

pick_lora_loss() {
  if [ -n "${LORA_LOSS:-}" ] && [ -f "$LORA_LOSS" ]; then
    echo "$LORA_LOSS"; return
  fi
  ls -td "$REPO"/outputs/BPv5/stage_b/b1_lora_l0/*/loss.log 2>/dev/null | head -1 || true
}

latest_step() {
  local f=$1
  [ -f "$f" ] || { echo 0; return; }
  awk -F, 'NR>1 {s=$4} END {if (s=="") print 0; else print int(s)}' "$f"
}

log "waiter start: LoRA-L0 step>=$TARGET_STEP then report vs baseline $BASELINE"

# wait until loss.log appears
LOSS=""
while true; do
  LOSS=$(pick_lora_loss)
  if [ -n "$LOSS" ]; then
    break
  fi
  if ! pgrep -f 'configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc' >/dev/null; then
    log "ERROR: LoRA train process gone and no loss.log yet"
    exit 2
  fi
  log "waiting for b1_lora_l0/loss.log ..."
  sleep "$POLL_SEC"
done
log "loss_log=$LOSS"

while true; do
  step=$(latest_step "$LOSS")
  log "lora_l0 step=$step / $TARGET_STEP"
  if [ "$step" -ge "$TARGET_STEP" ]; then
    break
  fi
  if ! pgrep -f 'configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc' >/dev/null; then
    log "ERROR: LoRA train exited early at step=$step"
    exit 2
  fi
  sleep "$POLL_SEC"
done

# HOM is written at save_freq (5000); wait briefly for 10000 ckpt HOM if needed
for i in $(seq 1 30); do
  run=$(dirname "$LOSS")
  if [ -f "$run/heldout_monitor.jsonl" ] && grep -q '"step": 10000' "$run/heldout_monitor.jsonl" 2>/dev/null; then
    break
  fi
  # also accept step==10000 without spaces after colon variants
  if [ -f "$run/heldout_monitor.jsonl" ] && grep -Eq '"step":[[:space:]]*10000' "$run/heldout_monitor.jsonl"; then
    break
  fi
  log "waiting for HOM @10000 (try $i) ..."
  sleep 20
done

log "generating report"
"$Q35/bin/python" tools/bpv5/report_b1_vs_lora_l0.py \
  --baseline "$BASELINE" \
  --lora "$(dirname "$LOSS")" \
  --step "$TARGET_STEP"

log "done. see outputs/BPv5/stage_b/reports/b1_vs_lora_l0_10k.md"
