#!/bin/bash
# Overnight: when baseline B-1 hits TARGET_STEP, launch B-1 with VLM LoRA layer 0
# unfrozen (comparison). Waits for free GPUs before launch (this pod is 4-GPU;
# baseline usually occupies all of them — keep polling until NPROC free cards exist,
# e.g. after you attach another pod / baseline exits).
#
#   nohup bash tools/bpv5/wait_b1_10k_then_lora_l0.sh &
#   tail -f outputs/BPv5/stage_b/logs/wait_b1_lora_l0_latest.log
#
# Env:
#   BASELINE_LOSS   default: newest outputs/BPv5/stage_b/b1/*/loss.log
#   TARGET_STEP     default 10000
#   NEED_GPUS       default 4
#   FREE_MIB_MAX    treat GPU free if memory.used <= this (default 1500)
#   CUDA_VISIBLE_DEVICES  if set, skip auto-pick and use as-is once free enough
#   POLL_SEC        default 60
#   PORT            default 29713
set -euo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"

TARGET_STEP=${TARGET_STEP:-10000}
POLL_SEC=${POLL_SEC:-60}
NEED_GPUS=${NEED_GPUS:-4}
FREE_MIB_MAX=${FREE_MIB_MAX:-1500}
PORT=${PORT:-29713}

LOG_DIR=$REPO/outputs/BPv5/stage_b/logs
mkdir -p "$LOG_DIR"
WAIT_LOG=$LOG_DIR/wait_b1_lora_l0_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$WAIT_LOG") 2>&1
ln -sfn "$(basename "$WAIT_LOG")" "$LOG_DIR/wait_b1_lora_l0_latest.log"

log() { echo "[$(date '+%F %T')] $*"; }

pick_baseline_loss() {
  if [ -n "${BASELINE_LOSS:-}" ] && [ -f "$BASELINE_LOSS" ]; then
    echo "$BASELINE_LOSS"
    return
  fi
  local newest
  newest=$(ls -td "$REPO"/outputs/BPv5/stage_b/b1/*/loss.log 2>/dev/null | head -1 || true)
  if [ -z "$newest" ]; then
    log "ERROR: no baseline B-1 loss.log under outputs/BPv5/stage_b/b1/"
    exit 1
  fi
  echo "$newest"
}

latest_step() {
  local f=$1
  [ -f "$f" ] || { echo 0; return; }
  awk -F, 'NR>1 {s=$4} END {if (s=="") print 0; else print int(s)}' "$f"
}

# Print free GPU indices (memory.used <= FREE_MIB_MAX), space-separated.
free_gpu_list() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' -v lim="$FREE_MIB_MAX" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0<=lim) print $1}' \
    | tr '\n' ' '
}

count_words() { echo $#; }

LOSS=$(pick_baseline_loss)
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
log "waiter start: target baseline step>=$TARGET_STEP then launch b1_lora_l0"
log "  loss_log=$LOSS"
log "  pod_gpus=$NGPU need_gpus=$NEED_GPUS free_mib_max=$FREE_MIB_MAX port=$PORT"
log "  config=configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc (lora_train_layer_ids=[0])"
if [ "${NGPU:-0}" -lt "$NEED_GPUS" ]; then
  log "WARN: pod only has $NGPU GPUs (<$NEED_GPUS). Will wait for free cards; attach more GPUs or lower NEED_GPUS to launch."
fi

# --- phase 1: baseline steps ---
while true; do
  step=$(latest_step "$LOSS")
  log "baseline step=$step / $TARGET_STEP"
  if [ "$step" -ge "$TARGET_STEP" ]; then
    break
  fi
  sleep "$POLL_SEC"
done
log "baseline reached $TARGET_STEP"

READY=$LOG_DIR/b1_lora_l0_READY_at_$(date +%Y%m%d_%H%M%S)
echo "baseline_step=$(latest_step "$LOSS")" >"$READY"
ln -sfn "$(basename "$READY")" "$LOG_DIR/b1_lora_l0_READY"
log "wrote $READY"

# --- phase 2: free GPUs ---
while true; do
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # User pinned devices: require those indices to be free.
    ok=1
    IFS=',' read -r -a want <<< "$CUDA_VISIBLE_DEVICES"
    used_csv=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
    for g in "${want[@]}"; do
      g=${g// /}
      u=$(echo "$used_csv" | awk -F',' -v idx="$g" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if($1==idx) print $2+0}')
      if [ -z "$u" ] || [ "$u" -gt "$FREE_MIB_MAX" ]; then
        ok=0
        break
      fi
    done
    if [ "$ok" -eq 1 ] && [ "${#want[@]}" -ge "$NEED_GPUS" ]; then
      GPUS="$CUDA_VISIBLE_DEVICES"
      break
    fi
    log "waiting for pinned CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES to free (used<=${FREE_MIB_MAX}MiB)"
  else
    free=$(free_gpu_list)
    nfree=$(count_words $free)
    log "free_gpus=[$free] count=$nfree / need=$NEED_GPUS"
    if [ "$nfree" -ge "$NEED_GPUS" ]; then
      # take first NEED_GPUS
      GPUS=$(echo $free | awk -v n="$NEED_GPUS" '{for(i=1;i<=n;i++){printf "%s%s", $i, (i<n?",":"")}}')
      break
    fi
  fi
  sleep "$POLL_SEC"
done

log "launching comparison B-1 LoRA-L0 on CUDA_VISIBLE_DEVICES=$GPUS"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PORT
FOREGROUND=0 bash tools/bpv5/train_stage_b.sh b1_lora_l0

log "launched. monitor:"
log "  tail -f $LOG_DIR/train_b1_lora_l0_latest.log"
log "  ls -lt $REPO/outputs/BPv5/stage_b/b1_lora_l0/"
