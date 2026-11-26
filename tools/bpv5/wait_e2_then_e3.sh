#!/bin/bash
# Wait until E2 training fully exits, free GPUs, then launch E3.
#   bash tools/bpv5/wait_e2_then_e3.sh
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1
LOG_DIR=$REPO/outputs/BPv5/stage_a/logs
mkdir -p "$LOG_DIR"
WAIT_LOG=$LOG_DIR/wait_e2_then_e3_$(date +%Y%m%d_%H%M%S).log
ln -sfn "$(basename "$WAIT_LOG")" "$LOG_DIR/wait_e2_then_e3_latest.log"

E2_LOSS=$REPO/outputs/BPv5/stage_a/e2/2026-11-16_19-59/loss.log
E2_STEPS_TARGET=${E2_STEPS_TARGET:-2000}
POLL_SEC=${POLL_SEC:-30}

log() { echo "[$(date '+%F %T')] $*" | tee -a "$WAIT_LOG"; }

e2_alive() {
  pgrep -f 'lerobot_train --config_path=configs/bpv5/stage_a_e2' >/dev/null \
    || pgrep -f 'accelerate launch .*stage_a_e2' >/dev/null
}

e2_last_step() {
  python3 - <<'PY' 2>/dev/null
import csv
from pathlib import Path
p=Path("/vla/workspace/my_tbot/outputs/BPv5/stage_a/e2/2026-11-16_19-59/loss.log")
# fallback: newest non-WRONG e2 loss
if not p.is_file():
  cands=sorted(Path("/vla/workspace/my_tbot/outputs/BPv5/stage_a/e2").glob("*/loss.log"))
  cands=[c for c in cands if "WRONG" not in str(c) and "OOM" not in str(c)]
  p=cands[-1] if cands else None
if not p or not p.is_file():
  print(0); raise SystemExit
rows=list(csv.DictReader(p.open()))
print(int(float(rows[-1]["step"])) if rows else 0)
PY
}

gpus_busy() {
  # any non-zero used memory
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{if($1+0>500) found=1} END{exit !found}'
}

log "waiter start; target E2 steps>=$E2_STEPS_TARGET then launch E3"
log "wait log: $WAIT_LOG"

while true; do
  step=$(e2_last_step)
  if e2_alive; then
    log "E2 running step≈$step / $E2_STEPS_TARGET"
  else
    log "E2 process gone; last_step≈$step"
    # require reaching target (or accept if already past) and GPUs free
    if [ "${step:-0}" -ge "$E2_STEPS_TARGET" ]; then
      break
    fi
    # died early — do not auto-start E3
    log "ERROR: E2 exited before step $E2_STEPS_TARGET (last=$step); not launching E3"
    exit 2
  fi
  sleep "$POLL_SEC"
done

log "E2 finished (step>=$E2_STEPS_TARGET). Waiting for GPUs to free..."
for i in $(seq 1 60); do
  if ! gpus_busy && ! e2_alive; then
    break
  fi
  # hard-kill stragglers after a bit
  if [ "$i" -eq 10 ]; then
    log "killing leftover e2 procs if any"
    pkill -9 -f 'lerobot_train --config_path=configs/bpv5/stage_a_e2' 2>/dev/null || true
    pkill -9 -f 'accelerate launch .*stage_a_e2' 2>/dev/null || true
  fi
  sleep 5
done

if gpus_busy; then
  log "ERROR: GPUs still busy after wait; not launching E3"
  nvidia-smi | tee -a "$WAIT_LOG"
  exit 3
fi

log "GPUs free. Launching E3..."
export FOREGROUND=0
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PORT=${PORT:-29672}
bash tools/bpv5/train_stage_a.sh e3 2>&1 | tee -a "$WAIT_LOG"
log "E3 launch issued. tail -f $LOG_DIR/train_e3_latest.log"
exit 0
