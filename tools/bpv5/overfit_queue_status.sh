#!/usr/bin/env bash
# Quick status for the unattended overfit queue.
set -euo pipefail
REPO=/vla/workspace/my_tbot
STATUS=$REPO/outputs/BPv5/stage_a/logs/overfit_queue_status.txt
PIDF=$REPO/outputs/BPv5/stage_a/logs/overfit_queue_daemon.pid
echo "=== daemon ==="
if [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; then
  echo "running pid=$(cat "$PIDF")"
else
  echo "NOT running (start: nohup bash tools/bpv5/run_overfit_queue_daemon.sh &)"
fi
echo "=== status file ==="
[ -f "$STATUS" ] && cat "$STATUS" || echo "(none yet)"
echo "=== gpus ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
echo "=== latest losses ==="
for n in rt1 rt8 idn32 agi32 rt1_alldims rt1_ae3loss agi32_alldims rt1_fullprefix rt1_ae05; do
  f=$(ls -t "$REPO/outputs/BPv5/stage_a/overfit_${n}"/*/loss.log 2>/dev/null | head -1 || true)
  if [ -n "${f:-}" ]; then
    awk -F, -v n="$n" 'NR==1{next} {l=$1;s=$4} END{printf "%-18s step=%-5s loss=%.4f\n", n,s,l}' "$f"
  else
    printf '%-18s (no loss.log)\n' "$n"
  fi
done
