#!/bin/bash
# Read-only Stage B status: running processes, GPUs, latest training log lines (step, loss, update / data time), checkpoints.
# usage: bash /home/jovyan/workspace/mytbot/tools/bpv4cdxb/status_b.sh [log lines, default 3]
N=${1:-3}
OUT=/home/jovyan/workspace/mytbot/outputs/BPv4CDX/stage_b
echo "===== $(date '+%F %T')"
printf "training processes (lerobot_train): %s   G9 evaluations: %s\n" "$(pgrep -fc 'lerobot_trai[n]')" "$(pgrep -fc 'gate_g[9]')"
echo "===== GPUs (index, memory used MiB, utilization)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
latest=$(ls -t "$OUT"/logs/train_*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  echo "===== $(basename "$latest") (modified $(date -r "$latest" '+%T'))"
  tr '\r' '\n' < "$latest" | grep -a -v '^\s*$' | tail -n "$N" | cut -c1-260
  echo "errors: $(grep -a -c Traceback "$latest")"
fi
for run in "$OUT"/b2_main*/; do
  [ -d "$run" ] || continue
  echo "===== checkpoints in $run: $(ls "$run"/checkpoints 2>/dev/null | grep -v last | tr '\n' ' ')"
done
