#!/bin/bash
# Read-only AE-3 status: pipeline steps, running processes, GPUs, latest log lines (training / bench / G3 / pipeline).
# usage: bash /home/jovyan/workspace/mytbot/tools/bpv4cdx/ae3_status.sh [log lines, default 5]
N=${1:-5}
OUT=/home/jovyan/workspace/mytbot/outputs/BPv4CDX/stage_a3
LOGS=$OUT/logs
echo "===== $(date '+%F %T')  pipeline steps ($OUT/pipeline_status.txt)"
tail -n 8 "$OUT/pipeline_status.txt" 2>/dev/null
echo "===== running"
printf "pipeline: %s   cache build: %s   training (lerobot_train): %s   gates: %s\n" \
  "$(pgrep -fc 'ae3_pipelin[e].sh')" "$(pgrep -fc 'BPV4CDX.build_cac[h]e')" \
  "$(pgrep -fc 'lerobot_trai[n]')" "$(pgrep -fc 'BPV4CDX.gate_g[0-9]')"
echo "===== GPUs (index, memory used, utilization)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';'; echo
echo "===== pipeline.log (last $N lines, modified $(date -r "$LOGS/pipeline.log" '+%T' 2>/dev/null))"
tail -n "$N" "$LOGS/pipeline.log" 2>/dev/null | tr '\r' '\n' | tail -n "$N" | cut -c1-300
latest=$(ls -t "$LOGS"/train_*.log "$LOGS"/bench_bs*.log 2>/dev/null | head -1)
if [ -n "$latest" ]; then
  echo "===== $(basename "$latest") (modified $(date -r "$latest" '+%T'))"
  tr '\r' '\n' < "$latest" | grep -a -v '^\s*$' | tail -n "$N" | cut -c1-300
fi
hom=$(ls -t "$OUT"/ae3_main/*/heldout_monitor.jsonl "$OUT"/ae3_main/heldout_monitor.jsonl 2>/dev/null | head -1)
[ -n "$hom" ] && { echo "===== HOM (last entry)"; tail -n 1 "$hom" | cut -c1-600; }
