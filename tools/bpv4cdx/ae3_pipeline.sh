#!/bin/bash
# Unattended AE-3 run after the cache build was started (plan 6.2 steps 2 - 5):
#   wait for `ae3.sh cache` -> resume it once if < 99% done -> G1 / G2 -> G3 (overfit 1000 steps) -> bench 16 32 64
#   -> batch = highest trajectories/s with peak GPU memory < 85% (fallback 16; learning rate unchanged) -> train ae3_main
# Stops (no training) when G1 / G2 or G3 does not pass. Progress: $OUT/logs/pipeline.log, one line per step in
# $OUT/pipeline_status.txt.
# usage: source tools/bpv4cdx/env.b200.sh; nohup bash tools/bpv4cdx/ae3_pipeline.sh > /dev/null 2>&1 &
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3
LOG=$OUT/logs/pipeline.log
STATUS=$OUT/pipeline_status.txt
AE3="bash tools/bpv4cdx/ae3.sh"
mkdir -p "$OUT/logs"
exec >> "$LOG" 2>&1
say() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }
passed() { python -c "import json,sys; sys.exit(0 if json.load(open('$1')).get('pass') else 1)" 2>/dev/null; }
building() { pgrep -f "policies.BPV4CDX.build_cac[h]e" > /dev/null; }

say "pipeline start (sampler $BPV4CDX_AE3_SAMPLER)"
while building; do sleep 300; done
say "cache build finished: $($AE3 progress 2>/dev/null | grep TOTAL)"
pct=$($AE3 progress 2>/dev/null | grep TOTAL | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
if [ -n "$pct" ] && python -c "import sys; sys.exit(0 if float('$pct') < 99 else 1)"; then
  say "cache only $pct% done: resuming the build once"
  BPV4CDX_CACHE_WORKERS=6 $AE3 cache > "$OUT/logs/cache_resume.log" 2>&1
  say "cache resume finished: $($AE3 progress 2>/dev/null | grep TOTAL)"
fi

$AE3 index
$AE3 g1g2
if passed "$OUT/g1_g2/g1_g2.json"; then say "G1/G2 pass"; else say "G1/G2 FAIL -> stop (see $OUT/g1_g2/)"; exit 1; fi

$AE3 train ae3_g3_overfit
$AE3 g3
if passed "$OUT/g3/g3.json"; then say "G3 pass"; else say "G3 FAIL -> stop (see $OUT/g3/)"; exit 1; fi

$AE3 bench 16 32 64 | tee "$OUT/logs/bench_summary.txt"
BS=$(python - "$OUT/logs/bench_summary.txt" <<'EOF'
import re, sys
limit = 0.85 * 179.1
best = None
for line in open(sys.argv[1]):
    m = re.search(r"batch (\d+)/GPU: .*-> (\d+) trajectories/s .*peak GPU memory ([\d.]+) GB", line)
    if m and float(m.group(3)) < limit and (best is None or int(m.group(2)) > best[1]):
        best = (int(m.group(1)), int(m.group(2)))
print(best[0] if best else 16)
EOF
)
say "bench done: $(tr '\n' ';' < "$OUT/logs/bench_summary.txt") -> batch $BS per GPU"

say "train ae3_main start (batch $BS per GPU, $((BS * BPV4CDX_NUM_GPUS)) trajectories per step)"
$AE3 train ae3_main --batch_size=$BS
say "train ae3_main exited"
