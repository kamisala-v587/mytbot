#!/usr/bin/env bash
# Wait for pb3_4src_c1 @1k HOM; if alt/wrong ≈ matched (no identity), keep notes and
# launch pb_rt_vobs (V-obs + RoboTwin-only). If identity already strong, do not auto-launch.
set -euo pipefail
RUN_ROOT=/vla/workspace/my_tbot/outputs/BPv5/stage_b/pb3_4src_c1/2026-11-20_20-15
NOTE=$RUN_ROOT/NOTE_1k_hom.txt
G9=$RUN_ROOT/g9/a_001000.json
TBOT=/vla/workspace/my_tbot
LOG=$TBOT/outputs/BPv5/stage_b/logs/watch_pb3_4src_c1_1k_then_rt_vobs.log
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "==== $(date -Is) watch start (g9=$G9)"

THRESH=${THRESH:-1.15}  # alt/wrong ratio below this ≈ "no identity"
while [[ ! -f "$G9" ]]; do
  step="?"
  if [[ -f "$RUN_ROOT/loss.log" ]]; then
    step=$(awk -F, 'NR>1{s=$4} END{print s+0}' "$RUN_ROOT/loss.log" 2>/dev/null || echo "?")
  fi
  echo "$(date -Is) waiting for $G9 (loss step≈$step)"
  sleep 60
done

python3 - <<'PY' | tee "$NOTE"
import json, os
from pathlib import Path
from datetime import datetime
g9 = Path("/vla/workspace/my_tbot/outputs/BPv5/stage_b/pb3_4src_c1/2026-11-20_20-15/g9/a_001000.json")
d = json.loads(g9.read_text())
t = d.get("ratios") or {}

def ratio(k):
    x = t.get(k) or {}
    return float(x["ratio"]) if isinstance(x, dict) and "ratio" in x else None

blank, alt, wrong = ratio("blank"), ratio("alt"), ratio("wrong")
thresh = float(os.environ.get("THRESH", "1.15"))
no_id = (alt is not None and alt < thresh) and (wrong is not None and wrong < thresh)
print(f"time={datetime.now().isoformat(timespec='seconds')}")
print(f"g9={g9}")
print(f"HOM@1k blank/match={blank} alt/match={alt} wrong/match={wrong}")
print(f"thresh={thresh} no_identity={no_id}")
print()
if no_id:
    print("CONCLUSION: On 4-source + full-prefix (+C1 idle), alt/wrong stay near 1 at 1k —")
    print("demo identity not engaged. Contrastive blank-margin cannot rescue a data regime")
    print("that does not support identity. Keep this run; next: V-obs + RoboTwin-only (pb_rt_vobs).")
    print("LAUNCH=1")
else:
    print("CONCLUSION: alt/wrong already separated from matched; do NOT auto-switch.")
    print("LAUNCH=0")
PY

if grep -q '^LAUNCH=1$' "$NOTE"; then
  echo "$(date -Is) stopping pb3_4src_c1 to free GPUs for pb_rt_vobs"
  pkill -f 'lerobot.scripts.lerobot_train --config_path=configs/bpv5/stage_b/probes/stage_b_pb3_4src_c1' || true
  pkill -f 'accelerate launch .*stage_b_pb3_4src_c1' || true
  sleep 8
  cd "$TBOT"
  CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/train_stage_b.sh pb_rt_vobs
  echo "$(date -Is) launched pb_rt_vobs"
else
  echo "$(date -Is) skip launch (identity already present or parse failed)"
fi
echo "==== $(date -Is) watch done"
