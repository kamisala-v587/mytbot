#!/bin/bash
# G4 / G5 / G7 (held-out) with the revised G5 protocol v2 and K = 3, for S-A and S-U, seeds 42 and 43 (one model per
# GPU, same partners and noise), then the two-seed check per sampler. Waits for run_seed43.sh to finish first.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_v2_eval.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/v2_eval
mkdir -p "$OUT/logs" "$R"
until grep -q "^done" "$OUT/run_seed43.log" 2>/dev/null; do sleep 60; done
run() { ls -d "$OUT/$1"/*/ | tail -1; }
LINE=$(grep -o 'seed43 steps: main [0-9]\+, su [0-9]\+' "$OUT/run_seed43.log")
S_MAIN=$(echo "$LINE" | awk '{print $4}' | tr -d ,)
S_SU=$(echo "$LINE" | awk '{print $6}')
[ -n "$S_MAIN" ] && [ -n "$S_SU" ] || { echo "seed43 steps not found"; exit 1; }
T=$(run ae2l_textonly)checkpoints/020000/pretrained_model
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R --textonly_ckpt $T --protocol v2 --seeds 3"
echo "eval v2 $(date '+%m-%d %H:%M') (seed43 steps: main $S_MAIN, su $S_SU)"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$(run ae2l_main)checkpoints/020000/pretrained_model" --name sa42 > "$OUT/logs/v2_sa42.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$(run ae2l_main_seed43)checkpoints/$S_MAIN/pretrained_model" --name sa43 > "$OUT/logs/v2_sa43.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$(run ae2l_su)checkpoints/020000/pretrained_model" --name su42 > "$OUT/logs/v2_su42.log" 2>&1 &
CUDA_VISIBLE_DEVICES=3 $EV --ckpt "$(run ae2l_su_seed43)checkpoints/$S_SU/pretrained_model" --name su43 > "$OUT/logs/v2_su43.log" 2>&1 &
wait
python - "$R" <<'EOF'
import json, sys
from pathlib import Path
from lerobot.policies.BPV4CDX.gate_g4_g7 import seeds_consistent
r = Path(sys.argv[1])
out = {}
for s in ("sa", "su"):
    a, b = json.loads((r / f"{s}42.json").read_text()), json.loads((r / f"{s}43.json").read_text())
    out[s] = seeds_consistent(a, b)
    print(s, out[s])
(r / "seeds_v2.json").write_text(json.dumps(out, indent=1))
EOF
echo "done $(date '+%m-%d %H:%M')"
