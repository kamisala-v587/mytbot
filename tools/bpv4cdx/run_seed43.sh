#!/bin/bash
# Second training seed (43) for the AE-2 large-scale S-A and S-U models, full held-out HOM at every save, then
#  - G4 / G5 / G7 on the held-out set with --ckpt_seed2 (seed 42 vs 43, same partners and noise) -> G7 seed check
#  - independent test (bp_probe, report only) for the seed-43 models
# The seed-43 step is chosen from its full HOM: lowest matched error among saves whose E_alt / E_match >= 1.10.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_seed43.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/seed43_eval
mkdir -p "$OUT/logs" "$R"
bash tools/bpv4cdx/gen_configs.sh > /dev/null || exit 1

for v in ae2l_main ae2l_su; do
  if ls -d "$OUT/${v}_seed43"/*/checkpoints/020000/pretrained_model > /dev/null 2>&1; then echo "skip $v seed43 (trained)"; continue; fi
  echo "train $v seed43 $(date '+%m-%d %H:%M')"
  BPV4CDX_PORT=29694 bash tools/bpv4cdx/train.sh "$v" --seed=43 --output_dir="$OUT/${v}_seed43" \
    --dataset.heldout_monitor_max_episodes=0 > "$OUT/logs/train_${v}_seed43.log" 2>&1 || { echo "train $v seed43 failed"; exit 1; }
done

run() { ls -d "$OUT/$1"/*/ | tail -1; }
pick() {  # step chosen from the full HOM of a run
  python - "$1" <<'EOF'
import json, sys
from pathlib import Path
rows = [json.loads(l) for l in Path(sys.argv[1], "heldout_monitor.jsonl").read_text().splitlines()]
ok = [r for r in rows if (r.get("alt_over_match") or 0) >= 1.10]
best = min(ok or rows, key=lambda r: r["mse_matched"])
print(f"{int(best['step']):06d}")
EOF
}
S_MAIN=$(pick "$(run ae2l_main_seed43)")
S_SU=$(pick "$(run ae2l_su_seed43)")
echo "seed43 steps: main $S_MAIN, su $S_SU"
T=$(run ae2l_textonly)checkpoints/020000/pretrained_model
A42=$(run ae2l_main)checkpoints/020000/pretrained_model
U42=$(run ae2l_su)checkpoints/020000/pretrained_model
A43=$(run ae2l_main_seed43)checkpoints/$S_MAIN/pretrained_model
U43=$(run ae2l_su_seed43)checkpoints/$S_SU/pretrained_model
PROBE=$BPV4CDX_REPO/configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R --textonly_ckpt $T"

echo "eval $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$A42" --ckpt_seed2 "$A43" --name sa_seeds > "$OUT/logs/seed43_sa.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$U42" --ckpt_seed2 "$U43" --name su_seeds > "$OUT/logs/seed43_su.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$A43" --name sa43_bp_probe --split all --dataset.repo_id_file="$PROBE" --dataset.data_root=null \
  > "$OUT/logs/seed43_sa_test.log" 2>&1 &
p2=$!
CUDA_VISIBLE_DEVICES=3 $EV --ckpt "$U43" --name su43_bp_probe --split all --dataset.repo_id_file="$PROBE" --dataset.data_root=null \
  > "$OUT/logs/seed43_su_test.log" 2>&1 &
p3=$!
wait $p0; echo "sa seeds exit $?"
wait $p1; echo "su seeds exit $?"
wait $p2; echo "sa43 test exit $?"
wait $p3; echo "su43 test exit $?"
echo "done $(date '+%m-%d %H:%M')"
