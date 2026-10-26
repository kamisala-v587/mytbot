#!/bin/bash
# Wave 2 of run_ae3_eval.sh (all sources): waits for the held-out cache build, writes its cache index (the AE-3 config
# then skips the uncached training episodes), then HOM on all held-out episodes and G4 / G5 v2 on 1,500 of them.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae3_eval_wave2.sh
set -u
cd "$BPV4CDX_REPO" || exit 1
E=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3_eval
R=$E/results
C=/vla/workspace/cache/bpv4cdx_ae3_su
M="python -m lerobot.policies.BPV4CDX"
DATA=(--manifest "$E/ae3_split_dev.json" --data_roots /public-data/dataset:/vla/workspace/data)
while pgrep -f "^python -m lerobot.policies.BPV4CDX.build_cache --manifest" > /dev/null; do sleep 60; done
$M.ae3_data progress "${DATA[@]}" --cache_dir "$C" 2>&1 | tail -9
$M.ae3_data cache-index "${DATA[@]}" --cache_dir "$C" --out "$C/_index.txt" || exit 1
echo "wave 2 $(date '+%m-%d %H:%M')"
for i in 0 1; do
  S=$([ $i = 0 ] && echo 0600000 || echo 0700000); s=$((10#$S / 10000))w
  CK=$E/ae3/checkpoints/$S/pretrained_model
  (CUDA_VISIBLE_DEVICES=$((i + 2)) $M.heldout_monitor --run "$E/ae3" --steps "$((10#$S))" --max_episodes 0 \
     --out_name "hom_all_$s.jsonl" --dataset.cache_index="$C/_index.txt" &&
   CUDA_VISIBLE_DEVICES=$((i + 2)) $M.gate_g4_g7 --ckpt "$CK" --out "$R/eval" --name "all_${s}_v2" --protocol v2 \
     --max_episodes 1500 --dataset.cache_index="$C/_index.txt") > "$E/logs/wave2_$s.log" 2>&1 &
done
wait
echo "done $(date '+%m-%d %H:%M')"
