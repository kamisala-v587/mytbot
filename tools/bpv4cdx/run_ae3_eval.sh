#!/bin/bash
# AE-3 checkpoints (B200, steps 600k / 700k) evaluated on this machine, report-v2 protocol.
# Wave 1 (RoboTwin, comparable with AE-2): G4 / G5 v1 + v2 (K = 1; AE-2 text-only model as reference), HOM with the
#   end-effector error, bp_probe test, G7-D, G7-S, attention maps, demo-encoder inspection.
# Wave 2 (all sources, after the held-out cache and the arrays are ready): HOM on all held-out episodes (by source),
#   G4 / G5 v2 on 1,500 task-stratified held-out episodes.
# Eval copies of the checkpoints: tools/bpv4cdx/make_ae3_eval_ckpts.py (variants ae3 / rt / probe).
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae3_eval.sh
set -u
cd "$BPV4CDX_REPO" || exit 1
E=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3_eval
R=$E/results
mkdir -p "$E/logs" "$R"/{eval,test,g7d,g7s,attn,zdemo}
M="python -m lerobot.policies.BPV4CDX"
T=$(ls -d "$BPV4CDX_OUT_ROOT"/BPv4CDX/stage_a/v2/ae2l_textonly/*/checkpoints/020000/pretrained_model | tail -1)
PROBE=configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt
LABELS=$BPV4CDX_OUT_ROOT/BPv4/probe/labels_probe.json
PR="--data probe --probe_repo_file $PROBE --labels $LABELS"
ck() { echo "$E/$1/checkpoints/$2/pretrained_model"; }
# the probe variant keeps its own cache dir: no BPV4CDX_CACHE_ROOT rewrite
noroot() { env -u BPV4CDX_CACHE_ROOT "$@"; }

echo "wave 1 $(date '+%m-%d %H:%M')"
for i in 0 1; do
  S=$([ $i = 0 ] && echo 0600000 || echo 0700000); s=$((10#$S / 10000))w
  (CUDA_VISIBLE_DEVICES=$i $M.gate_g4_g7 --ckpt "$(ck rt $S)" --textonly_ckpt "$T" --out "$R/eval" --name "rt_$s" &&
   CUDA_VISIBLE_DEVICES=$i $M.gate_g4_g7 --ckpt "$(ck rt $S)" --textonly_ckpt "$T" --out "$R/eval" --name "rt_${s}_v2" --protocol v2 &&
   CUDA_VISIBLE_DEVICES=$i noroot $M.gate_g4_g7 --ckpt "$(ck probe $S)" --textonly_ckpt "$T" --out "$R/test" --name "probe_$s" --split all &&
   CUDA_VISIBLE_DEVICES=$i $M.heldout_monitor --run "$E/rt" --steps "$((10#$S))" --max_episodes 0 --out_name "hom_rt_$s.jsonl") \
    > "$E/logs/wave1_gate_$s.log" 2>&1 &
done
(for S in 0600000 0700000; do s=$((10#$S / 10000))w
   CUDA_VISIBLE_DEVICES=2 noroot $M.probe_zdemo --ckpt "$(ck probe $S)" $PR --out "$R/g7d" --name "${s}_probe" &&
   CUDA_VISIBLE_DEVICES=2 $M.probe_zdemo --ckpt "$(ck rt $S)" --data heldout --out "$R/g7d" --name "${s}_heldout" &&
   CUDA_VISIBLE_DEVICES=2 $M.attn_map --ckpt "$(ck rt $S)" --episodes 100 --out "$R/attn" --name "${s}_heldout" &&
   CUDA_VISIBLE_DEVICES=2 $M.zdemo_inspect --ckpt "$(ck rt $S)" --out "$R/zdemo" --name "${s}_heldout"
 done) > "$E/logs/wave1_diag.log" 2>&1 &
(for S in 0600000 0700000; do s=$((10#$S / 10000))w
   CUDA_VISIBLE_DEVICES=3 $M.saliency_zdemo --ckpt "$(ck rt $S)" --data heldout --episodes 100 --figures 3 --out "$R/g7s" --name "${s}_heldout" &&
   CUDA_VISIBLE_DEVICES=3 noroot $M.saliency_zdemo --ckpt "$(ck probe $S)" $PR --episodes 0 --figures 3 --out "$R/g7s" --name "${s}_probe"
 done) > "$E/logs/wave1_g7s.log" 2>&1 &

echo "waiting for the held-out cache and the arrays $(date '+%m-%d %H:%M')"
while pgrep -f "BPV4CDX.build_cache --manifest $E/ae3_split_dev.json" > /dev/null || pgrep -f "ae3_data prepare-arrays" > /dev/null; do
  sleep 60
done
$M.ae3_data progress --manifest "$E/ae3_split_dev.json" --data_roots /public-data/dataset:/vla/workspace/data \
  --cache_dir /vla/workspace/cache/bpv4cdx_ae3_su 2>&1 | tail -9
echo "wave 2 $(date '+%m-%d %H:%M')"
for i in 0 1; do
  S=$([ $i = 0 ] && echo 0600000 || echo 0700000); s=$((10#$S / 10000))w
  (CUDA_VISIBLE_DEVICES=$((i + 2)) $M.heldout_monitor --run "$E/ae3" --steps "$((10#$S))" --max_episodes 0 --out_name "hom_all_$s.jsonl" &&
   CUDA_VISIBLE_DEVICES=$((i + 2)) $M.gate_g4_g7 --ckpt "$(ck ae3 $S)" --out "$R/eval" --name "all_${s}_v2" --protocol v2 --max_episodes 1500) \
    > "$E/logs/wave2_$s.log" 2>&1 &
done
wait
echo "done $(date '+%m-%d %H:%M')"
