#!/bin/bash
# AE-2 supplement (same setting as AE-2 at larger scale: 50 tasks, 20k steps, seed 42): S-UF with P5, then P64 with
# S-UF and S-U. Builds the S-UF cache (64 tokens stored, training + held-out + bp_probe), G1 / G2, trains the three
# models one after another (full held-out HOM at every save), and while the next model trains runs the report-v2
# evaluations of the finished one at step 20000:
#   GPU0  G4 / G5 / G7 held-out, protocol v1 (K = 1) then protocol v2 (K = 3)
#   GPU1  independent test (bp_probe), G7-D probes (bp_probe + held-out)
#   GPU2  G7-S saliency v2 (held-out 100 + bp_probe)
#   GPU3  attention maps (held-out 100) and demo-encoder inspection
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2p.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/ae2p
mkdir -p "$OUT/logs" "$R"/{eval,test,g7d,g7s,attn,zdemo,g1_g2}
bash tools/bpv4cdx/gen_configs.sh > /dev/null || exit 1
GEN=$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_local
TRAIN_LIST=$BPV4CDX_REPO/configs/ds_ids/pro6k/bpv4cdx_train_rel.txt
PROBE=$BPV4CDX_REPO/configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt
LABELS=$BPV4CDX_OUT_ROOT/BPv4/probe/labels_probe.json
SUF=$BPV4CDX_CACHE_ROOT/${BPV4CDX_CACHE_PREFIX:-bpv4cdx_}suf

echo "cache suf $(date '+%m-%d %H:%M')"
bash tools/bpv4cdx/build_cache.sh suf "$TRAIN_LIST" > "$OUT/logs/ae2p_cache_train.log" 2>&1 || { echo "cache (train) failed"; exit 1; }
bash tools/bpv4cdx/build_cache.sh suf "$PROBE" > "$OUT/logs/ae2p_cache_probe.log" 2>&1 || { echo "cache (probe) failed"; exit 1; }
N=$(find "$SUF" -name 'ep*.pt' | wc -l)
echo "suf cache files: $N (expected 7700 = 7500 RoboTwin + 200 bp_probe)"
[ "$N" -ge 7700 ] || { echo "suf cache incomplete"; exit 1; }

echo "g1/g2 $(date '+%m-%d %H:%M')"
for v in ae2l_suf ae2l_suf_p64; do
  CUDA_VISIBLE_DEVICES=0 python -m lerobot.policies.BPV4CDX.gate_g1_g2 --config "$GEN/$v.json" --out "$R/g1_g2/$v" \
    > "$OUT/logs/ae2p_g1g2_$v.log" 2>&1 || { echo "G1/G2 $v failed"; exit 1; }
  python -c "import json; r = json.load(open('$R/g1_g2/$v/g1_g2.json')); print('$v', {k: r[k]['pass'] for k in ('G1', 'G2')})"
done

ck() { ls -d "$OUT/$1"/*/checkpoints/020000/pretrained_model 2>/dev/null | tail -1; }
T=$(ck ae2l_textonly)
PR="--data probe --probe_repo_file $PROBE --labels $LABELS"

evaluate() {  # $1 = variant, $2 = short name
  local C
  C=$(ck "$1")
  local EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R/eval --textonly_ckpt $T"
  (CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$C" --name "$2" &&
   CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$C" --name "${2}_v2" --protocol v2 --seeds 3) > "$OUT/logs/ae2p_eval_$2.log" 2>&1 &
  (CUDA_VISIBLE_DEVICES=1 python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out "$R/test" --textonly_ckpt "$T" --ckpt "$C" \
     --name "${2}_bp_probe" --split all --dataset.repo_id_file="$PROBE" --dataset.data_root=null &&
   CUDA_VISIBLE_DEVICES=1 python -m lerobot.policies.BPV4CDX.probe_zdemo --ckpt "$C" $PR --out "$R/g7d" --name "${2}_probe" &&
   CUDA_VISIBLE_DEVICES=1 python -m lerobot.policies.BPV4CDX.probe_zdemo --ckpt "$C" --data heldout --out "$R/g7d" --name "${2}_heldout") \
    > "$OUT/logs/ae2p_test_g7d_$2.log" 2>&1 &
  (CUDA_VISIBLE_DEVICES=2 python -m lerobot.policies.BPV4CDX.saliency_zdemo --ckpt "$C" --data heldout --episodes 100 \
     --figures 3 --out "$R/g7s" --name "${2}_heldout" &&
   CUDA_VISIBLE_DEVICES=2 python -m lerobot.policies.BPV4CDX.saliency_zdemo --ckpt "$C" $PR --episodes 0 --figures 3 \
     --out "$R/g7s" --name "${2}_probe") > "$OUT/logs/ae2p_g7s_$2.log" 2>&1 &
  (CUDA_VISIBLE_DEVICES=3 python -m lerobot.policies.BPV4CDX.attn_map --ckpt "$C" --episodes 100 --out "$R/attn" --name "${2}_heldout" &&
   CUDA_VISIBLE_DEVICES=3 python -m lerobot.policies.BPV4CDX.zdemo_inspect --ckpt "$C" --out "$R/zdemo" --name "${2}_heldout") \
    > "$OUT/logs/ae2p_attn_$2.log" 2>&1 &
}

for pair in ae2l_suf:suf ae2l_suf_p64:suf_p64 ae2l_su_p64:su_p64; do
  v=${pair%%:*}; s=${pair##*:}
  if [ -z "$(ck "$v")" ]; then
    echo "train $v $(date '+%m-%d %H:%M')"
    BPV4CDX_PORT=29695 bash tools/bpv4cdx/train.sh "$v" --output_dir="$OUT/$v" \
      --dataset.heldout_monitor_max_episodes=0 > "$OUT/logs/train_$v.log" 2>&1 || { echo "train $v failed"; exit 1; }
  else
    echo "skip $v (trained)"
  fi
  echo "eval $v $(date '+%m-%d %H:%M') (in background)"
  evaluate "$v" "$s"
done
wait
echo "done $(date '+%m-%d %H:%M')"
