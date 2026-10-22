#!/bin/bash
# AE-2 sample-size check (subset held-out only, never the independent test):
#  1) evaluation noise: the subset main model with K = 3 noise draws and 3 position partners; partner seeds 1, 2
#  2) detection power: the earlier full-data ae_main on the same 150 subset held-out episodes
#  3) training seed: the subset main model retrained with seed 43, same protocol
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2_samplesize.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/g4_g5_samplesize
mkdir -p "$OUT/logs" "$R"
OLD=/vla/workspace/models/BPVA/BPv4CDX/stage_a
SUBSET=$BPV4CDX_REPO/configs/ds_ids/pro6k/bpv4cdx_g45_subset_rel.txt
ckpt() { ls -d "$OUT/$1"/*/checkpoints/last/pretrained_model | tail -1; }
MAIN=$(ckpt ae_g45_subset)
T=$(ckpt ae_g45_subset_textonly)
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R"

echo "eval $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$MAIN" --textonly_ckpt "$T" --name subset_K3_pos3 --seeds 3 --n_pos 3 > "$OUT/logs/ss_k3.log" 2>&1 &
p0=$!
(CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$MAIN" --textonly_ckpt "$T" --name subset_pseed1 --partner_seed 1 &&
 CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$MAIN" --textonly_ckpt "$T" --name subset_pseed2 --partner_seed 2) > "$OUT/logs/ss_pseed.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$OLD/ae_main/2026-10-20_11-28/checkpoints/010000/pretrained_model" \
  --textonly_ckpt "$OLD/ae_textonly/2026-10-20_11-29/checkpoints/010000/pretrained_model" --name fullmodel_on_subset \
  --dataset.repo_id_file="$SUBSET" > "$OUT/logs/ss_full.log" 2>&1 &
p2=$!
wait $p0; echo "K3 exit $?"
wait $p1; echo "partner seeds exit $?"
wait $p2; echo "full model exit $?"

echo "train seed43 $(date +%H:%M)"
BPV4CDX_PORT=29692 bash tools/bpv4cdx/train.sh ae_g45_subset --seed=43 --output_dir="$OUT/ae_g45_subset_seed43" \
  > "$OUT/logs/train_ae_g45_subset_seed43.log" 2>&1 || { echo "train seed43 failed"; exit 1; }
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$(ckpt ae_g45_subset_seed43)" --textonly_ckpt "$T" --name subset_seed43 > "$OUT/logs/ss_seed43.log" 2>&1
echo "seed43 eval exit $?"
echo "done $(date +%H:%M)"
