#!/bin/bash
# AE-2 at larger scale (50 tasks, 20k steps): train main / text-only / S-U one after another on all GPUs, pick the
# main checkpoint on the held-out set, then G4 / G5 with the fixed protocol (K = 1, one alternate and one position
# partner, partner seed 0) for the main and S-U models at the same step; sensitivity runs (partner seeds 1 / 2, K = 3
# with 3 position partners) and the 10-task slice (first-round subset) are reported, not used for the verdict.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2l.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
R=$OUT/ae2l_eval
mkdir -p "$OUT/logs" "$R"
bash tools/bpv4cdx/gen_configs.sh > /dev/null || exit 1

for v in ae2l_main ae2l_textonly ae2l_su; do
  if ls -d "$OUT/$v"/*/checkpoints/020000/pretrained_model > /dev/null 2>&1; then echo "skip $v (trained)"; continue; fi
  echo "train $v $(date '+%m-%d %H:%M')"
  BPV4CDX_PORT=29693 bash tools/bpv4cdx/train.sh "$v" --output_dir="$OUT/$v" > "$OUT/logs/train_$v.log" 2>&1 || { echo "train $v failed"; exit 1; }
done

run() { ls -d "$OUT/$1"/*/ | tail -1; }
echo "select $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=0 python -m lerobot.policies.BPV4CDX.select_ckpt --run "$(run ae2l_main)" --out "$R" --name select_main \
  > "$OUT/logs/ae2l_select.log" 2>&1 || { echo "select failed"; exit 1; }
STEP=$(python -c "import json,pathlib; print(pathlib.Path(json.load(open('$R/select_main.json'))['selected']).parent.name)")
echo "selected step $STEP"
MAIN=$(run ae2l_main)checkpoints/$STEP/pretrained_model
T=$(run ae2l_textonly)checkpoints/$STEP/pretrained_model
SU=$(run ae2l_su)checkpoints/$STEP/pretrained_model
SUBSET=$BPV4CDX_REPO/configs/ds_ids/pro6k/bpv4cdx_g45_subset_rel.txt
EV="python -m lerobot.policies.BPV4CDX.gate_g4_g7 --out $R --textonly_ckpt $T"

echo "eval $(date '+%m-%d %H:%M')"
CUDA_VISIBLE_DEVICES=0 $EV --ckpt "$MAIN" --name main > "$OUT/logs/ae2l_main.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 $EV --ckpt "$SU" --name su > "$OUT/logs/ae2l_su.log" 2>&1 &
p1=$!
(CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$MAIN" --name main_pseed1 --partner_seed 1 &&
 CUDA_VISIBLE_DEVICES=2 $EV --ckpt "$MAIN" --name main_pseed2 --partner_seed 2) > "$OUT/logs/ae2l_pseed.log" 2>&1 &
p2=$!
(CUDA_VISIBLE_DEVICES=3 $EV --ckpt "$MAIN" --name main_subset10 --dataset.repo_id_file="$SUBSET" &&
 CUDA_VISIBLE_DEVICES=3 $EV --ckpt "$MAIN" --name main_K3_pos3 --seeds 3 --n_pos 3) > "$OUT/logs/ae2l_extra.log" 2>&1 &
p3=$!
wait $p0; echo "main exit $?"
wait $p1; echo "su exit $?"
wait $p2; echo "partner seeds exit $?"
wait $p3; echo "subset10 / K3 exit $?"
echo "done $(date '+%m-%d %H:%M')"
