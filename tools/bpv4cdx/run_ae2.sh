#!/bin/bash
# AE-2 on this machine: train the subset main model, the text-only control and the S-U control (2000 steps each,
# all GPUs, one after another), then G4 / G5 on the subset held-out episodes for the main and the S-U model.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae2.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
mkdir -p "$OUT/logs"
bash tools/bpv4cdx/gen_configs.sh > /dev/null || exit 1

for v in ae_g45_subset ae_g45_subset_textonly ae_g45_subset_su; do
  echo "train $v $(date +%H:%M)"
  BPV4CDX_PORT=29691 bash tools/bpv4cdx/train.sh "$v" --output_dir="$OUT/$v" > "$OUT/logs/train_$v.log" 2>&1 || { echo "train $v failed"; exit 1; }
done

ckpt() { ls -d "$OUT/$1"/*/checkpoints/last/pretrained_model | tail -1; }
T=$(ckpt ae_g45_subset_textonly)
echo "eval $(date +%H:%M)"
CUDA_VISIBLE_DEVICES=0 python -m lerobot.policies.BPV4CDX.gate_g4_g7 --ckpt "$(ckpt ae_g45_subset)" --textonly_ckpt "$T" \
  --out "$OUT/g4_g5" --name ae_g45_subset > "$OUT/logs/g45_main.log" 2>&1 &
p1=$!
CUDA_VISIBLE_DEVICES=1 python -m lerobot.policies.BPV4CDX.gate_g4_g7 --ckpt "$(ckpt ae_g45_subset_su)" --textonly_ckpt "$T" \
  --out "$OUT/g4_g5" --name ae_g45_subset_su > "$OUT/logs/g45_su.log" 2>&1 &
p2=$!
wait $p1; echo "eval main exit $?"
wait $p2; echo "eval su exit $?"
echo "done $(date +%H:%M)"
