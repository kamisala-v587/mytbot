#!/bin/bash
# AE-0 (G0, G1, G2) and AE-1 (G3 training + gate) on this machine; G0 / G1-G2 / G3 run on separate GPUs.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_ae0_ae1.sh [out_subdir]
set -u
cd "$BPV4CDX_REPO" || exit 1
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/${1:-v2}
GEN=$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_local
mkdir -p "$OUT/logs"
bash tools/bpv4cdx/gen_configs.sh > /dev/null || exit 1

CUDA_VISIBLE_DEVICES=0 python -m lerobot.policies.BPV4CDX.gate_g0 \
  --repo_file configs/ds_ids/pro6k/bpv4cdx_train_rel.txt --data_root "$BPV4CDX_DATA_ROOT" --out "$OUT/g0" \
  --episodes 100 --sv_episodes 20 --sheets 3 --qwen_path "$BPV4CDX_QWEN_PATH" > "$OUT/logs/g0.log" 2>&1 &
p0=$!
CUDA_VISIBLE_DEVICES=1 python -m lerobot.policies.BPV4CDX.gate_g1_g2 --config "$GEN/ae_main.json" --out "$OUT/g1_g2" \
  --online_episodes 3 > "$OUT/logs/g1_g2.log" 2>&1 &
p1=$!
(
  CUDA_VISIBLE_DEVICES=2 BPV4CDX_PORT=29681 bash tools/bpv4cdx/train.sh ae_g3_overfit --output_dir="$OUT/ae_g3_overfit" &&
  CK=$(ls -d "$OUT"/ae_g3_overfit/*/checkpoints/last/pretrained_model | tail -1) &&
  echo "CKPT $CK" &&
  CUDA_VISIBLE_DEVICES=2 python -m lerobot.policies.BPV4CDX.gate_g3 --ckpt "$CK" --out "$OUT/g3" --seeds 5
) > "$OUT/logs/g3.log" 2>&1 &
p2=$!
wait $p0; echo "g0 exit $?"
wait $p1; echo "g1_g2 exit $?"
wait $p2; echo "g3 exit $?"
