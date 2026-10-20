#!/bin/bash
# Train one bpv4cdx variant on all GPUs (or 1 GPU for ae_g3_overfit).
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/train.sh <variant> [extra draccus overrides ...]
# variants: see tools/bpv4cdx/make_gate_configs.py (ae_main, ae_textonly, ae_su, ae_g3_overfit, ae_g45_subset, ...)
set -u
NAME=$1
shift
cd "$BPV4CDX_REPO" || exit 1
GEN=$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_local
python tools/bpv4cdx/make_gate_configs.py --out_dir "$GEN" --data_root "$BPV4CDX_DATA_ROOT" \
  --cache_root "$BPV4CDX_CACHE_ROOT" --qwen_path "$BPV4CDX_QWEN_PATH" --out_root "$BPV4CDX_OUT_ROOT" \
  --stats_root "$BPV4CDX_STATS_ROOT" --repo_root "$BPV4CDX_REPO/configs/ds_ids/pro6k" > /dev/null || exit 1
NPROC=$BPV4CDX_NUM_GPUS
[ "$NAME" = "ae_g3_overfit" ] && NPROC=1
mkdir -p "$BPV4CDX_OUT_ROOT/BPv4CDX/logs"
accelerate launch --num_processes="$NPROC" --main_process_port "${BPV4CDX_PORT:-29650}" \
  -m lerobot.scripts.lerobot_train --config_path="$GEN/$NAME.json" "$@" \
  2>&1 | tee "$BPV4CDX_OUT_ROOT/BPv4CDX/logs/train_${NAME}_$(date +%m%d_%H%M).log"
