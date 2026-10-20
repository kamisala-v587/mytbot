#!/bin/bash
# Regenerate configs/Pro6k/bpv4cdx_plan/generated_local/*.json from ae_main.jsonc with this machine's paths.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/gen_configs.sh
set -u
cd "$BPV4CDX_REPO" || exit 1
python tools/bpv4cdx/make_gate_configs.py --out_dir "$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_local" \
  --data_root "$BPV4CDX_DATA_ROOT" --cache_root "$BPV4CDX_CACHE_ROOT" --qwen_path "$BPV4CDX_QWEN_PATH" \
  --out_root "$BPV4CDX_OUT_ROOT" --stats_root "$BPV4CDX_STATS_ROOT" --repo_root "$BPV4CDX_REPO/configs/ds_ids/pro6k" \
  --num_gpus "$BPV4CDX_NUM_GPUS"
