#!/bin/bash
# Current-observation ViT cache of the BP RoboTwin training index (dataset.obs_cache_dir), all GPUs, resumable.
# usage: source tools/bpv4cdxb/env.local.sh; bash tools/bpv4cdxb/build_obs_cache.sh [index] [out]
set -u
INDEX=${1:-$BPV4CDXB_REPO/outputs/BPv4CDX/stage_b/b4_sft_bp_v1_index.json}
OUT=${2:-/vla/workspace/cache/bpv4cdxb_obs_vit}
# any main-path run with the same dataset settings (only the preprocessing fields are used)
TC=${OBS_CACHE_TRAIN_CONFIG:-$BPV4CDXB_REPO/outputs/BPv4CDX/stage_b/b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47/checkpoints/050000/pretrained_model/train_config.json}
PER=${OBS_CACHE_PROCS_PER_GPU:-2}
WORKERS=${OBS_CACHE_WORKERS:-12}
LOGS=$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/obs_cache_logs
mkdir -p "$LOGS"
cd "$BPV4CDXB_REPO" || exit 1
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
WORLD=$((BPV4CDXB_NUM_GPUS * PER))
pids=()
for ((r = 0; r < WORLD; r++)); do
  CUDA_VISIBLE_DEVICES=$((r % BPV4CDXB_NUM_GPUS)) python -m lerobot.policies.BPV4CDXB.build_obs_cache \
    --train_config "$TC" --index "$INDEX" --out "$OUT" --rank $r --world $WORLD --workers "$WORKERS" \
    > "$LOGS/rank$r.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "obs cache finished (status $status): $OUT, $(find "$OUT" -name 'ep*.npy' | wc -l) episode files"
exit $status
