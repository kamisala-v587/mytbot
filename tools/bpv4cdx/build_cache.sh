#!/bin/bash
# Build the 32-slot ViT cache for one sampler on all GPUs (resumable: existing episode files are skipped).
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/build_cache.sh sa [repo_file]
set -u
SAMPLER=${1:-sa}
REPO_FILE=${2:-$BPV4CDX_REPO/configs/ds_ids/pro6k/bpv4cdx_train_rel.txt}
CACHE_DIR=$BPV4CDX_CACHE_ROOT/bpv4cdx_$SAMPLER
LOG_DIR=$BPV4CDX_OUT_ROOT/BPv4CDX/cache_logs/$SAMPLER
mkdir -p "$LOG_DIR"
cd "$BPV4CDX_REPO" || exit 1
WORKERS=${BPV4CDX_CACHE_WORKERS:-12}
pids=()
for ((r = 0; r < BPV4CDX_NUM_GPUS; r++)); do
  CUDA_VISIBLE_DEVICES=$r python -m lerobot.policies.BPV4CDX.build_cache \
    --repo_file "$REPO_FILE" --data_root "$BPV4CDX_DATA_ROOT" --cache_dir "$CACHE_DIR" --sampler "$SAMPLER" \
    --qwen_path "$BPV4CDX_QWEN_PATH" --rank $r --world "$BPV4CDX_NUM_GPUS" --num_workers "$WORKERS" \
    > "$LOG_DIR/rank$r.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "cache $SAMPLER finished (status $status): $CACHE_DIR"
exit $status
