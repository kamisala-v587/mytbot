#!/bin/bash
# Stage B demo ViT cache (P5, AE-3 S-U v2, same files as Stage A) for a manifest subset on this machine.
# Resumable: existing episode files are skipped. Held-out episodes and RoboTwin first.
# usage: source tools/bpv4cdxb/env.local.sh; bash tools/bpv4cdxb/build_vit_cache.sh [manifest]
set -u
# default: full pro6k (repos.txt) without egodex; pass et3_split.json for the Stage-B subset
MANIFEST=${1:-$BPV4CDXB_REPO/configs/ds_ids/local/pro6k_noego_split.json}
ROOTS=${BPV4CDXB_DATA_ROOTS:-/public-data/dataset:/vla/workspace/data:/public-data/dataset/agibot-world3.0-fix/beta2025}
CACHE_DIR=${BPV4CDXB_VIT_CACHE:-/vla/workspace/cache/bpv4cdx_ae3_su}
ARRAYS=${BPV4CDXB_ARRAY_STORE:-/vla/workspace/cache/ae3_arrays_dev}
PER=${BPV4CDXB_CACHE_PROCS_PER_GPU:-3}
WORKERS=${BPV4CDXB_CACHE_WORKERS:-12}
LOGS=$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/vit_cache_logs
mkdir -p "$LOGS"
cd "$BPV4CDXB_REPO" || exit 1
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
WORLD=$((BPV4CDXB_NUM_GPUS * PER))
pids=()
for ((r = 0; r < WORLD; r++)); do
  CUDA_VISIBLE_DEVICES=$((r % BPV4CDXB_NUM_GPUS)) python -m lerobot.policies.BPV4CDX.build_cache \
    --manifest "$MANIFEST" --data_roots "$ROOTS" --array_store "$ARRAYS" --cache_dir "$CACHE_DIR" \
    --sampler su --qwen_path "$BPV4CDXB_QWEN_PATH" --store_spatial 5 --episodes_per_batch 4 \
    --order heldout_first --num_workers "$WORKERS" --rank $r --world $WORLD > "$LOGS/rank$r.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "vit cache finished (status $status): $CACHE_DIR; failures: $CACHE_DIR/_failures"
exit $status
