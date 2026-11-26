#!/bin/bash
# BPv5 demo cache (wall ViT P5) for the whole 80-D dataset on this machine: one items list, then GPUS x PROCS shards
# of lerobot.policies.BPV5QFormer.build_cache in the background. Rerunning resumes (existing files are skipped).
# usage: bash tools/bpv5/build_cache.sh [cache_dir] [procs_per_gpu] [workers_per_proc]
# logs: <cache_dir>/_logs/rank<r>.log; progress: grep "done" <cache_dir>/_logs/*.log | tail
set -u
CACHE=${1:-/vla/workspace/cache/bpv5_cleaned_p5}
PROCS=${2:-2}
WORKERS=${3:-16}
GPUS=(${GPUS:-0 1 2 3})
cd /vla/workspace/my_tbot || exit 1
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$CACHE/_logs"
ITEMS=$CACHE/_items.json
[ -f "$ITEMS" ] || python -m lerobot.policies.BPV5QFormer.build_cache --cache_dir "$CACHE" --items_file "$ITEMS" --only_items \
  > "$CACHE/_logs/items.log" 2>&1 || { echo "items list failed: $CACHE/_logs/items.log"; exit 1; }
WORLD=$(( ${#GPUS[@]} * PROCS ))
rank=0
for g in "${GPUS[@]}"; do
  for p in $(seq 1 "$PROCS"); do
    CUDA_VISIBLE_DEVICES=$g setsid nohup python -m lerobot.policies.BPV5QFormer.build_cache --cache_dir "$CACHE" \
      --items_file "$ITEMS" --rank $rank --world $WORLD --num_workers "$WORKERS" --episodes_per_batch 8 \
      > "$CACHE/_logs/rank$rank.log" 2>&1 < /dev/null &
    rank=$((rank + 1))
  done
done
echo "started $WORLD shards on GPUs ${GPUS[*]}; items $(python -c "import json;print(len(json.load(open('$ITEMS'))))") -> $CACHE"
