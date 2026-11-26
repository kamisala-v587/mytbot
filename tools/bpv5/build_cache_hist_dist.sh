#!/bin/bash
# BPv5 Stage B history ViT P5 cache (head camera, control-step stride 25), multi-node launcher.
# Foreground; one invocation per node. Resumable: shares <cache>/_items.json and skips
# episodes whose .pt already exists, so node / GPU count may differ across runs.
#   bash tools/bpv5/build_cache_hist_dist.sh [CACHE]
# Env overrides: NNODES NODE_RANK PROCS_PER_GPU WORKERS FRAME_CHUNK STRIDE GPU_DECODE_PIXELS MAX_RETRY
#
# Platform execution example (env as on the job UI):
#   cd /vla/workspace/my_tbot
#   bash tools/bpv5/build_cache_hist_dist.sh
# with HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1
set -u
CACHE=${1:-/vla/workspace/cache/bpv5_wall_hist_s25}
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va

NNODES=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${RANK:-0}}}
NGPU=$(nvidia-smi -L | wc -l)
PROCS=${PROCS_PER_GPU:-2}
PER_NODE=$((NGPU * PROCS))
WORLD=$((NNODES * PER_NODE))
NCPU=$(nproc)
AUTO_W=$((NCPU / PER_NODE - 2)); [ "$AUTO_W" -lt 4 ] && AUTO_W=4; [ "$AUTO_W" -gt 16 ] && AUTO_W=16
WORKERS=${WORKERS:-$AUTO_W}
FRAME_CHUNK=${FRAME_CHUNK:-32}
STRIDE=${STRIDE:-25}
GPU_DECODE_PIXELS=${GPU_DECODE_PIXELS:-921600}
MAX_RETRY=${MAX_RETRY:-5}

cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
unset NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_DISABLE

echo "[node $NODE_RANK/$NNODES] host=$(hostname) gpus=$NGPU procs/gpu=$PROCS world=$WORLD cpus=$NCPU workers=$WORKERS stride=$STRIDE frame_chunk=$FRAME_CHUNK"
[ "$NODE_RANK" -lt "$NNODES" ] || { echo "NODE_RANK >= NNODES, set NNODES/NODE_RANK explicitly"; exit 1; }
for d in /physical-ai-data/dataset_80D /physical-ai-data/norm_stats/action80_new /public-data/dataset \
         /vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm "$Q35/bin/python"; do
  [ -e "$d" ] || { echo "missing: $d (mount it)"; exit 1; }
done
python -c "import torch, transformers, torchcodec; assert torch.cuda.is_available(); print('torch', torch.__version__, 'tf', transformers.__version__)" || exit 1

mkdir -p "$CACHE/_logs"
ITEMS=$CACHE/_items.json
if [ ! -f "$ITEMS" ]; then
  if [ "$NODE_RANK" -eq 0 ]; then
    python -m lerobot.policies.BPV5QFormer.build_cache_hist --cache_dir "$CACHE" --items_file "$ITEMS" --only_items \
      > "$CACHE/_logs/items.log" 2>&1 || { echo "items list failed"; exit 1; }
  else
    until [ -f "$ITEMS" ]; do sleep 30; done
  fi
fi

kill_orphans() {
  ps -eo pid,ppid,cmd | awk -v q="$Q35/bin/python" '$2==1 && index($0,q) && /multiprocessing/ {print $1}' | xargs -r kill -9
}

run_shard() {
  local gpu=$1 rank=$2 log=$CACHE/_logs/node${NODE_RANK}_rank$2.log
  for attempt in $(seq 1 "$MAX_RETRY"); do
    echo "=== attempt $attempt $(date) ===" >> "$log"
    CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.policies.BPV5QFormer.build_cache_hist --cache_dir "$CACHE" \
      --items_file "$ITEMS" --rank "$rank" --world "$WORLD" --num_workers "$WORKERS" \
      --stride "$STRIDE" --frame_chunk "$FRAME_CHUNK" --gpu_decode_pixels "$GPU_DECODE_PIXELS" \
      >> "$log" 2>&1 < /dev/null && return 0
    echo "[rank $rank] exit $?, retry in 60s" | tee -a "$log"
    sleep 60
    kill_orphans
  done
  return 1
}

trap 'pkill -P $$; kill_orphans; exit 130' INT TERM
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  for p in $(seq 0 $((PROCS - 1))); do
    rank=$((NODE_RANK * PER_NODE + g * PROCS + p))
    run_shard "$g" "$rank" & pids+=($!)
  done
done

fails=0
for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
kill_orphans
echo "[node $NODE_RANK] done, failed shards=$fails, cache files=$(find "$CACHE" -name 'ep*.pt' | wc -l)"
[ "$fails" -eq 0 ]
