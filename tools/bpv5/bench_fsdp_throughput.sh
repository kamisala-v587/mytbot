#!/bin/bash
# Short DDP vs FSDP throughput probe for BPv5 Stage B v2 full.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/bpv5/bench_fsdp_throughput.sh
#   CUDA_VISIBLE_DEVICES=0,1,2,3 STEPS=20 WRAP=layer bash tools/bpv5/bench_fsdp_throughput.sh
#
# Writes logs under outputs/BPv5/bv2/fsdp_bench/<stamp>/.
# Does NOT change default stage_b_v2_full.jsonc; uses CLI overrides.
set -euo pipefail
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va
cd "$REPO"
# shellcheck disable=SC1091
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}

CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/stage_b_v2_full.jsonc}
PRETRAINED=${PRETRAINED:-}
STEPS=${STEPS:-40}
WRAP=${WRAP:-layer}
PORT=${PORT:-29831}
BATCH=${BATCH:-2}
SEGS=${SEGS:-2}
ACCUM=${ACCUM:-3}

if [ -n "${NUM_GPUS:-}" ]; then
  NPROC=$NUM_GPUS
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a _gpus <<< "$CUDA_VISIBLE_DEVICES"
  NPROC=${#_gpus[@]}
else
  NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
[ "${NPROC:-0}" -ge 2 ] || { echo "need >=2 GPUs for FSDP bench (got NPROC=$NPROC)"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_ROOT=$REPO/outputs/BPv5/bv2/fsdp_bench/$STAMP
mkdir -p "$OUT_ROOT"
echo "bench out=$OUT_ROOT nproc=$NPROC steps=$STEPS wrap=$WRAP"

COMMON_OVERRIDES=(
  --steps="$STEPS"
  --batch_size="$BATCH"
  --gradient_accumulation_steps="$ACCUM"
  --save_checkpoint=false
  --eval_freq=0
  --log_freq=1
  --gpu_stats_freq=5
  --dataset.segments_per_trajectory="$SEGS"
  --dataset.pairing_storage=shared_bucket
  --dataset.init_timing=true
  --dataset.heldout_monitor=false
  --dataset.max_trajectories="${MAX_TRAJECTORIES:-0}"
  --ddp_find_unused_parameters=true
  --policy.preflight_memory_check=false
  --wandb.enable=false
)
if [ -n "$PRETRAINED" ]; then
  COMMON_OVERRIDES+=(--policy.pretrained_path="$PRETRAINED")
fi

run_one() {
  local backend=$1
  local wrap=$2
  local tag=$3
  local out_dir=$OUT_ROOT/$tag
  local log=$OUT_ROOT/${tag}.log
  mkdir -p "$out_dir"
  echo "=== $tag backend=$backend wrap=$wrap ===" | tee "$log"
  # FSDP requires Accelerate mixed_precision=no so BF16/FP32 params are preserved.
  local mp=no
  if [ "$backend" = "ddp" ]; then
    mp=${MIXED_PRECISION:-no}
  fi
  accelerate launch \
    --num_processes="$NPROC" \
    --main_process_port "$PORT" \
    --mixed_precision "$mp" \
    -m lerobot.scripts.lerobot_train \
    --config_path="$CONFIG" \
    --output_dir="$out_dir" \
    --distributed_backend="$backend" \
    --fsdp_wrap_policy="$wrap" \
    --fsdp_sharding_strategy=full_shard \
    --fsdp_use_orig_params=true \
    --fsdp_compile_layers=false \
    "${COMMON_OVERRIDES[@]}" \
    ${EXTRA_OVERRIDES:-} \
    2>&1 | tee -a "$log"
  PORT=$((PORT + 1))
}

BACKENDS=${BACKENDS:-ddp,fsdp}
IFS=',' read -r -a _backends <<< "$BACKENDS"
for backend in "${_backends[@]}"; do
  backend=$(echo "$backend" | tr -d '[:space:]')
  [ -n "$backend" ] || continue
  if [ "$backend" = "ddp" ]; then
    run_one ddp root "ddp_b${BATCH}_s${SEGS}_a${ACCUM}"
  elif [ "$backend" = "fsdp" ]; then
    run_one fsdp "$WRAP" "fsdp_${WRAP}_b${BATCH}_s${SEGS}_a${ACCUM}"
  else
    echo "bad backend=$backend"; exit 1
  fi
done

echo "done. logs in $OUT_ROOT"
ls -la "$OUT_ROOT"
