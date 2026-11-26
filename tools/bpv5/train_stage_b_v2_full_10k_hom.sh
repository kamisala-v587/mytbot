#!/bin/bash
# Stage B v2 full：本机 4 卡探测训 10000 步，每 1000 步存 ckpt 并跑 HOM。
# 对齐 stage_b_v2_full.jsonc；默认 DDP + shared_bucket。若 FSDP 已裁定达标可：
#   DISTRIBUTED_BACKEND=fsdp FSDP_WRAP_POLICY=layer bash tools/bpv5/train_stage_b_v2_full_10k_hom.sh
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1

CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/stage_b_v2_full.jsonc}
STEPS=${STEPS:-10000}
SAVE_FREQ=${SAVE_FREQ:-1000}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-29750}
DISTRIBUTED_BACKEND=${DISTRIBUTED_BACKEND:-ddp}
FSDP_WRAP_POLICY=${FSDP_WRAP_POLICY:-layer}
STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-$REPO/outputs/BPv5/bv2/full_10k_hom_${DISTRIBUTED_BACKEND}_${STAMP}}

EXTRA=(
  --dataset.pairing_storage=shared_bucket
  --dataset.init_timing=true
  --dataset.heldout_monitor=true
  --steps="$STEPS"
  --save_checkpoint=true
  --save_freq="$SAVE_FREQ"
  --ddp_find_unused_parameters=true
  --distributed_backend="$DISTRIBUTED_BACKEND"
  --fsdp_wrap_policy="$FSDP_WRAP_POLICY"
  --fsdp_sharding_strategy=full_shard
  --fsdp_use_orig_params=true
  --fsdp_compile_layers=false
  --output_dir="$OUT_DIR"
  --wandb.enable=false
)

if [ "$DISTRIBUTED_BACKEND" = "fsdp" ]; then
  # FSDP needs accelerate mixed_precision=no
  export MIXED_PRECISION=no
fi

echo "out=$OUT_DIR backend=$DISTRIBUTED_BACKEND steps=$STEPS save_freq=$SAVE_FREQ"
NPROC_PER_NODE="$NPROC_PER_NODE" MASTER_PORT="$MASTER_PORT" FOREGROUND="${FOREGROUND:-1}" \
  bash tools/bpv5/train_stage_a_dist.sh "$CONFIG" "${EXTRA[@]}"
