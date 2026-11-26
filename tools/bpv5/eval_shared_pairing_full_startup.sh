#!/bin/bash
# Stage B v2：四卡全量数据启动计时（shared_bucket，默认对比上次 list 全量基线）。
#
#   cd /vla/workspace/my_tbot
#   NPROC_PER_NODE=4 bash tools/bpv5/eval_shared_pairing_full_startup.sh
#
# Env:
#   PAIRING_STORAGE=shared_bucket|list   (default shared_bucket)
#   STEPS=3
#   MASTER_PORT=29742
#   FOREGROUND=1
#   HOM=0   # 默认关 HOM，避免 val 配对干扰启动计时；要复现完整旧启动链可 HOM=1

set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1

CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/stage_b_v2_full.jsonc}
PAIRING_STORAGE=${PAIRING_STORAGE:-shared_bucket}
STEPS=${STEPS:-3}
MASTER_PORT=${MASTER_PORT:-29742}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
HOM=${HOM:-0}
FOREGROUND=${FOREGROUND:-1}
export FOREGROUND

nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU"; exit 1; }
[ -f "$CONFIG" ] || { echo "missing $CONFIG"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=$REPO/outputs/BPv5/bv2/pairing_full_startup/${PAIRING_STORAGE}_${STAMP}
LOG_DIR=$REPO/outputs/BPv5/bv2/pairing_full_startup/logs
mkdir -p "$LOG_DIR" "$OUT_DIR"
RUN_LOG=$LOG_DIR/${PAIRING_STORAGE}_${STAMP}.log
SUMMARY=$LOG_DIR/summary_${STAMP}.txt

{
  echo "stamp=$STAMP host=$(hostname)"
  echo "config=$CONFIG pairing_storage=$PAIRING_STORAGE steps=$STEPS hom=$HOM"
  echo "nproc=$NPROC_PER_NODE master_port=$MASTER_PORT"
  echo "baseline_4gpu_list: Creating dataset 17:17:49 → M1 partners 17:27:59 (~610s) → Creating policy 17:27:59"
  echo "baseline_log: outputs/BPv5/stage_a/logs/train_stage_b_v2_full_node0_20261125_171658.log"
  echo "log=$RUN_LOG"
  echo "----"
} | tee "$SUMMARY"

EXTRA_HOM=(--dataset.heldout_monitor=false)
if [ "$HOM" = "1" ]; then
  EXTRA_HOM=(--dataset.heldout_monitor=true)
fi

set +e
NPROC_PER_NODE="$NPROC_PER_NODE" \
NNODES="$NNODES" \
NODE_RANK="$NODE_RANK" \
MASTER_PORT="$MASTER_PORT" \
FOREGROUND=1 \
bash tools/bpv5/train_stage_a_dist.sh "$CONFIG" \
  --dataset.pairing_storage="$PAIRING_STORAGE" \
  --dataset.init_timing=true \
  --dataset.max_trajectories=0 \
  --ddp_find_unused_parameters=true \
  --steps="$STEPS" \
  --save_checkpoint=false \
  --wandb.enable=false \
  --output_dir="$OUT_DIR" \
  "${EXTRA_HOM[@]}" \
  2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

{
  echo "exit_code=$RC"
  echo "--- timing / pairing ---"
  grep -nE "init_timing|split_pair_build|M1 partners|Creating dataset|Creating policy|bpv5_qformer dataset split=train|stage B split=train M1" "$RUN_LOG" || true
} | tee -a "$SUMMARY"

echo "done summary=$SUMMARY"
exit "$RC"
