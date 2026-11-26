#!/bin/bash
# Stage B v2：部分数据对比 pairing_storage=list vs shared_bucket 的初始化耗时。
#
# 在 GPU 服务器上（建议 4 卡）直接跑：
#   cd /vla/workspace/my_tbot
#   NPROC_PER_NODE=4 bash tools/bpv5/eval_shared_pairing_init.sh
#
# 常用环境变量：
#   MODES=list,shared_bucket     # 默认两边都跑；可改成只跑一边
#   MAX_TRAJECTORIES=2000        # 部分轨迹，0=不截断（会很慢）
#   STEPS=6                      # 训练步数，只需过完 init
#   NPROC_PER_NODE=4
#   MASTER_PORT=29741
#   FOREGROUND=1                 # 默认前台；挂后台用 FOREGROUND=0
#   CONFIG=configs/bpv5/stage_b/v2/pairing_ab_partial.jsonc
#
# 日志：
#   outputs/BPv5/bv2/pairing_ab/logs/<mode>_<stamp>.log
#   outputs/BPv5/bv2/pairing_ab/logs/summary_<stamp>.txt
# 关注日志里的 init_timing / split_pair_build / pairing_storage。

set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1

CONFIG=${CONFIG:-configs/bpv5/stage_b/v2/pairing_ab_partial.jsonc}
MODES=${MODES:-list,shared_bucket}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-2000}
STEPS=${STEPS:-6}
MASTER_PORT=${MASTER_PORT:-29741}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
FOREGROUND=${FOREGROUND:-1}
export FOREGROUND

[ -f "$CONFIG" ] || { echo "missing config: $CONFIG"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU (nvidia-smi -L failed); run this on a GPU node"; exit 1; }

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR=$REPO/outputs/BPv5/bv2/pairing_ab/logs
mkdir -p "$LOG_DIR"
SUMMARY=$LOG_DIR/summary_${STAMP}.txt

{
  echo "stamp=$STAMP host=$(hostname)"
  echo "config=$CONFIG"
  echo "modes=$MODES max_trajectories=$MAX_TRAJECTORIES steps=$STEPS"
  echo "nproc=$NPROC_PER_NODE nnodes=$NNODES master_port=$MASTER_PORT"
  echo "----"
} | tee "$SUMMARY"

IFS=',' read -r -a MODE_ARR <<< "$MODES"
FAIL=0
for MODE in "${MODE_ARR[@]}"; do
  MODE=$(echo "$MODE" | tr -d '[:space:]')
  [ -n "$MODE" ] || continue
  case "$MODE" in
    list|shared_bucket) ;;
    *) echo "bad MODE=$MODE (use list or shared_bucket)"; exit 1 ;;
  esac

  OUT_DIR=$REPO/outputs/BPv5/bv2/pairing_ab/${MODE}_${STAMP}
  RUN_LOG=$LOG_DIR/${MODE}_${STAMP}.log
  echo ""
  echo "======== RUN pairing_storage=$MODE ========" | tee -a "$SUMMARY"
  echo "log=$RUN_LOG" | tee -a "$SUMMARY"
  echo "output_dir=$OUT_DIR" | tee -a "$SUMMARY"

  # 复用现有 accelerate 启动器；额外 override 打开计时与部分数据。
  set +e
  NPROC_PER_NODE="$NPROC_PER_NODE" \
  NNODES="$NNODES" \
  NODE_RANK="$NODE_RANK" \
  MASTER_PORT="$MASTER_PORT" \
  FOREGROUND=1 \
  bash tools/bpv5/train_stage_a_dist.sh "$CONFIG" \
    --dataset.pairing_storage="$MODE" \
    --dataset.init_timing=true \
    --dataset.max_trajectories="$MAX_TRAJECTORIES" \
    --dataset.heldout_monitor=false \
    --ddp_find_unused_parameters=true \
    --steps="$STEPS" \
    --save_checkpoint=false \
    --wandb.enable=false \
    --output_dir="$OUT_DIR" \
    2>&1 | tee "$RUN_LOG"
  RC=${PIPESTATUS[0]}
  set -e

  echo "exit_code=$RC mode=$MODE" | tee -a "$SUMMARY"
  # 抽关键计时行，方便直接看对比
  echo "--- timing snippets ($MODE) ---" | tee -a "$SUMMARY"
  grep -nE "init_timing|split_pair_build|pairing_storage|partner_map|SharedPartner|build_partner|construct_total|rank_partition|scan_|M1 partners" "$RUN_LOG" \
    | tee -a "$SUMMARY" || true
  if [ "$RC" -ne 0 ]; then
    FAIL=1
    echo "WARN: mode=$MODE failed rc=$RC; continue next mode if any" | tee -a "$SUMMARY"
  fi
done

echo "" | tee -a "$SUMMARY"
echo "done. summary=$SUMMARY" | tee -a "$SUMMARY"
echo "compare: grep -nE 'init_timing|split_pair_build|M1 partners' $LOG_DIR/*_${STAMP}.log"
exit "$FAIL"
