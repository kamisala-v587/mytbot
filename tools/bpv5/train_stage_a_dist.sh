#!/bin/bash
# BPv5 Stage A multi-node / multi-GPU train launcher (one invocation per node).
#
#   bash tools/bpv5/train_stage_a_dist.sh [dist8|main|local2|smoke|/path/to.jsonc] [draccus overrides...]
#
# Env (platform PET_* / torchrun-style aliases accepted):
#   NNODES / PET_NNODES / WORLD_SIZE
#   NODE_RANK / PET_NODE_RANK / RANK
#   MASTER_ADDR / PET_MASTER_ADDR   (default 127.0.0.1)
#   MASTER_PORT / PET_MASTER_PORT / PORT  (default 29700 for dist8; 29660 otherwise)
#   NPROC_PER_NODE   (default = nvidia-smi GPU count)
#   MIXED_PRECISION  (default no; policy already bf16)
#   FOREGROUND       默认 1（前台 tee）。平台任务必须前台，否则 bash 立刻退出会被判成功并杀 pod。
#                    本机想挂后台再 FOREGROUND=0。
#
# Examples:
#   # 平台 / 单机 8 卡（前台，默认）
#   bash tools/bpv5/train_stage_a_dist.sh dist8
#
#   # 两机 × 4 卡（每节点前台各跑一次）
#   NNODES=2 NODE_RANK=0 NPROC_PER_NODE=4 MASTER_ADDR=<ip> MASTER_PORT=29700 bash tools/bpv5/train_stage_a_dist.sh dist8
#   NNODES=2 NODE_RANK=1 NPROC_PER_NODE=4 MASTER_ADDR=<ip> MASTER_PORT=29700 bash tools/bpv5/train_stage_a_dist.sh dist8
#
# Logs (shared FS):
#   outputs/BPv5/stage_a/logs/train_<name>_node<rank>_<stamp>.log
# Tail node0:
#   tail -f /vla/workspace/my_tbot/outputs/BPv5/stage_a/logs/train_dist8_node0_latest.log
set -u
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va
VARIANT=${1:-dist8}
shift || true

cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
unset NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_DISABLE

case "$VARIANT" in
  dist8)  CONFIG=configs/bpv5/stage_a/stage_a_main_dist8.jsonc; NAME=dist8; DEFAULT_PORT=29700 ;;
  main)   CONFIG=configs/bpv5/stage_a/stage_a_main.jsonc; NAME=main; DEFAULT_PORT=29660 ;;
  local2) CONFIG=configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_local2.jsonc; NAME=local2; DEFAULT_PORT=29661 ;;
  smoke)  CONFIG=configs/bpv5/historyOrTestOrEx/2026-11-16/stage_a_smoke.jsonc; NAME=smoke; DEFAULT_PORT=29662 ;;
  *)      CONFIG=$VARIANT; NAME=$(basename "$CONFIG" .jsonc); DEFAULT_PORT=29700 ;;
esac
[ -f "$CONFIG" ] || { echo "missing config: $CONFIG"; exit 1; }

NNODES=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${RANK:-0}}}
MASTER_ADDR=${MASTER_ADDR:-${PET_MASTER_ADDR:-127.0.0.1}}
MASTER_PORT=${MASTER_PORT:-${PET_MASTER_PORT:-${PORT:-$DEFAULT_PORT}}}
MIXED_PRECISION=${MIXED_PRECISION:-no}

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -ge 1 ] || { echo "no GPU (nvidia-smi -L empty)"; exit 1; }
NPROC_PER_NODE=${NPROC_PER_NODE:-$NGPU}
NUM_PROCESSES=$((NNODES * NPROC_PER_NODE))

[ "$NODE_RANK" -lt "$NNODES" ] || { echo "NODE_RANK=$NODE_RANK >= NNODES=$NNODES"; exit 1; }

export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}
DATA_ROOT=$BPV5_DATA_ROOT
STATS_ROOT=$BPV5_STATS_ROOT
for d in "$DATA_ROOT" \
         "$STATS_ROOT/stats_index.json" \
         /vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm \
         "$Q35/bin/python"; do
  [ -e "$d" ] || { echo "missing: $d (mount it)"; exit 1; }
done
# ViT cache may still be building for new sources; warn instead of hard-fail.
CACHE_DIR=${BPV5_CACHE_DIR:-/vla/workspace/cache/bpv5_cleaned_p5}
[ -e "$CACHE_DIR" ] || echo "WARN: missing cache_dir $CACHE_DIR (build before long run)"
python -c "import torch, transformers; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpus', torch.cuda.device_count())" || exit 1

LOG_DIR=$REPO/outputs/BPv5/stage_a/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/train_${NAME}_node${NODE_RANK}_${STAMP}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/train_${NAME}_node${NODE_RANK}_latest.log"

echo "[node $NODE_RANK/$NNODES] host=$(hostname) gpus=$NGPU nproc_per_node=$NPROC_PER_NODE world=$NUM_PROCESSES"
echo "[node $NODE_RANK/$NNODES] master=$MASTER_ADDR:$MASTER_PORT config=$CONFIG"
echo "[node $NODE_RANK/$NNODES] log=$LOG"
echo "[node $NODE_RANK/$NNODES] extras=$*"
echo "[node $NODE_RANK/$NNODES] norm_stats=$STATS_ROOT"
echo "[node $NODE_RANK/$NNODES] data_root=$DATA_ROOT cache=$CACHE_DIR"

CMD=(
  accelerate launch
  --multi_gpu
  --num_processes "$NUM_PROCESSES"
  --num_machines "$NNODES"
  --machine_rank "$NODE_RANK"
  --main_process_ip "$MASTER_ADDR"
  --main_process_port "$MASTER_PORT"
  --mixed_precision "$MIXED_PRECISION"
  -m lerobot.scripts.lerobot_train
  --config_path="$CONFIG"
)

# 默认前台：平台看到「started pid=…」就判成功，是因为旧默认 nohup 后台后 bash 立刻退出。
if [ "${FOREGROUND:-1}" = "0" ]; then
  setsid nohup "${CMD[@]}" "$@" >"$LOG" 2>&1 < /dev/null &
  echo "started pid=$! -> $LOG (background; FOREGROUND=0)"
  echo "tail:  tail -f $LOG"
  if [ "$NODE_RANK" -eq 0 ]; then
    echo "tail node0 latest:  tail -f $LOG_DIR/train_${NAME}_node0_latest.log"
  fi
else
  echo "[node $NODE_RANK/$NNODES] FOREGROUND=1 → blocking until train exits; log=$LOG"
  "${CMD[@]}" "$@" 2>&1 | tee "$LOG"
  exit "${PIPESTATUS[0]}"
fi
