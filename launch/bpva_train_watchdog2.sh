#!/usr/bin/env bash
set -uo pipefail

# BPVA 训练守护脚本：启动训练、检查 loss，并从最新完整 checkpoint 自动恢复。
# 推荐启动：
#   cd /vla/workspace/my_tbot
#   nohup bash launch/bpva_train_watchdog2.sh > logs/bpva_watchdog2.nohup.log 2>&1 &
#
# 可选环境变量：
#   CUDA_DEVICE=0,1          使用的 GPU（当前节点四卡 NCCL 异常，默认两卡 0,1）
#   NUM_PROCESSES=2          accelerate 进程数（默认 2）
#   GRAD_ACCUM_STEPS=2       梯度累积步数（两卡维持有效 batch 32）
#   CHECK_INTERVAL=60      检查间隔，秒（默认 60）
#   STALE_THRESHOLD=1800   loss.log 无更新多久判定卡死，秒（默认 1800）
#   STARTUP_GRACE=1800     每次启动后等待首条新 loss 的宽限期，秒（默认 1800）
#   MAX_ABS_LOSS=1000      loss 绝对值上限；设为 0 仅检查 NaN/Inf（默认 1000）
#   RESTART_DELAY=60       重启前等待秒数（默认 60）

PROJECT_ROOT="/vla/workspace/my_tbot"
INITIAL_CONFIG="${PROJECT_ROOT}/configs/bpva_train2.jsonc"
OUTPUT_BASE="${PROJECT_ROOT}/outputs/BPVA/SFT-rand2"
LOG_DIR="${PROJECT_ROOT}/logs"
CONDA_SH="/vla/.conda/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="mytbot"

CUDA_DEVICE="${CUDA_DEVICE:-0,1}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
CHECK_INTERVAL="${CHECK_INTERVAL:-60}"
STALE_THRESHOLD="${STALE_THRESHOLD:-1800}"
STARTUP_GRACE="${STARTUP_GRACE:-1800}"
MAX_ABS_LOSS="${MAX_ABS_LOSS:-1000}"
RESTART_DELAY="${RESTART_DELAY:-60}"

mkdir -p "${LOG_DIR}"
cd "${PROJECT_ROOT}" || exit 1

# 防止同一个守护脚本被重复启动；不会干扰其他训练脚本。
exec 9>"/tmp/my_tbot_bpva_train_watchdog2.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T')] 已有 bpva_train_watchdog2.sh 在运行，退出。"
  exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
  echo "[$(date '+%F %T')] 找不到 Conda 初始化脚本：${CONDA_SH}"
  exit 1
fi
# shellcheck disable=SC1090
source "${CONDA_SH}"
conda activate "${CONDA_ENV}" || exit 1

export HF_HOME=/vla/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

# 当前节点 NCCL 2.26 的四卡 SHM 通信会触发 CUDA 700；已用两卡 Socket smoke 验证。
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"

WATCHDOG_LOG="${LOG_DIR}/bpva_watchdog2_$(date '+%F_%H-%M-%S').log"
CHILD_PID=""
CURRENT_RUN=""
ATTEMPT=0
STOP_REQUESTED=0

log() {
  local message="[$(date '+%F %T')] $*"
  echo "${message}" | tee -a "${WATCHDOG_LOG}"
}

is_uint() {
  [[ "$1" =~ ^[0-9]+$ ]]
}

for value_name in CHECK_INTERVAL STALE_THRESHOLD STARTUP_GRACE RESTART_DELAY NUM_PROCESSES GRAD_ACCUM_STEPS; do
  value="${!value_name}"
  if ! is_uint "${value}"; then
    log "配置错误：${value_name}=${value}，必须是非负整数。"
    exit 1
  fi
done

stop_child() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    log "正在停止训练进程组 pid=${CHILD_PID}"
    kill -TERM -- "-${CHILD_PID}" 2>/dev/null || kill -TERM "${CHILD_PID}" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "${CHILD_PID}" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "${CHILD_PID}" 2>/dev/null; then
      log "训练进程未及时退出，发送 KILL。"
      kill -KILL -- "-${CHILD_PID}" 2>/dev/null || kill -KILL "${CHILD_PID}" 2>/dev/null || true
    fi
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  CHILD_PID=""
}

on_signal() {
  STOP_REQUESTED=1
  log "收到停止信号，守护脚本将退出且不再恢复训练。"
  stop_child
  exit 130
}
trap on_signal INT TERM HUP

latest_run_dir() {
  # 只接受本次启动后创建/更新的目录，避免误监控之前遗留的训练。
  local not_before="${1:-0}"
  local newest="" newest_mtime=0 candidate mtime
  shopt -s nullglob
  for candidate in "${OUTPUT_BASE}"/????-??-??/??-??-??_bpva_train; do
    [[ -d "${candidate}" ]] || continue
    mtime=$(stat -c %Y "${candidate}" 2>/dev/null || echo 0)
    if (( mtime >= not_before && mtime > newest_mtime )); then
      newest_mtime=${mtime}
      newest="${candidate}"
    fi
  done
  shopt -u nullglob
  printf '%s' "${newest}"
}

checkpoint_is_complete() {
  local checkpoint="$1"
  local model_files=("${checkpoint}"/pretrained_model/*.safetensors)
  [[ -s "${checkpoint}/pretrained_model/train_config.json" ]] || return 1
  [[ -s "${checkpoint}/training_state/training_step.json" ]] || return 1
  [[ -s "${checkpoint}/training_state/optimizer_state.safetensors" ]] || return 1
  [[ -e "${model_files[0]}" && -s "${model_files[0]}" ]] || return 1
}

latest_checkpoint() {
  local run_dir="$1" checkpoint_root checkpoint
  local checkpoints=()
  checkpoint_root="${run_dir}/checkpoints"
  [[ -d "${checkpoint_root}" ]] || return 1

  shopt -s nullglob
  checkpoints=("${checkpoint_root}"/[0-9]*)
  shopt -u nullglob
  ((${#checkpoints[@]} > 0)) || return 1

  while IFS= read -r checkpoint; do
    if [[ "$(basename "${checkpoint}")" =~ ^[0-9]+$ ]] && checkpoint_is_complete "${checkpoint}"; then
      printf '%s' "${checkpoint}"
      return 0
    fi
  done < <(printf '%s\n' "${checkpoints[@]}" | sort -r)
  return 1
}

loss_status() {
  # 输出 OK、MISSING、EMPTY、NO_DATA、MALFORMED、NONFINITE 或 TOO_LARGE。
  local loss_log="$1"
  [[ -f "${loss_log}" ]] || { echo MISSING; return; }
  [[ -s "${loss_log}" ]] || { echo EMPTY; return; }

  awk -F',' -v max_abs="${MAX_ABS_LOSS}" '
    NR == 1 {
      for (i = 1; i <= NF; i++) if ($i == "loss") loss_col = i
      next
    }
    NF > 1 { last = $0 }
    END {
      if (!loss_col) { print "MALFORMED"; exit }
      if (last == "") { print "NO_DATA"; exit }
      n = split(last, fields, ",")
      value = fields[loss_col]
      lower = tolower(value)
      if (lower ~ /nan|inf/) { print "NONFINITE"; exit }
      if (value !~ /^[-+]?([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][-+]?[0-9]+)?$/) {
        print "MALFORMED"; exit
      }
      numeric = value + 0
      if ((max_abs + 0) > 0 && (numeric > max_abs || numeric < -max_abs)) {
        print "TOO_LARGE"; exit
      }
      print "OK"
    }
  ' "${loss_log}"
}

last_loss_summary() {
  local loss_log="$1"
  [[ -f "${loss_log}" ]] || { echo "loss.log 不存在"; return; }
  awk -F',' '
    NR == 1 {
      for (i = 1; i <= NF; i++) {
        if ($i == "step") step_col = i
        if ($i == "timestamp") time_col = i
        if ($i == "loss") loss_col = i
      }
      next
    }
    NF > 1 { step=$step_col; ts=$time_col; loss=$loss_col }
    END {
      if (step == "") print "尚无 loss 记录"
      else printf "step=%s timestamp=%s loss=%s", step, ts, loss
    }
  ' "${loss_log}"
}

start_training() {
  local checkpoint="$1" attempt_log config_path
  local command=(accelerate launch --num_processes="${NUM_PROCESSES}" -m lerobot.scripts.lerobot_train)
  ATTEMPT=$((ATTEMPT + 1))
  attempt_log="${LOG_DIR}/bpva_train2_attempt_$(printf '%03d' "${ATTEMPT}")_$(date '+%F_%H-%M-%S').log"

  if [[ -n "${checkpoint}" ]]; then
    config_path="${checkpoint}/pretrained_model/train_config.json"
    command+=(--resume=true --config_path="${config_path}" --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}")
    CURRENT_RUN="$(dirname "$(dirname "${checkpoint}")")"
    log "第 ${ATTEMPT} 次启动：从 checkpoint 恢复：${checkpoint}"
  else
    command+=(--config_path="${INITIAL_CONFIG}" --gradient_accumulation_steps="${GRAD_ACCUM_STEPS}")
    CURRENT_RUN=""
    log "第 ${ATTEMPT} 次启动：没有可恢复 checkpoint，从初始配置开始。"
  fi
  log "本次训练控制台日志：${attempt_log}"

  CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" setsid "${command[@]}" >>"${attempt_log}" 2>&1 &
  CHILD_PID=$!
  log "训练进程组已启动，pid=${CHILD_PID}，GPU=${CUDA_DEVICE}"
}

monitor_attempt() {
  local started_at="$1" baseline_loss_mtime="$2"
  local now age loss_log train_log status current_mtime

  while kill -0 "${CHILD_PID}" 2>/dev/null; do
    if (( STOP_REQUESTED )); then
      return 2
    fi

    if [[ -z "${CURRENT_RUN}" ]]; then
      CURRENT_RUN="$(latest_run_dir "${started_at}")"
      if [[ -n "${CURRENT_RUN}" ]]; then
        log "检测到本次输出目录：${CURRENT_RUN}"
      fi
    fi

    if [[ -n "${CURRENT_RUN}" ]]; then
      loss_log="${CURRENT_RUN}/loss.log"
      train_log="${CURRENT_RUN}/train.log"

      if [[ -f "${train_log}" ]] && rg -q "End of training" "${train_log}" 2>/dev/null; then
        log "检测到正常结束标志。"
        wait "${CHILD_PID}" 2>/dev/null || true
        CHILD_PID=""
        return 0
      fi

      status="$(loss_status "${loss_log}")"
      case "${status}" in
        NONFINITE|MALFORMED|TOO_LARGE)
          log "loss 异常（${status}）：$(last_loss_summary "${loss_log}")"
          stop_child
          return 1
          ;;
      esac

      now=$(date +%s)
      age=$((now - started_at))
      if [[ -f "${loss_log}" ]]; then
        current_mtime=$(stat -c %Y "${loss_log}" 2>/dev/null || echo 0)
        # 恢复训练时旧 loss.log 已经存在，因此必须等它在本次启动后更新过，
        # 才开始应用常规 stale 检查；否则使用 STARTUP_GRACE。
        if (( current_mtime > baseline_loss_mtime )); then
          if (( now - current_mtime > STALE_THRESHOLD )); then
            log "loss.log 已 $((now - current_mtime)) 秒未更新，判定训练卡死：$(last_loss_summary "${loss_log}")"
            stop_child
            return 1
          fi
        elif (( age > STARTUP_GRACE )); then
          log "启动 ${age} 秒后仍无新 loss，判定训练卡死。"
          stop_child
          return 1
        fi
      elif (( age > STARTUP_GRACE )); then
        log "启动 ${age} 秒后仍未生成 loss.log，判定训练卡死。"
        stop_child
        return 1
      fi
    fi

    sleep "${CHECK_INTERVAL}"
  done

  wait "${CHILD_PID}" 2>/dev/null
  local exit_code=$?
  CHILD_PID=""

  if [[ -n "${CURRENT_RUN}" ]] && [[ -f "${CURRENT_RUN}/train.log" ]] \
      && rg -q "End of training" "${CURRENT_RUN}/train.log" 2>/dev/null; then
    log "训练进程正常结束，exit_code=${exit_code}。"
    return 0
  fi

  log "训练进程意外退出，exit_code=${exit_code}。"
  return 1
}

log "BPVA 训练守护开始。watchdog_log=${WATCHDOG_LOG}"
log "GPU=${CUDA_DEVICE}，进程数=${NUM_PROCESSES}，梯度累积=${GRAD_ACCUM_STEPS}，有效 batch=8x${NUM_PROCESSES}x${GRAD_ACCUM_STEPS}"
log "NCCL 兼容模式：P2P=${NCCL_P2P_DISABLE}，SHM=${NCCL_SHM_DISABLE}，cuMem=${NCCL_CUMEM_ENABLE}，网卡=${NCCL_SOCKET_IFNAME}"
log "检查间隔=${CHECK_INTERVAL}s，loss 超时=${STALE_THRESHOLD}s，启动宽限=${STARTUP_GRACE}s，loss 绝对值上限=${MAX_ABS_LOSS}"

resume_checkpoint=""
while (( ! STOP_REQUESTED )); do
  baseline_loss_mtime=0
  if [[ -n "${resume_checkpoint}" ]]; then
    CURRENT_RUN="$(dirname "$(dirname "${resume_checkpoint}")")"
    if [[ -f "${CURRENT_RUN}/loss.log" ]]; then
      baseline_loss_mtime=$(stat -c %Y "${CURRENT_RUN}/loss.log" 2>/dev/null || echo 0)
    fi
  fi

  start_training "${resume_checkpoint}"
  attempt_started_at=$(date +%s)

  if monitor_attempt "${attempt_started_at}" "${baseline_loss_mtime}"; then
    log "BPVA 训练已完成，守护脚本退出。"
    exit 0
  fi

  if (( STOP_REQUESTED )); then
    exit 130
  fi

  if [[ -z "${CURRENT_RUN}" ]]; then
    CURRENT_RUN="$(latest_run_dir "${attempt_started_at}")"
  fi

  resume_checkpoint=""
  if [[ -n "${CURRENT_RUN}" ]]; then
    resume_checkpoint="$(latest_checkpoint "${CURRENT_RUN}" || true)"
  fi

  if [[ -n "${resume_checkpoint}" ]]; then
    log "将从最近的完整 checkpoint 恢复：${resume_checkpoint}"
  else
    log "当前训练目录中没有完整 checkpoint，只能重新执行初始训练。"
  fi
  log "${RESTART_DELAY} 秒后重启。"
  sleep "${RESTART_DELAY}"
done
