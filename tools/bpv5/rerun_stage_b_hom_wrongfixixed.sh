#!/usr/bin/env bash
# Re-run Stage B HOM under fixed wrong pairing (same source / prefer same robot / diff task).
# Writes heldout_monitor_wrongfixixed_{nolang,lang}.jsonl — does not overwrite legacy files.
set -euo pipefail
ROOT=/vla/workspace/my_tbot
source "$ROOT/tools/bpv4cdxb/env.local.sh"
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/vla/workspace/shiliangchao/env/q35va/bin/python
LOGDIR="$ROOT/outputs/BPv5/stage_b/logs"
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOGDIR/hom_wrongfixixed_${STAMP}.log"

run_one() {
  local gpu="$1" run="$2" steps="$3" mode="$4" max_ep="${5:-16}"
  local out lang_flag
  if [[ "$mode" == "nolang" ]]; then
    out="heldout_monitor_wrongfixixed_nolang.jsonl"
    lang_flag="--pad_language"
  else
    out="heldout_monitor_wrongfixixed_lang.jsonl"
    lang_flag="--keep_language"
  fi
  echo "[gpu$gpu] $run steps=$steps mode=$mode" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -m lerobot.policies.BPV5StageB.heldout_monitor \
    --run "$run" --steps "$steps" --max_episodes "$max_ep" \
    $lang_flag --overwrite --out_name "$out" \
    >>"$LOG" 2>&1
  echo "[gpu$gpu] DONE $run $mode" | tee -a "$LOG"
}

SB="$ROOT/outputs/BPv5/stage_b"

# Wave 1: four GPUs in parallel (pad-language / identity)
run_one 0 "$SB/pb3/2026-11-20_10-00" 10000 nolang &
run_one 1 "$SB/pb_rt_vobs/2026-11-20_21-27" 3000 nolang &
run_one 2 "$SB/pb3_4src/2026-11-20_16-37" 3000 nolang &
run_one 3 "$SB/pb_rt_mix50k/2026-11-20_23-34" 39000 nolang &
wait

# Wave 2
run_one 0 "$SB/pb3_4src_c1/2026-11-20_20-15" 1000 nolang &
run_one 1 "$SB/pb_fp_notext30k/2026-11-22_00-22" 12000 nolang &
run_one 2 "$SB/ae_then_b3/2026-11-22_20-29" 2000 nolang &
run_one 3 "$SB/ae_then_b3/2026-11-22_23-33" 13000 nolang &
wait

# Wave 3: baselines + with-language for key RT / 4src claims
run_one 0 "$SB/b1/2026-11-19_02-10" 10000 nolang &
run_one 1 "$SB/pb2_1/2026-11-20_00-41" 9000 nolang &
run_one 2 "$SB/pb3/2026-11-20_10-00" 10000 lang &
run_one 3 "$SB/pb_rt_vobs/2026-11-20_21-27" 3000 lang &
wait

# Wave 4: with-language for 4src identity claim
run_one 0 "$SB/pb3_4src/2026-11-20_16-37" 3000 lang &
run_one 1 "$SB/pb_rt_mix50k/2026-11-20_23-34" 39000 lang &
wait

echo "ALL DONE log=$LOG" | tee -a "$LOG"
