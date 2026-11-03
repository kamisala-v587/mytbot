#!/bin/bash
# BP RoboTwin G9 (lerobot.policies.BPV4CDXB.gate_g9_bp) for several checkpoints of one run, in parallel.
# usage: source tools/bpv4cdxb/env.local.sh; bash tools/bpv4cdxb/run_g9_bp.sh <run dir> [steps] [gpus]
#   steps: space-separated checkpoint steps (default "4000 8000 12000 16000 20000")
#   gpus:  space-separated GPU ids, round-robin over the checkpoints (default "1 2 3")
# results: outputs/BPv4CDX/stage_b/g9_bp/<run name>_<step>k.json (+ _rows.jsonl), logs in .../g9_bp/logs/
set -u
RUN=${1:?run dir}
STEPS=${2:-"4000 8000 12000 16000 20000"}
GPUS=(${3:-1 2 3})
cd "$BPV4CDXB_REPO" || exit 1
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
OUT=$BPV4CDXB_OUT_ROOT/BPv4CDX/stage_b/g9_bp
INDEX=$OUT/b4_sft_g9_index.json
mkdir -p "$OUT/logs"
[ -f "$INDEX" ] || python -m lerobot.policies.BPV4CDXB.gate_g9_bp index --out "$INDEX" || exit 1
NAME=$(basename "$(dirname "$RUN")")
i=0
pids=()
for s in $STEPS; do
  ck=$(printf "%s/checkpoints/%06d/pretrained_model" "$RUN" "$s")
  tag=${NAME}_$((s / 1000))k
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}
  CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.policies.BPV4CDXB.gate_g9_bp run --ckpt "$ck" --index "$INDEX" \
    --out "$OUT" --name "$tag" > "$OUT/logs/$tag.log" 2>&1 &
  pids+=($!)
  echo "$tag -> GPU $gpu (log $OUT/logs/$tag.log)"
  i=$((i + 1))
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "G9 finished (status $status): $OUT"
exit $status
