#!/bin/bash
# G9 on BP RoboTwin for every N-th checkpoint of a run, in parallel over the GPUs, then a summary table.
# usage: source tools/bpv4cdxb/env.local.sh; bash tools/bpv4cdxb/g9_bprobotwin.sh <run>/checkpoints [every_steps=4000]
# env: G9_OUT (default outputs/BPv4CDX/stage_b/g9_bp), G9_QUERIES (600), G9_GPUS (0,1,2,3), G9_WORKERS (6), G9_BATCH (8)
set -u
CKPTS=$1
EVERY=${2:-4000}
cd "$BPV4CDXB_REPO" || exit 1
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
OUT=${G9_OUT:-$BPV4CDXB_REPO/outputs/BPv4CDX/stage_b/g9_bp}
IFS=, read -r -a GPUS <<< "${G9_GPUS:-0,1,2,3}"
[ -f "$OUT/g9_index.json" ] || python tools/bpv4cdxb/g9_bprobotwin.py prepare --out "$OUT" || exit 1
mkdir -p "$OUT/results" "$OUT/logs"
i=0; pids=()
for d in "$CKPTS"/[0-9]*; do
  step=$((10#$(basename "$d")))
  (( step % EVERY == 0 )) || continue
  name=$(printf "step%06d" "$step")
  [ -f "$OUT/results/$name.json" ] && { echo "skip $name (done)"; continue; }
  gpu=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1))
  echo "G9 $name on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.policies.BPV4CDXB.gate_g9_main --ckpt "$d/pretrained_model" \
    --index "$OUT/g9_index.json" --groups "$OUT/g9_groups.json" --out "$OUT/results" --name "$name" \
    --queries "${G9_QUERIES:-600}" --workers "${G9_WORKERS:-6}" --batch "${G9_BATCH:-8}" > "$OUT/logs/$name.log" 2>&1 &
  pids+=($!)
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
python tools/bpv4cdxb/g9_bprobotwin.py summary --out "$OUT"
exit $status
