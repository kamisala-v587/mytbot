#!/usr/bin/env bash
# Offline HOM over Stage A checkpoints (matched / blank / alt / wrong), 1 GPU per shard.
# Usage:
#   bash tools/bpv5/run_stage_a_hom_offline.sh \
#     /vla/workspace/my_tbot/outputs/BPv5/stage_a/main/2026-11-13_14-46 [max_episodes]
set -euo pipefail
ROOT=/vla/workspace/my_tbot
source "$ROOT/tools/bpv4cdxb/env.local.sh"
RUN="${1:?run dir with checkpoints/}"
MAX_EP="${2:-32}"
OUT_NAME="${3:-heldout_monitor.jsonl}"
PY=/vla/workspace/shiliangchao/env/q35va/bin/python
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

mapfile -t STEPS < <(ls -1 "$RUN/checkpoints" | grep -E '^[0-9]+$' | sort -n)
if ((${#STEPS[@]} == 0)); then
  echo "no numeric checkpoints under $RUN/checkpoints" >&2
  exit 1
fi
NGPU=$(nvidia-smi -L | wc -l)
echo "run=$RUN steps=${STEPS[*]} max_ep=$MAX_EP gpus=$NGPU"

tmpdir="$RUN/.hom_shards_$$"
mkdir -p "$tmpdir"
pids=()
i=0
for step in "${STEPS[@]}"; do
  gpu=$((i % NGPU))
  shard="$tmpdir/gpu${gpu}.jsonl"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "[gpu $gpu] step $step"
    "$PY" -m lerobot.policies.BPV5QFormer.heldout_monitor \
      --run "$RUN" --max_episodes "$MAX_EP" --steps "$step" \
      --alternate --out_name ".hom_shards_$$/gpu${gpu}_${step}.jsonl"
  ) >"$tmpdir/log_${step}.txt" 2>&1 &
  pids+=($!)
  i=$((i + 1))
  # keep at most NGPU jobs in flight
  if ((${#pids[@]} >= NGPU)); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]:-}"; do wait "$pid"; done

# merge shard lines (one record per step) into OUT_NAME
"$PY" - <<PY
import json
from pathlib import Path
run = Path("$RUN")
tmpdir = Path("$tmpdir")
rows = []
for p in sorted(tmpdir.glob("gpu*_*.jsonl")):
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
rows.sort(key=lambda r: int(r["step"]))
# keep last offline record per step
by = {}
for r in rows:
    by[int(r["step"])] = r
out = run / "$OUT_NAME"
with out.open("w", encoding="utf-8") as f:
    for step in sorted(by):
        f.write(json.dumps(by[step], ensure_ascii=False) + "\n")
print(f"wrote {len(by)} records -> {out}")
for step, r in sorted(by.items()):
    m = r.get("mse_matched")
    print(f"  {step}: matched={m}")
PY

echo "logs: $tmpdir/log_*.txt"
