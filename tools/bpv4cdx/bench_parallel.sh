#!/bin/bash
# Per-GPU batch speed / memory test, all sizes at once: each size on its own 2 GPUs, RoboTwin repos only (fast
# dataset load; the step cost does not depend on the source), 200 steps, no saves, no HOM.
# usage: source tools/bpv4cdx/env.b200.sh; bash tools/bpv4cdx/bench_parallel.sh [16 32 64]
set -u
cd "$BPV4CDX_REPO" || exit 1
SIZES=(${@:-16 32 64})
GEN=$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_ae3
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3
LOGS=$OUT/logs
mkdir -p "$LOGS"
pids=()
mons=()
for i in "${!SIZES[@]}"; do
  BS=${SIZES[$i]}
  GPUS="$((2 * i)),$((2 * i + 1))"
  rm -rf "$OUT/pbench/bs$BS"
  (while sleep 5; do nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits; done) > "$LOGS/pbench_bs$BS.mem" &
  mons+=($!)
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --num_processes=2 --main_process_port $((29700 + i)) \
    -m lerobot.scripts.lerobot_train --config_path="$GEN/ae3_main.json" --batch_size=$BS --steps=200 \
    --save_checkpoint=false --dataset.heldout_monitor=false --policy.scheduler_warmup_steps=20 \
    '--dataset.manifest_sources=["RoboTwin-LeRobot-v3.0"]' \
    --output_dir="$OUT/pbench/bs$BS" > "$LOGS/pbench_bs$BS.log" 2>&1 &
  pids+=($!)
  echo "batch $BS/GPU on GPUs $GPUS -> $LOGS/pbench_bs$BS.log"
done
for p in "${pids[@]}"; do wait "$p"; done
kill "${mons[@]}" 2>/dev/null
for BS in "${SIZES[@]}"; do
  python - "$LOGS/pbench_bs$BS.log" "$LOGS/pbench_bs$BS.mem" "$BS" "$BPV4CDX_NUM_GPUS" <<'EOF'
import re, sys
log, mem, bs, gpus = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = open(log, errors="ignore").read()
rows = [(int(s), float(u), float(d)) for s, u, d in re.findall(r"(\d+)/200 \[.*?updt_s=([\d.]+), data_s=([\d.]+)", text)]
rows = [r for r in rows if r[0] > 50] or rows
peak = max((int(x) for x in open(mem).read().split() if x.isdigit()), default=0)
if not rows:
    print(f"batch {bs}/GPU: failed, see {log}"); sys.exit()
u = sum(r[1] for r in rows) / len(rows); d = sum(r[2] for r in rows) / len(rows)
print(f"batch {bs}/GPU: update {u:.3f} s, data {d:.3f} s per step -> {bs * gpus / max(u, d):.0f} trajectories/s on "
      f"{gpus} GPUs (x4 segments), peak GPU memory {peak / 1024:.1f} GB, 300k steps ~{300000 * max(u, d) / 3600:.0f} h")
EOF
done
