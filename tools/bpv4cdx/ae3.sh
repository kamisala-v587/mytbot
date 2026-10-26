#!/bin/bash
# bpv4cdx AE-3 steps (plan: code/conclusion/bpv4CDX/bpv4cdx AE-3 计划 v1.md). `source tools/bpv4cdx/env.b200.sh` first.
# usage: bash tools/bpv4cdx/ae3.sh <step> [args]
#   selfcheck            environment, data list, norm stats per robot type, URDF, disk
#   manifest             split manifest (held-out episodes) from $BPV4CDX_REPO_FILE -> $BPV4CDX_MANIFEST (once, then copy)
#   arrays               per-repo memory-mapped state / action arrays -> $BPV4CDX_ARRAY_STORE
#   audit                G0 sampler audit (S-U vs S-UF gripper-event hit rate) -> decision for BPV4CDX_AE3_SAMPLER
#   configs              generated_ae3/{ae3_main,ae3_g3_overfit,ae3_smoke}.json
#   cache                P5 ViT cache, NUM_GPUS x CACHE_PROCS_PER_GPU processes, held-out episodes first (resumable)
#   index | progress     cache index for training on a partial cache / progress per source
#   bench [16 32 64]     per-GPU batch size speed / memory test (200 steps each) before choosing the batch
#   train [variant]      training (default ae3_main; ae3_g3_overfit runs on 1 GPU); refreshes the cache index first
#   g3 | g1g2 | g47      evaluations on this machine (G3 overfit, G1/G2 data checks, G4-G7 on held-out)
set -u
STEP=$1
shift
cd "$BPV4CDX_REPO" || exit 1
SAMPLER=${BPV4CDX_AE3_SAMPLER:-suf}
CACHE_DIR=$BPV4CDX_CACHE_ROOT/bpv4cdx_ae3_$SAMPLER
GEN=${BPV4CDX_AE3_GEN:-$BPV4CDX_REPO/configs/Pro6k/bpv4cdx_plan/generated_ae3}
OUT=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3
LOGS=$OUT/logs
mkdir -p "$LOGS"
DATA=(--manifest "$BPV4CDX_MANIFEST" --data_roots "$BPV4CDX_DATA_ROOTS")
M="python -m lerobot.policies.BPV4CDX"

case $STEP in
  selfcheck)
    python tools/bpv4cdx/selfcheck_ae3.py --repo_file "$BPV4CDX_REPO_FILE" --qwen_path "$BPV4CDX_QWEN_PATH" \
      --stats_root "$BPV4CDX_STATS_ROOT" --cache_root "$BPV4CDX_CACHE_ROOT" --array_store "$BPV4CDX_ARRAY_STORE" \
      --min_gpus "$BPV4CDX_NUM_GPUS" --out "$OUT/selfcheck_ae3.json" "$@" ;;
  manifest)
    $M.ae3_data manifest --repo_file "$BPV4CDX_REPO_FILE" --out "$BPV4CDX_MANIFEST" "$@" ;;
  arrays)
    $M.ae3_data prepare-arrays "${DATA[@]}" --store "$BPV4CDX_ARRAY_STORE" --workers 64 "$@" ;;
  audit)
    $M.ae3_data audit "${DATA[@]}" --array_store "$BPV4CDX_ARRAY_STORE" --repos_per_source 40 --out "$OUT/g0_audit.json" "$@" ;;
  configs)
    python tools/bpv4cdx/make_ae3_configs.py --manifest "$BPV4CDX_MANIFEST" --data_roots "$BPV4CDX_DATA_ROOTS" \
      --cache_root "$BPV4CDX_CACHE_ROOT" --array_store "$BPV4CDX_ARRAY_STORE" --qwen_path "$BPV4CDX_QWEN_PATH" \
      --stats_root "$BPV4CDX_STATS_ROOT" --out_root "$BPV4CDX_OUT_ROOT" --sampler "$SAMPLER" \
      --num_gpus "$BPV4CDX_NUM_GPUS" ${BPV4CDX_AE3_DECAY_LR:+--decay_lr "$BPV4CDX_AE3_DECAY_LR"} --out_dir "$GEN" "$@" ;;
  cache)
    PER=${BPV4CDX_CACHE_PROCS_PER_GPU:-2}
    WORLD=$((BPV4CDX_NUM_GPUS * PER))
    mkdir -p "$LOGS/cache"
    pids=()
    for ((r = 0; r < WORLD; r++)); do
      CUDA_VISIBLE_DEVICES=$((r % BPV4CDX_NUM_GPUS)) $M.build_cache "${DATA[@]}" --array_store "$BPV4CDX_ARRAY_STORE" \
        --cache_dir "$CACHE_DIR" --sampler "$SAMPLER" --qwen_path "$BPV4CDX_QWEN_PATH" --store_spatial 5 \
        --episodes_per_batch "${BPV4CDX_CACHE_EPISODES_PER_BATCH:-4}" --order heldout_first \
        --num_workers "${BPV4CDX_CACHE_WORKERS:-12}" --rank $r --world $WORLD "$@" > "$LOGS/cache/rank$r.log" 2>&1 &
      pids+=($!)
    done
    status=0
    for p in "${pids[@]}"; do wait "$p" || status=1; done
    echo "cache finished (status $status): $CACHE_DIR; failures: $CACHE_DIR/_failures"
    exit $status ;;
  index)
    $M.ae3_data cache-index "${DATA[@]}" --cache_dir "$CACHE_DIR" --out "$CACHE_DIR/_index.txt" "$@" ;;
  progress)
    $M.ae3_data progress "${DATA[@]}" --cache_dir "$CACHE_DIR" "$@" ;;
  train)
    NAME=${1:-ae3_main}
    [ $# -gt 0 ] && shift
    $M.ae3_data cache-index "${DATA[@]}" --cache_dir "$CACHE_DIR" --out "$CACHE_DIR/_index.txt" || exit 1
    NPROC=$BPV4CDX_NUM_GPUS
    [ "$NAME" = "ae3_g3_overfit" ] && NPROC=1
    accelerate launch --num_processes="$NPROC" --main_process_port "${BPV4CDX_PORT:-29650}" \
      -m lerobot.scripts.lerobot_train --config_path="$GEN/$NAME.json" "$@" \
      2>&1 | tee "$LOGS/train_${NAME}_$(date +%m%d_%H%M).log" ;;
  bench)
    # per-GPU batch 16 / 32 / 64 (or the given sizes): 200 steps each, no saves, no HOM; step time, data time, memory
    $M.ae3_data cache-index "${DATA[@]}" --cache_dir "$CACHE_DIR" --out "$CACHE_DIR/_index.txt" || exit 1
    SIZES=${*:-16 32 64}
    for BS in $SIZES; do
      LOG=$LOGS/bench_bs$BS.log
      (while sleep 5; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; done) > "$LOGS/bench_bs$BS.mem" &
      MON=$!
      accelerate launch --num_processes="$BPV4CDX_NUM_GPUS" --main_process_port "${BPV4CDX_PORT:-29650}" \
        -m lerobot.scripts.lerobot_train --config_path="$GEN/ae3_main.json" --batch_size=$BS --steps=200 \
        --save_checkpoint=false --dataset.heldout_monitor=false --policy.scheduler_warmup_steps=20 \
        --output_dir="$OUT/bench/bs$BS" > "$LOG" 2>&1
      kill $MON
      python - "$LOG" "$LOGS/bench_bs$BS.mem" "$BS" "$BPV4CDX_NUM_GPUS" <<'EOF'
import re, sys
log, mem, bs, gpus = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = open(log, errors="ignore").read()
rows = [(int(s), float(u), float(d)) for s, u, d in re.findall(r"(\d+)/200 \[.*?updt_s=([\d.]+), data_s=([\d.]+)", text)]
rows = [r for r in rows if r[0] > 50] or rows
peak = max((int(x) for x in open(mem).read().split() if x.isdigit()), default=0)
if not rows:
    why = "out of GPU memory" if "OutOfMemoryError" in text else "failed"
    print(f"batch {bs}/GPU: {why}, see {log}"); sys.exit()
u = sum(r[1] for r in rows) / len(rows); d = sum(r[2] for r in rows) / len(rows)
print(f"batch {bs}/GPU: update {u:.3f} s, data {d:.3f} s per step -> {bs * gpus / max(u, d):.0f} trajectories/s "
      f"(x4 segments), peak GPU memory {peak / 1024:.1f} GB{' [data-bound]' if d > u else ''}")
EOF
    done ;;
  g3)
    CKPT=$(ls -d "$OUT"/ae3_g3_overfit/*/checkpoints/001000/pretrained_model 2>/dev/null | tail -1)
    [ -z "$CKPT" ] && CKPT=$(ls -d "$OUT"/ae3_g3_overfit/checkpoints/001000/pretrained_model)
    $M.gate_g3 --ckpt "$CKPT" --out "$OUT/g3" "$@" ;;
  g1g2)
    $M.gate_g1_g2 --config "$GEN/ae3_main.json" --out "$OUT/g1_g2" "$@" ;;
  g47)
    # usage: ae3.sh g47 <ckpt pretrained_model dir> <name> [--max_episodes 2000 --sources ...]
    CKPT=$1; NAME=$2; shift 2
    $M.gate_g4_g7 --ckpt "$CKPT" --protocol v2 --split val --out "$OUT/g4_g7" --name "$NAME" "$@" ;;
  *)
    echo "unknown step $STEP"; exit 2 ;;
esac
