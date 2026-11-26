#!/bin/bash
# BPv5 wall ViT P5 cache — multi-node, pause/resume safe.
#
# Sources under data root (/vla/workspace/cache/bpv5_cleaned_root):
#   agibot / interndata / robotwin(full clean+randomized)
#   droid      <- /physical-ai-data/cleaned_dataset/droid-v3/data_clean2.0-20261114
#   molmoact   <- MolmoAct-DatasetV3.0 tabletop+household train leaves
#   robotwin25 <- /vla/workspace/data/robotwin2.5_bp
#
#   cache dir: /vla/workspace/cache/bpv5_cleaned_p5
#
# 单机 8 卡（含新增源，需重建 items）:
#   cd /vla/workspace/my_tbot
#   FORCE_ITEMS_REBUILD=1 NNODES=1 NODE_RANK=0 PROCS_PER_GPU=2 bash tools/bpv5/build_cache_dist.sh
#
# Progress:
#   tail -f /vla/workspace/cache/bpv5_cleaned_p5/_logs/PROGRESS.txt
#
# Env: NNODES NODE_RANK PROCS_PER_GPU WORKERS EPB GPU_DECODE_PIXELS MAX_RETRY
#      FORCE_ITEMS_REBUILD CACHE_LOG_EVERY
set -u

CACHE=/vla/workspace/cache/bpv5_cleaned_p5
REPO=/vla/workspace/my_tbot
Q35=/vla/workspace/shiliangchao/env/q35va
DATA_ROOT=/vla/workspace/cache/bpv5_cleaned_root
VLM=/vla/workspace/models/wall-oss/wall-oss-1.0-lite-vlm

AGIBOT_SRC=/physical-ai-data/cleaned_dataset/agibotworld/full-20261106-1943
IDN_SRC=/physical-ai-data/cleaned_dataset/full-InternData-A1-v3.0/full-20261114-2126
RT_SRC=/physical-ai-data/cleaned_dataset/RoboTwin-LeRobot-v3.0/full-20261105-1329
DROID_SRC=/physical-ai-data/cleaned_dataset/droid-v3/data_clean2.0-20261114
MOLMO_SRC=/physical-ai-data/download_tmp_dengjun/MolmoAct-DatasetV3.0
RT25_SRC=/vla/workspace/data/robotwin2.5_bp

NNODES=${NNODES:-${PET_NNODES:-${WORLD_SIZE:-1}}}
NODE_RANK=${NODE_RANK:-${PET_NODE_RANK:-${RANK:-0}}}
NGPU=$(nvidia-smi -L | wc -l)
PROCS=${PROCS_PER_GPU:-2}
PER_NODE=$((NGPU * PROCS))
WORLD=$((NNODES * PER_NODE))
NCPU=$(nproc)
AUTO_W=$((NCPU / PER_NODE - 2)); [ "$AUTO_W" -lt 4 ] && AUTO_W=4; [ "$AUTO_W" -gt 16 ] && AUTO_W=16
WORKERS=${WORKERS:-$AUTO_W}
EPB=${EPB:-8}
GPU_DECODE_PIXELS=${GPU_DECODE_PIXELS:-921600}
MAX_RETRY=${MAX_RETRY:-5}
FORCE_ITEMS_REBUILD=${FORCE_ITEMS_REBUILD:-0}
export CACHE_LOG_EVERY=${CACHE_LOG_EVERY:-50}

cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh
export BPV5_DATA_ROOT="$DATA_ROOT"
export LD_LIBRARY_PATH=$Q35/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
unset NCCL_DEBUG NCCL_SOCKET_IFNAME NCCL_IB_DISABLE

echo "[node $NODE_RANK/$NNODES] host=$(hostname) gpus=$NGPU procs/gpu=$PROCS world=$WORLD cpus=$NCPU workers=$WORKERS"
echo "[node $NODE_RANK] cache=$CACHE data_root=$DATA_ROOT sources=agibot,interndata,robotwin(full),droid,molmoact,robotwin25"
[ "$NODE_RANK" -lt "$NNODES" ] || { echo "NODE_RANK >= NNODES, set NNODES/NODE_RANK explicitly"; exit 1; }

ensure_symlinks() {
  mkdir -p "$DATA_ROOT/droid" "$DATA_ROOT/molmoact" "$DATA_ROOT/robotwin25"
  # droid: single LeRobot leaf
  ln -sfn "$DROID_SRC" "$DATA_ROOT/droid/full"
  # molmoact: two train leaves
  ln -sfn "$MOLMO_SRC/molmoact_dataset_tabletop/train" "$DATA_ROOT/molmoact/tabletop"
  ln -sfn "$MOLMO_SRC/molmoact_dataset_household/train" "$DATA_ROOT/molmoact/household"
  # robotwin2.5_bp: task -> leaf (2-level under source)
  for task in "$RT25_SRC"/*; do
    [ -d "$task" ] || continue
    base=$(basename "$task")
    case "$base" in
      .*|README.md|*.json) continue ;;
    esac
    ln -sfn "$task" "$DATA_ROOT/robotwin25/$base"
  done
  # robotwin full: ensure randomized leaves are linked (clean already present)
  if [ -d "$RT_SRC" ]; then
    mkdir -p "$DATA_ROOT/robotwin"
    for task in "$RT_SRC"/*; do
      [ -d "$task" ] || continue
      tname=$(basename "$task")
      case "$tname" in
        _*|.*|cleaning*|*.json) continue ;;
      esac
      mkdir -p "$DATA_ROOT/robotwin/$tname"
      for leaf in "$task"/*; do
        [ -d "$leaf" ] || continue
        [ -f "$leaf/meta/info.json" ] || continue
        ln -sfn "$leaf" "$DATA_ROOT/robotwin/$tname/$(basename "$leaf")"
      done
    done
  fi
}

for d in "$DATA_ROOT/agibot" "$DATA_ROOT/interndata" "$DATA_ROOT/robotwin" \
         "$AGIBOT_SRC" "$IDN_SRC" "$RT_SRC" "$DROID_SRC" "$MOLMO_SRC" "$RT25_SRC" \
         "$VLM" "$Q35/bin/python" \
         /physical-ai-data/norm_stats/action80_new; do
  [ -e "$d" ] || { echo "missing: $d (mount / symlink it)"; exit 1; }
done

ensure_symlinks
python -c "import torch, transformers, torchcodec; assert torch.cuda.is_available(); print('torch', torch.__version__, 'tf', transformers.__version__)" || exit 1

mkdir -p "$CACHE/_logs"
ITEMS=$CACHE/_items.json

if [ "$FORCE_ITEMS_REBUILD" = "1" ] && [ "$NODE_RANK" -eq 0 ]; then
  echo "[node 0] FORCE_ITEMS_REBUILD=1 → remove $ITEMS"
  rm -f "$ITEMS"
fi

if [ ! -f "$ITEMS" ]; then
  if [ "$NODE_RANK" -eq 0 ]; then
    echo "[node 0] building $ITEMS (filesystem walk; robotwin full + droid/molmoact/robotwin25) ..."
    CACHE="$CACHE" BPV5_DATA_ROOT="$DATA_ROOT" python - <<'PY' > "$CACHE/_logs/items.log" 2>&1
from __future__ import annotations
import json, os
from pathlib import Path
from collections import Counter
from lerobot.policies.BPV5QFormer import data80d as D

CACHE = Path(os.environ["CACHE"])
D.configure(root=os.environ["BPV5_DATA_ROOT"])
SOURCES = {"agibot", "interndata", "robotwin", "droid", "molmoact", "robotwin25"}
# Deep filesystem walk (any depth) so InternData + new sources all appear.
repos = [r for r in D._iter_leaf_repos() if r.split("/")[0] in SOURCES]
print("repos", len(repos), Counter(r.split("/")[0] for r in repos), flush=True)
assert repos, "no repos discovered under cleaned_root; check symlinks"
items = []
for i, rel in enumerate(repos, 1):
    n = int(D.info(rel).get("total_episodes") or 0)
    if n <= 0:
        n = len(D.episodes(rel))
    items += [[rel, e] for e in range(n)]
    if i % 200 == 0 or i == len(repos):
        print(f"[{i}/{len(repos)}] items={len(items)} last={rel} n={n}", flush=True)
path = CACHE / "_items.json"
tmp = path.with_suffix(".tmp")
tmp.write_text(json.dumps(items))
os.replace(tmp, path)
print("wrote", path, "n=", len(items), flush=True)
print("by_src", Counter(r.split("/")[0] for r, _ in items), flush=True)
PY
    [ -f "$ITEMS" ] || { echo "items list failed: $CACHE/_logs/items.log"; exit 1; }
  else
    echo "[node $NODE_RANK] waiting for $ITEMS ..."
    until [ -f "$ITEMS" ]; do sleep 30; done
  fi
fi
echo "[node $NODE_RANK] items=$(python -c "import json;print(len(json.load(open('$ITEMS'))))")"

kill_orphans() {
  ps -eo pid,ppid,cmd | awk -v q="$Q35/bin/python" '$2==1 && index($0,q) && /multiprocessing/ {print $1}' | xargs -r kill -9
}

run_shard() {
  local gpu=$1 rank=$2 log=$CACHE/_logs/node${NODE_RANK}_rank$2.log
  if [ -f "$log" ] && [ -s "$log" ]; then
    mv -f "$log" "${log}.prev.$(date +%Y%m%d_%H%M%S)" 2>/dev/null || true
  fi
  : > "$log"
  for attempt in $(seq 1 "$MAX_RETRY"); do
    echo "=== attempt $attempt $(date) world=$WORLD gpu=$gpu ===" | tee -a "$log"
    CUDA_VISIBLE_DEVICES=$gpu python -m lerobot.policies.BPV5QFormer.build_cache --cache_dir "$CACHE" \
      --items_file "$ITEMS" --vlm_path "$VLM" --rank "$rank" --world "$WORLD" --num_workers "$WORKERS" \
      --episodes_per_batch "$EPB" --gpu_decode_pixels "$GPU_DECODE_PIXELS" \
      >> "$log" 2>&1 < /dev/null && return 0
    echo "[rank $rank] exit $?, retry in 60s" | tee -a "$log"
    sleep 60
    kill_orphans
  done
  return 1
}

summarize_progress() {
  local out=$CACHE/_logs/PROGRESS.txt
  python - <<'PY' "$CACHE" "$WORLD" "$NODE_RANK" "$out"
import re, sys, time
from pathlib import Path
cache, world, node, out = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
logs = cache / "_logs"
rows = []
for p in sorted(logs.glob("progress_rank*.txt")):
    t = p.read_text().strip().splitlines()[-1] if p.stat().st_size else ""
    m = re.search(
        r"rank=(\d+) status=(\w+) pending=(\d+) done=(\d+) failed=(\d+) rate=([0-9.]+) eta_h=([0-9.\-]+)",
        t,
    )
    if m:
        rows.append(tuple(m.groups()))
if not rows:
    for p in sorted(logs.glob(f"node{node}_rank*.log")):
        rm = re.search(r"node\d+_rank(\d+)\.log", p.name)
        if not rm:
            continue
        text = p.read_text(errors="ignore")
        pend = re.findall(r"pending episodes: (\d+)", text)
        done = re.findall(r"(\d+) / (\d+) done, (\d+) failed, ([0-9.]+) ep/s, eta ([0-9.]+) h", text)
        if done:
            d, tot, fail, rate, eta = done[-1]
            rows.append((rm.group(1), "run", tot, d, fail, rate, eta))
        elif pend:
            st = "idle" if pend[-1] == "0" else "start"
            rows.append((rm.group(1), st, pend[-1], "0", "0", "0", "-1"))
rows.sort(key=lambda r: int(r[0]))
active = sum(1 for r in rows if r[1] in {"run", "start"} and int(r[2]) > 0)
tot_pend = sum(int(r[2]) for r in rows)
tot_done = sum(int(r[3]) for r in rows)
tot_fail = sum(int(r[4]) for r in rows)
rates = [float(r[5]) for r in rows if float(r[5]) > 0]
rate = sum(rates)
left = max(0, tot_pend - tot_done - tot_fail)
eta = left / rate / 3600 if rate > 0 else -1
lines = [
    f"[{time.strftime('%F %T')}] node={node} world={world} active_ranks={active}/{len(rows)} "
    f"shard_pending={tot_pend} built={tot_done} failed={tot_fail} rate≈{rate:.2f} ep/s eta≈{eta:.2f} h",
]
for r in rows:
    lines.append(
        f"  rank{r[0]:>2}: {r[1]:5} {r[3]}/{r[2]} fail={r[4]} {float(r[5]):.2f} ep/s eta={r[6]} h"
    )
out.write_text("\n".join(lines) + "\n")
print(lines[0], flush=True)
PY
}

trap 'pkill -P $$; kill_orphans; exit 130' INT TERM
pids=()
for g in $(seq 0 $((NGPU - 1))); do
  for p in $(seq 0 $((PROCS - 1))); do
    rank=$((NODE_RANK * PER_NODE + g * PROCS + p))
    run_shard "$g" "$rank" & pids+=($!)
  done
done

(
  while true; do
    alive=0
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive=1 && break; done
    summarize_progress || true
    [ "$alive" -eq 1 ] || break
    sleep 30
  done
  summarize_progress || true
) &
prog_pid=$!

fails=0
for pid in "${pids[@]}"; do wait "$pid" || fails=$((fails + 1)); done
wait "$prog_pid" 2>/dev/null || true
kill_orphans
summarize_progress || true
echo "[node $NODE_RANK] done, failed shards=$fails, cache files=$(find "$CACHE" -name 'ep*.pt' | wc -l)"
echo "[node $NODE_RANK] progress: $CACHE/_logs/PROGRESS.txt"
[ "$fails" -eq 0 ]
