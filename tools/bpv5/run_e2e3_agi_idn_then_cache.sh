#!/bin/bash
# 1) 50-traj 速度冒烟（4 卡）→ 估 step/s
# 2) 串行正式 E2′ → E3′（30 个满缓存 leaf，~1.1 万 ep）
# 3) deadline 06:45 硬停并恢复本机 ViT cache
#
#   CACHE_DEADLINE='2026-11-17 06:45:00' bash tools/bpv5/run_e2e3_agi_idn_then_cache.sh
set -euo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export BPV5_DATA_ROOT=/vla/workspace/cache/bpv5_cleaned_root

LOG_DIR=$REPO/outputs/BPv5/stage_a/logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
ORCH=$LOG_DIR/orch_e2e3_agi_idn_${STAMP}.log
ln -sfn "$(basename "$ORCH")" "$LOG_DIR/orch_e2e3_agi_idn_latest.log"

CACHE_DEADLINE=${CACHE_DEADLINE:-$(date -d '06:45' +'%Y-%m-%d %H:%M:%S')}
deadline_epoch=$(date -d "$CACHE_DEADLINE" +%s)
now_epoch=$(date +%s)
if [ "$now_epoch" -ge "$deadline_epoch" ]; then
  CACHE_DEADLINE=$(date -d 'tomorrow 06:45' +'%Y-%m-%d %H:%M:%S')
  deadline_epoch=$(date -d "$CACHE_DEADLINE" +%s)
fi

resume_cache() {
  echo "[$(date '+%F %T')] resume local vit cache" | tee -a "$ORCH"
  nohup env NNODES=2 NODE_RANK=0 PROCS_PER_GPU=2 bash tools/bpv5/build_cache_local.sh \
    > /vla/workspace/cache/bpv5_cleaned_p5/_logs/local_launcher_resume_${STAMP}.log 2>&1 &
  echo "[$(date '+%F %T')] cache pid=$!" | tee -a "$ORCH"
}

wait_pid_deadline() {
  local pid=$1 name=$2 port=$3 config=$4
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$(date +%s)" -ge "$deadline_epoch" ]; then
      echo "[$(date '+%F %T')] deadline; stop $name" | tee -a "$ORCH"
      kill "$pid" 2>/dev/null || true
      pkill -f "main_process_port $port" 2>/dev/null || true
      pkill -f "lerobot.scripts.lerobot_train --config_path=$config" 2>/dev/null || true
      sleep 5
      break
    fi
    sleep 20
  done
}

echo "[$(date '+%F %T')] orch SERIAL; formal=30 full-cache leaves; deadline=$CACHE_DEADLINE" | tee -a "$ORCH"

# --- speed smoke (optional; 50-traj is data-bound and underestimates formal wall rate) ---
SKIP_SMOKE=${SKIP_SMOKE:-0}
if [ "$SKIP_SMOKE" = "1" ]; then
  echo "[$(date '+%F %T')] SKIP_SMOKE=1; use prior formal rate band" | tee -a "$ORCH"
else
  echo "[$(date '+%F %T')] speed smoke: 50 traj / 200 steps" | tee -a "$ORCH"
  SPEED_OUT=$LOG_DIR/launch_speed_${STAMP}.txt
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29700 bash tools/bpv5/train_stage_a.sh \
    configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_speed_agi_idn.jsonc | tee "$SPEED_OUT" | tee -a "$ORCH"
  SPEED_PID=$(grep -oP 'started pid=\K[0-9]+' "$SPEED_OUT" | tail -1)
  wait_pid_deadline "$SPEED_PID" speed 29700 configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_speed_agi_idn.jsonc
fi

# estimate rate: 50-traj smoke is data-bound (data_s≫updt); formal ~11k ep matches E1–E3
# Prefer compute-bound estimate: 1/(median(update_s)+0.25 data), clamp to prior formal band.
RATE=$(python3 - <<'PY'
import csv, statistics as st
from pathlib import Path
rate = 0.50  # prior E1–E3 wall ≈0.55; conservative default
cands = sorted(
    Path('/vla/workspace/my_tbot/outputs/BPv5/stage_a/speed_agi_idn').glob('*/loss.log'),
    key=lambda p: p.stat().st_mtime, reverse=True,
)
if cands:
    rows = list(csv.DictReader(cands[0].open()))
    ups = [float(r['update_s']) for r in rows if r.get('update_s')]
    dls = [float(r['dataloading_s']) for r in rows if r.get('dataloading_s')]
    if ups:
        med_u = st.median(ups)
        med_d = st.median(dls) if dls else 0.25
        # Tiny smoke sets inflate data_s; formal historically ~0.2s. Cap assumed data.
        assumed_d = min(med_d, 0.30)
        rate = 1.0 / max(0.5, med_u + assumed_d)
        print(
            f'rows={len(rows)} med_updt={med_u:.3f} med_data={med_d:.3f} '
            f'assumed_d={assumed_d:.3f} raw_rate={rate:.4f}',
            file=__import__('sys').stderr,
        )
# Clamp to observed formal band (batch12 × 4gpu QFormer)
rate = max(0.40, min(0.60, rate))
print(f'{rate:.4f}')
PY
)
echo "[$(date '+%F %T')] measured_rate=${RATE} step/s" | tee -a "$ORCH"

# budget: leave 20min for e3 swap+scan, split remaining for two runs, cap 15000
LEFT=$(( deadline_epoch - $(date +%s) - 1200 ))
if [ "$LEFT" -lt 1800 ]; then LEFT=1800; fi
# each run gets half; steps = rate * seconds * 0.85 safety
STEPS=$(python3 - <<PY
rate=float("$RATE") or 0.3
left=int("$LEFT")
per=left/2
steps=int(per*rate*0.85)
steps=max(2000, min(15000, steps))
print(steps)
PY
)
echo "[$(date '+%F %T')] formal_steps_each=$STEPS (budget_left_s=$LEFT)" | tee -a "$ORCH"
python3 - <<PY
import json
from pathlib import Path
for p in ['configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_e4_agi_idn_team.jsonc','configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_e5_agi_idn_mixed.jsonc']:
  cfg=json.loads(Path(p).read_text())
  cfg['steps']=int("$STEPS")
  cfg['policy']['scheduler_decay_steps']=int("$STEPS")
  cfg['policy']['scheduler_warmup_steps']=min(500, max(100, int("$STEPS")//20))
  Path(p).write_text(json.dumps(cfg, indent=2, ensure_ascii=False)+'\n')
  print('updated', p, 'steps', cfg['steps'])
PY

run_one() {
  local name=$1 config=$2 port=$3
  local out=$LOG_DIR/launch_${name}_${STAMP}.txt
  local left=$(( deadline_epoch - $(date +%s) ))
  if [ "$left" -lt 600 ]; then
    echo "[$(date '+%F %T')] skip $name (<10min)" | tee -a "$ORCH"; return 0
  fi
  echo "[$(date '+%F %T')] start $name left=${left}s" | tee -a "$ORCH"
  CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=$port bash tools/bpv5/train_stage_a.sh "$config" | tee "$out" | tee -a "$ORCH"
  local pid; pid=$(grep -oP 'started pid=\K[0-9]+' "$out" | tail -1)
  wait_pid_deadline "$pid" "$name" "$port" "$config"
  echo "[$(date '+%F %T')] $name done/stopped" | tee -a "$ORCH"
}

run_one e2 configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_e4_agi_idn_team.jsonc 29701
run_one e3 configs/bpv5/historyOrTestOrEx/2026-11-17/stage_a_e5_agi_idn_mixed.jsonc 29702
resume_cache
echo "[$(date '+%F %T')] orch done" | tee -a "$ORCH"
