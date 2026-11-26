#!/usr/bin/env bash
# PB-3 @10k: Damntwin E3 OOD only, 10 eps/task (quick look; no E2 train).
#
# Env split (same as tools/eval_rdemo_smoke.sh):
#   - clients / eval_report_sweep: conda myrobotwin (SAPIEN + curobo)
#   - policy server: started by the sweep via tools/bpv4cdxb/env.local.sh (q35va)
set -euo pipefail
ROOT_TBOT=/vla/workspace/my_tbot
ROOT_DT=/vla/workspace/damntwin
LOG=$ROOT_TBOT/outputs/BPv5/stage_b/logs/pb3_damntwin_e2_e3.log
MY_PY=/vla/.conda/miniconda3/envs/myrobotwin/bin/python
mkdir -p "$(dirname "$LOG")"

export DAMNTWIN_ASSETS=${DAMNTWIN_ASSETS:-/vla/workspace/my_robotwin/assets}
export DAMNTWIN_DATA=${DAMNTWIN_DATA:-/vla/workspace/my_robotwin/data_bp}
export DAMNTWIN_SAPIEN_OVERLAY=${DAMNTWIN_SAPIEN_OVERLAY:-/vla/workspace/sapien_physx540fix}
export BPV5_DEMO_CACHE_DIR=${BPV5_DEMO_CACHE_DIR:-/vla/workspace/cache/bpv5_cleaned_p5}

cd "$ROOT_DT"
# shellcheck disable=SC1091
source ./damntwin-env.sh

{
  echo "==== $(date -Is) PB-3 Damntwin E3-only@10 start (client=$MY_PY)"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv || true
  "$MY_PY" -c 'import curobo, sapien, transforms3d, toppra, mplib; from envs.robot.planner import CuroboPlanner; print("client sim ok", CuroboPlanner)'
  "$MY_PY" tools/eval_report_sweep.py --models pb3 --evals e3 --ckpts 10k --compare ''
  echo "==== $(date -Is) PB-3 Damntwin E3-only@10 done"
} 2>&1 | tee -a "$LOG"
