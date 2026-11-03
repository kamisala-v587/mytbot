#!/bin/bash
# Generalization sweep of the B-4 post-training checkpoints, all on expert-verified new scenes (scene bank):
#   ood_<ckpt>:  the 16 quick-sweep training tasks, training candidate sets, 10 episodes per task (new scenes only)
#   held_<ckpt>: held-out candidate sets (19 sets x 10) + held-out tasks (2 tasks x 20)
# then tools/bpv4cdxb/ood_heldout_report.py.
# usage: setsid nohup bash tools/bpv4cdxb/ood_heldout_sweep.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=$REPO/outputs/BPv4CDX/stage_b
RUN2=$ST/b4_sft_full/2026-10-30_02-46/2026-10-30_02-47/checkpoints
RUN3=$ST/b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47/checkpoints
SPECS=("c38k:$RUN2/018000/pretrained_model" "c50k:$RUN2/030000/pretrained_model"
       "c80k:$RUN3/030000/pretrained_model" "c100k:$RUN3/050000/pretrained_model")
HELD_TASKS=$(python - <<'PY'
import yaml
s = yaml.safe_load(open("/vla/workspace/my_robotwin/bp_tasks/splits/v1.yaml"))
print(",".join(sorted(s["heldout_sets"]) + list(s["heldout_tasks"])))
PY
)

echo "$(date '+%F %T') start: ${SPECS[*]%%:*}"
SWEEP_SCENES=bank SWEEP_EPISODES=10 SWEEP_PORT0=8031 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh ood "${SPECS[@]}" &
p1=$!
sleep 30  # stagger the server start-up
SWEEP_SCENES=bank SWEEP_TASKS=$HELD_TASKS SWEEP_GROUPS="heldout_sets, heldout_tasks" SWEEP_EPISODES=0 \
  SWEEP_HELDOUT_SETS=10 SWEEP_HELDOUT_TASKS=20 SWEEP_PORT0=8041 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh held "${SPECS[@]}" &
p2=$!
wait $p1 $p2
echo "$(date '+%F %T') report"
python tools/bpv4cdxb/ood_heldout_report.py
echo "$(date '+%F %T') all done"
