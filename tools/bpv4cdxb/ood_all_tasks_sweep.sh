#!/bin/bash
# Top up ood_<ckpt> (new scenes, training candidate sets, 10 episodes per task) from the 16 quick-sweep tasks to all
# 38 training tasks of split v1 for the best checkpoints; existing episodes are kept. Then the report.
# usage: setsid nohup bash tools/bpv4cdxb/ood_all_tasks_sweep.sh > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=$REPO/outputs/BPv4CDX/stage_b
RUN2=$ST/b4_sft_full/2026-10-30_02-46/2026-10-30_02-47/checkpoints
RUN3=$ST/b4_sft_full2/2026-10-30_13-45/2026-10-30_13-47/checkpoints
SPECS=("c50k:$RUN2/030000/pretrained_model" "c80k:$RUN3/030000/pretrained_model" "c100k:$RUN3/050000/pretrained_model")
ALL=$(awk -F/ '{print $(NF-1)}' /vla/workspace/my_robotwin/bp_tasks/splits/v1/train_repos.txt | sort -u | paste -sd,)

echo "$(date '+%F %T') start: ${SPECS[*]%%:*}, $(echo "$ALL" | tr ',' '\n' | wc -l) tasks"
SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=4 SWEEP_PORT0=8031 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh ood "${SPECS[@]}"
echo "$(date '+%F %T') report"
python tools/bpv4cdxb/ood_heldout_report.py
echo "$(date '+%F %T') all done"
