#!/bin/bash
# Scale-up evaluation of BP 350k and q35 at 100k (identical scenes for both), after the given process (0k/35k run):
#   1. held-out sets 10 -> 25 episodes per set (held-out tasks stay at 20)
#   3. training scenes: all 38 tasks, 30 episodes per task (was 16 tasks x 10)
#   2. new scenes: 30 episodes per task (was 10), the added ones with per-step traces
#   then failure modes, scene-novelty gaps, paired statistics (_paired_bp350k_vs_q35_100k.md) and the report figures.
# Existing episodes are kept (plans of the larger budgets contain the smaller ones; run_eval resumes per uid).
# usage: setsid nohup bash tools/bpv4cdxb/scaleup_100k.sh <pid to wait for> > <log> 2>&1 &
set -u
WAIT_PID=${1:-}
REPO=/vla/workspace/my_tbot
RT=/vla/workspace/my_robotwin
RT_PY=/vla/.conda/miniconda3/envs/myrobotwin/bin/python
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
ST=$REPO/outputs/BPv4CDX/stage_b
B350="c100k:$ST/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05/checkpoints/100000/pretrained_model"
Q35="c100k:$ST/q35_bp_baseline/2026-10-31_22-24/2026-10-31_22-25/checkpoints/100000/pretrained_model"
ALL=$(awk -F/ '{print $(NF-1)}' $RT/bp_tasks/splits/v1/train_repos.txt | sort -u | paste -sd,)
HELD=$(python - <<'PY'
import yaml
s = yaml.safe_load(open("/vla/workspace/my_robotwin/bp_tasks/splits/v1.yaml"))
print(",".join(sorted(s["heldout_sets"]) + list(s["heldout_tasks"])))
PY
)
if [ -n "$WAIT_PID" ]; then
  echo "$(date '+%F %T') waiting for pid $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
sleep 60
pair() {  # <sweep suffix> <env...>: BP 350k on GPUs 0-2 (offset 0), q35 on 2-0 (offset 2), in parallel
  local s=$1; shift
  env "$@" SWEEP_GPU_OFFSET=0 SWEEP_PORT0=8031 bash tools/bpv4cdxb/sim_sweep_bp2.sh "b350$s" "$B350" &
  sleep 20
  env "$@" SWEEP_GPU_OFFSET=2 SWEEP_PORT0=8041 bash tools/bpv4cdxb/sim_sweep_bp2.sh "q35$s" "$Q35" &
  wait
}
echo "$(date '+%F %T') step 1: held-out sets -> 25 per set"
pair held SWEEP_SCENES=bank SWEEP_TASKS=$HELD "SWEEP_GROUPS=heldout_sets, heldout_tasks" SWEEP_EPISODES=0 \
  SWEEP_HELDOUT_SETS=25 SWEEP_HELDOUT_TASKS=20 SWEEP_TASKS_PER_GPU=8
echo "$(date '+%F %T') step 3: training scenes, 38 tasks x 30"
pair train SWEEP_SCENES=train SWEEP_TASKS=$ALL SWEEP_EPISODES=30 SWEEP_TASKS_PER_GPU=8
echo "$(date '+%F %T') step 2: new scenes, 38 tasks x 30 (traced)"
pair ood SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=30 SWEEP_TRACE=1 SWEEP_TASKS_PER_GPU=8
echo "$(date '+%F %T') analysis"
cd "$RT" || exit 1
for d in b350ood_c100k q35ood_c100k; do
  $RT_PY -m bp_tasks.eval_policy.failure_modes eval_result_bp/$d --out_md eval_result_bp/$d/_failure_modes_38tasks.md > /dev/null \
    && echo "failure modes: $d"
done
$RT_PY -m bp_tasks.eval_policy.demo_scene_gap eval_result_bp/b350ood_c100k eval_result_bp/b350held_c100k \
  --out eval_result_bp/_gap_b350_c100k.jsonl | tail -1
$RT_PY -m bp_tasks.eval_policy.demo_scene_gap eval_result_bp/q35ood_c100k eval_result_bp/q35held_c100k \
  --out eval_result_bp/_gap_q35_c100k.jsonl | tail -1
cd "$REPO" || exit 1
python code/eval_paired_stats.py
python code/eval_350k_figs.py > /dev/null && echo "figures regenerated"
echo "$(date '+%F %T') all done"
