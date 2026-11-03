#!/bin/bash
# BP (B-4 350k start) evaluation, stage A for the chosen checkpoints: new scenes, all 38 training tasks x 10
# (b350ood_<c>) || held-out at half budget: 19 sets x 5 + 2 tasks x 10 (b350held_<c>; the 5 / 10 episodes are the first
# ones of the full 10 / 20 plan, so a later top-up only runs the rest). Then the scene-novelty gaps.
# usage: setsid nohup bash tools/bpv4cdxb/b350_stage_a.sh c50k c80k c100k > <log> 2>&1 &
set -u
REPO=/vla/workspace/my_tbot
RT=/vla/workspace/my_robotwin
cd "$REPO" || exit 1
source tools/bpv4cdxb/env.local.sh >/dev/null 2>&1
RUN=$REPO/outputs/BPv4CDX/stage_b/b4_350k_sft/2026-11-02_12-04/2026-11-02_12-05
# runs next to stage B (ports 8051+): start once the training has exited
SPECS=()
for c in "$@"; do
  s=$(( ${c//[ck]/} * 1000 ))
  SPECS+=("$c:$(printf "%s/checkpoints/%06d/pretrained_model" "$RUN" $s)")
done
ALL=$(awk -F/ '{print $(NF-1)}' $RT/bp_tasks/splits/v1/train_repos.txt | sort -u | paste -sd,)
HELD=$(python - <<'PY'
import yaml
s = yaml.safe_load(open("/vla/workspace/my_robotwin/bp_tasks/splits/v1.yaml"))
print(",".join(sorted(s["heldout_sets"]) + list(s["heldout_tasks"])))
PY
)
echo "$(date '+%F %T') stage A: ${SPECS[*]%%:*}"
SWEEP_SCENES=bank SWEEP_TASKS=$ALL SWEEP_EPISODES=10 SWEEP_TASKS_PER_GPU=4 SWEEP_PORT0=8031 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh b350ood "${SPECS[@]}" &
p1=$!
sleep 30
SWEEP_SCENES=bank SWEEP_TASKS=$HELD SWEEP_GROUPS="heldout_sets, heldout_tasks" SWEEP_EPISODES=0 \
  SWEEP_HELDOUT_SETS=5 SWEEP_HELDOUT_TASKS=10 SWEEP_TASKS_PER_GPU=3 SWEEP_PORT0=8041 \
  bash tools/bpv4cdxb/sim_sweep_bp.sh b350held "${SPECS[@]}" &
p2=$!
wait $p1 $p2
cd "$RT" || exit 1
for c in "$@"; do
  /vla/.conda/miniconda3/envs/myrobotwin/bin/python -m bp_tasks.eval_policy.demo_scene_gap eval_result_bp/b350ood_$c \
    eval_result_bp/b350held_$c --out eval_result_bp/_gap_b350_$c.jsonl 2>&1 | tail -1
done
echo "$(date '+%F %T') stage A done"
