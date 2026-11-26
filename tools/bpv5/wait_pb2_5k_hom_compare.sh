#!/bin/bash
# Wait for PB-2 (b1_lora_l0) to hit TARGET_STEP and write HOM, then compare vs PB-1 @ same step.
#   nohup bash tools/bpv5/wait_pb2_5k_hom_compare.sh &
set -euo pipefail
REPO=/vla/workspace/my_tbot
cd "$REPO"
Q35=/vla/workspace/shiliangchao/env/q35va
TARGET_STEP=${TARGET_STEP:-5000}
POLL_SEC=${POLL_SEC:-30}
PB1=${PB1:-$REPO/outputs/BPv5/stage_b/b1/2026-11-19_02-10}
PB2=${PB2:-$REPO/outputs/BPv5/stage_b/b1_lora_l0/2026-11-19_13-40}
LOG_DIR=$REPO/outputs/BPv5/stage_b/logs
REPORTS=$REPO/outputs/BPv5/stage_b/reports
mkdir -p "$LOG_DIR" "$REPORTS"
WAIT_LOG=$LOG_DIR/wait_pb2_5k_hom_$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$WAIT_LOG") 2>&1
ln -sfn "$(basename "$WAIT_LOG")" "$LOG_DIR/wait_pb2_5k_hom_latest.log"

log() { echo "[$(date '+%F %T')] $*"; }

latest_step() {
  local f=$1
  [ -f "$f" ] || { echo 0; return; }
  awk -F, 'NR>1 {s=$4} END {if (s=="") print 0; else print int(s)}' "$f"
}

hom_has_step() {
  local f=$1 step=$2
  [ -f "$f" ] || return 1
  grep -Eq "\"step\":[[:space:]]*${step}([,}[:space:]])" "$f"
}

LOSS=$PB2/loss.log
HOM=$PB2/heldout_monitor.jsonl
log "wait PB-2 step>=$TARGET_STEP then HOM, compare vs PB-1"
log "  pb1=$PB1"
log "  pb2=$PB2"

while true; do
  step=$(latest_step "$LOSS")
  log "pb2 step=$step / $TARGET_STEP"
  if [ "$step" -ge "$TARGET_STEP" ]; then
    break
  fi
  if ! pgrep -f 'configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc' >/dev/null; then
    log "ERROR: PB-2 process gone at step=$step"
    exit 2
  fi
  sleep "$POLL_SEC"
done

log "reached $TARGET_STEP; waiting for HOM line (train save_freq HOM)"
for i in $(seq 1 90); do
  if hom_has_step "$HOM" "$TARGET_STEP"; then
    log "HOM @$TARGET_STEP ready"
    break
  fi
  # checkpoint appearing is a good sign HOM is about to/already run
  if [ -d "$PB2/checkpoints/$(printf '%06d' "$TARGET_STEP")" ] || [ -d "$PB2/checkpoints/$(printf '%07d' "$TARGET_STEP")" ]; then
    log "ckpt dir present; still waiting HOM try=$i"
  else
    log "waiting HOM try=$i ..."
  fi
  if ! pgrep -f 'configs/bpv5/stage_b/probes/stage_b1_lora_l0.jsonc' >/dev/null; then
    # process may still finish HOM after last step — keep waiting a bit
    if [ "$i" -gt 20 ]; then
      log "WARN: train process gone; continue waiting HOM"
    fi
  fi
  sleep 20
done

if ! hom_has_step "$HOM" "$TARGET_STEP"; then
  log "ERROR: no HOM @$TARGET_STEP in $HOM"
  exit 3
fi

"$Q35/bin/python" - <<PY
import json
from pathlib import Path

pb1 = Path("$PB1") / "heldout_monitor.jsonl"
pb2 = Path("$HOM")
step = $TARGET_STEP
reports = Path("$REPORTS")

def load_at(path, step):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    hit = [r for r in rows if int(r["step"]) == step]
    if not hit:
        raise SystemExit(f"no step={step} in {path}")
    return hit[-1]

def loss_at(run, step):
    import csv
    p = run / "loss.log"
    rows = list(csv.DictReader(p.open()))
    near = min(rows, key=lambda r: abs(int(float(r["step"])) - step))
    return {"step": int(float(near["step"])), "loss": float(near["loss"]), "grad_norm": float(near["grad_norm"])}

a, b = load_at(pb1, step), load_at(pb2, step)
la = loss_at(Path("$PB1"), step)
lb = loss_at(Path("$PB2"), step)

keys = ["mse_matched", "mse_blank", "mse_alternate", "mse_wrong"]
ratio_keys = ["blank_over_match", "alt_over_match", "wrong_over_match"]

payload = {
    "step": step,
    "pb1": {"run": str(Path("$PB1")), "hom": a, "loss": la},
    "pb2": {"run": str(Path("$PB2")), "hom": b, "loss": lb},
}
(reports / "pb1_vs_pb2_hom_5k.json").write_text(json.dumps(payload, indent=2))

lines = []
lines.append(f"# PB-1 vs PB-2 HOM @ {step}\n")
lines.append("| | PB-1 冻全 LoRA | PB-2 LoRA L0 | Δ (PB2−PB1) |")
lines.append("|---|---:|---:|---:|")
lines.append(f"| train loss | {la['loss']:.4f} | {lb['loss']:.4f} | {lb['loss']-la['loss']:+.4f} |")
for k in keys:
    lines.append(f"| {k} | {a[k]:.4f} | {b[k]:.4f} | {b[k]-a[k]:+.4f} |")
for k in ratio_keys:
    ra, rb = a[k]["ratio"], b[k]["ratio"]
    lines.append(f"| {k} | {ra:.4f} | {rb:.4f} | {rb-ra:+.4f} |")

# verdict
bm = b["blank_over_match"]["ratio"]
am = b["alt_over_match"]["ratio"]
wm = b["wrong_over_match"]["ratio"]
bm1 = a["blank_over_match"]["ratio"]
lines.append("\n## Reading\n")
if bm > 1.05 or am > 1.05 or wm > 1.05:
    lines.append("- **PB-2 出现示教增益迹象**（至少一项比值明显 >1）。放开 L0 可能有效，继续看到 10k。")
elif bm - bm1 > 0.03 or am - a["alt_over_match"]["ratio"] > 0.03:
    lines.append("- **相对 PB-1 略有拉开，但绝对值仍弱**；L0 有一点作用，是否够用看 10k / 是否上辅助损失。")
else:
    lines.append("- **PB-2@5k 仍接近 PB-1（比值≈1）**；单层 LoRA 在此点尚不足以恢复示教依赖，等 10k；若仍贴 1 再考虑 C1 边际损失。")
lines.append(f"\n- PB-1 advice: `{a.get('advice')}`")
lines.append(f"- PB-2 advice: `{b.get('advice')}`")
lines.append(f"\nJSON: `{(reports / 'pb1_vs_pb2_hom_5k.json')}`\n")

md = reports / "pb1_vs_pb2_hom_5k.md"
md.write_text("\n".join(lines) + "\n")
print(md.read_text())
print(f"wrote {md}")
PY

log "done"
