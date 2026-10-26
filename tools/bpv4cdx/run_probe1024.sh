#!/bin/bash
# bp_probe was re-recorded on 2026-10-24: re-run the bp_probe evaluations on the new data for AE-3 (600k / 700k) and
# the AE-2 references (S-U, S-A at 20k) with caches built from the new data and labels regenerated from it
# (probe_labels.py). AE-3's bp_probe G4 / G5 (run_ae3_eval.sh, results/test/probe_*.json) already used the new data.
# usage: source tools/bpv4cdx/env.local.sh; bash tools/bpv4cdx/run_probe1024.sh
set -u
cd "$BPV4CDX_REPO" || exit 1
E=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a3_eval
R=$E/results
V2=$BPV4CDX_OUT_ROOT/BPv4CDX/stage_a/v2
mkdir -p "$E/logs" "$R"/{test1024,g7d1024,g7s1024}
M="env -u BPV4CDX_CACHE_ROOT python -m lerobot.policies.BPV4CDX"
PROBE=configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt
LABELS=$E/labels_probe_20261024.json
PR="--data probe --probe_repo_file $PROBE --labels $LABELS"
T=$(ls -d "$V2"/ae2l_textonly/*/checkpoints/020000/pretrained_model | tail -1)

# eval copies of the AE-2 models pointing at the new bp_probe caches
python - "$V2" "$E" <<'EOF'
import json, sys
from pathlib import Path
v2, e = Path(sys.argv[1]), Path(sys.argv[2])
for name, run, sampler in (("ae2probe_su", "ae2l_su", "su"), ("ae2probe_sa", "ae2l_main", "sa")):
    src = sorted(v2.glob(f"{run}/*/checkpoints/020000/pretrained_model"))[-1]
    dst = e / name / "checkpoints" / "0020000" / "pretrained_model"
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("model.safetensors", "config.json", "stats.json"):
        if not (dst / f).exists():
            (dst / f).symlink_to(src / f)
    cfg = json.loads((src / "train_config.json").read_text())
    cfg["dataset"]["cache_dir"] = f"/vla/workspace/cache/bpv4cdx_{sampler}_probe1024"
    cfg["dataset"]["repo_id_file"] = "/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_probe.txt"
    cfg["dataset"]["data_root"] = None
    (dst / "train_config.json").write_text(json.dumps(cfg, indent=2))
    print(dst)
EOF
ck() { echo "$E/$1/checkpoints/$2/pretrained_model"; }
declare -A CK=([ae3_60w]="$(ck probe 0600000)" [ae3_70w]="$(ck probe 0700000)"
               [ae2_su]="$(ck ae2probe_su 0020000)" [ae2_sa]="$(ck ae2probe_sa 0020000)")
echo "start $(date '+%m-%d %H:%M')"
(for n in ae2_su ae2_sa; do
   CUDA_VISIBLE_DEVICES=0 $M.gate_g4_g7 --ckpt "${CK[$n]}" --textonly_ckpt "$T" --out "$R/test1024" --name "$n" --split all
 done
 for n in ae3_60w ae3_70w ae2_su ae2_sa; do
   CUDA_VISIBLE_DEVICES=0 $M.probe_zdemo --ckpt "${CK[$n]}" $PR --out "$R/g7d1024" --name "${n}_probe"
 done) > "$E/logs/probe1024_a.log" 2>&1 &
(for n in ae3_60w ae3_70w ae2_su ae2_sa; do
   CUDA_VISIBLE_DEVICES=1 $M.saliency_zdemo --ckpt "${CK[$n]}" $PR --episodes 0 --figures 3 --out "$R/g7s1024" --name "${n}_probe"
 done) > "$E/logs/probe1024_b.log" 2>&1 &
wait
echo "done $(date '+%m-%d %H:%M')"
