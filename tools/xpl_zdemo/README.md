# xpl_zdemo — F1 real Z_demo FEATURE cache producer for XR1

Produces the **real** per-episode Qwen3.5 ViT feature bank the XR1 BPv4
encoder-in-policy recipe consumes (`XPolicyLab.utils.bp.zdemo.ZdemoCache`):
3500 episodes of the converted RoboDojo root → pinned S-U embodiment sampler
(32 slots) → frames → Qwen3.5-0.8B ViT (224 px / 64 tok) → P5 pooling →
`feats [32,3,5,1024] bf16` records. Every piece of pinned logic (sampler,
record assembly, pooling, ViT, processor, frozen loading) is IMPORTED from
this my_tbot checkout @ `94859f7`; nothing is re-implemented here.

## Files

| File | Purpose |
|---|---|
| `episode_source.py` | root scan (`<chunk>/episode_*.json`, key = `<task_dir>/<episode_stem>`) + loud per-episode loading (state `[N,14]` concat in mask order; camera mp4 paths); `--check-videos` audit CLI |
| `plan_slots.py` | the pinned `Embodiment` (arx_x5 dual-arm 6+1, grippers at dims 6/13) + `plan_for()` via `sample_su_emb` (version- and range-checked); self-check CLI |
| `build_feats.py` | the producer (shardable/resumable, failures jsonl, reconcile) + `--finalize` (index.json + manifest.json from the frozen delivery) |
| `frame_probe.py` | §3.5 frame-convention probe (reader → float/255 → resize → processor → 64 tok grids) |
| `parity_frame_path.py` | PARITY A: my frame path vs an upstream cached ViT record (cosine ≥ 0.99 gate) |
| `parity_encode_ref.py` | PARITY B upstream side: frozen Stage-A encoder over MY records → `z_ref.pt` |

XR1-side throwaways (XPolicyLab, not committed): `bp/tools/parity_zdemo_xr1.py`
(parity B XR1 side), `bp/tools/zdemo_coverage_check.py` (T7.1 coverage check).

## Cache format (the reader's contract — `utils/bp/zdemo.py`)

- `<root>/<key>.pt` = dict with EXACTLY `feats` (bf16 `[32,3,5,1024]`),
  `time` (f32 `[32]`, slot ratios in [0,1]), `camera_valid` (bool `[32,3]`),
  `sampler_version` (the STRING `su-uniform_active_v2_embodiment`), `spatial` (5).
- `index.json` = `{key: "<key>.pt"}` for every written record (keys must equal
  the dataset's episode set — XR1 validates full coverage at init).
- `manifest.json` = `frozen` block from `describe(load_frozen(StageA_AE_frozen/0750000))`
  (real `ckpt_sha256 89ef6e438045…`, sampler block, `spatial_tokens_used: 5`)
  + `"synthetic": false` + `producer` provenance (tool/root/qwen_path/episodes/
  failures/date/`git_head_my_tbot`).

Data source per episode json: `num_frames`; `proprios.{left_arm_joint [N,6],
left_gripper_pos [N,1], right_arm_joint [N,6], right_gripper_pos [N,1]}`
(concatenated in that order — matches the embodiment mask);
`observations.{ego,wrist_left,wrist_right}[0].path` (absolute mp4, `start` 0).

## Commands (q35va env: `source /vla/workspace/my_tbot/tools/bpv4cdxb/env.local.sh`)

```bash
ROOT=/vla/workspace/shiliangchao/RoboDojo_runtime/XPolicyLab/policy/Xiaomi_Robotics_1/xiaomi_robotics_1/xr1/data/RoboDojo-all-arx_x5-ee/data
OUT=/vla/workspace/cache/xpl_zdemo_feats/xr1_robodojo

# preflight (all green 2026-10-29: 3500/3500 OK, 0 bad)
python episode_source.py --root $ROOT --check-videos
python plan_slots.py   --root $ROOT --sample 3500
python frame_probe.py  --episode $ROOT/chunk-0000/episode_0000000.json

# build (8 GPUs: one rank per GPU; resumable — rerun skips existing records)
for r in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$r python build_feats.py --root $ROOT --out $OUT \
      --rank $r --world 8 --episodes_per_batch 4 &
done; wait

# finalize (T4): reconciles built+failed==scanned, writes index.json + manifest.json
python build_feats.py --root $ROOT --out $OUT --finalize

# parity (A on GPUs of this machine; B needs one GPU + the mibot env for the XR1 side)
CUDA_VISIBLE_DEVICES=0 python parity_frame_path.py --count 3
CUDA_VISIBLE_DEVICES=0 python parity_encode_ref.py --cache $OUT --out $OUT/../parity --count 8
```

Failure semantics: broken episodes → `$OUT/_failures/rank<r>.jsonl`; per-rank
reconcile `built + failed == scanned`; `--finalize` aborts unless every scanned
episode has a record OR a failure line (and warns that failed keys will break
XR1 coverage init). `--limit N` / `--dry_run` for smokes.

## Verified results (2026-10-29, this 2-GPU machine)

- Preflight: scan 3500 (1000+1000+1000+500); audit `3500 OK, 0 bad` (incl. mp4
  frame-count == `num_frames`); probe `[3,480,640]` uint8 → `[3,224,224]` →
  `image_grid_thw [1,16,16]`, merge 2 → **64 tokens/image**; plan self-check:
  all 3500 plans version `su-uniform_active_v2_embodiment`, 32 slots; largest
  idle prefix `chunk-0001/episode_0001160` (lo=575) correctly excluded.
- Smoke: 8 episodes built (0.94 MB/record; ~3.3 GB projected for 3500) in the
  real cache root; record layout byte-checked against the reader's exact set.
- **PARITY A PASS** (3 episodes, `move_stapler_pad_bp/I0_L1_pad_inner` vs
  `/vla/workspace/cache/bpv4cdxb_su_p1`): my torchcodec reader vs
  `LeRobotDataset(video_backend="torchcodec")` max abs diff `0.00e+00`;
  P64 cosine `1.0001/1.0004/1.0004`, P5 cosine `0.999990/1.000015/1.000015`
  (gate ≥ 0.99). Frame convention = upstream's **CHW float [0,1]** into
  `resize_with_pad` (the session prompt's "[1,H,W,C]" phrasing is wrong; the
  binding rule "mirror build_cache.py:77-98 exactly" is what's implemented).
- **PARITY B PASS** (8 real smoke records): upstream frozen encoder
  (`load_frozen`, fp32+autocast) vs XR1 vendored encoder
  (`ZDemoEncoderConfig`, strict `load_zdemo_encoder(A/0750000)`, bf16 cast) —
  max abs diff **0.039062** < bar 0.1908 (= 0.02 × max(1, mean_abs_z×10),
  mean abs z 0.9542). fp32-vs-fp32 is exactly 0.0 on both sides.
- my_tbot git scope: only `tools/xpl_zdemo/` added; HEAD `94859f7…`; the four
  pre-existing modifications untouched.

## Remaining (post-build, run by the operator)

1. The 8-GPU build + `--finalize` above (then re-run `parity_encode_ref.py` +
   `parity_zdemo_xr1.py` verbatim for the record — same numbers expected).
2. XR1 consumer re-verification WITHOUT `ZDEMO_ALLOW_SYNTHETIC` (mibot env,
   `policy/Xiaomi_Robotics_1`): coverage check, `gpu_probe.py --zdemo` ×2
   recipes, 2+2 two-phase chain smoke — commands in `bp/HANDOFF_V4.md` F1.
