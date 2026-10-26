# bpv4cdx AE-3 paths on the 8 x B200 machine (no internet, no RoboTwin simulator). Copy to env.b200.sh, edit, then
# `source tools/bpv4cdx/env.b200.sh`. All AE-3 steps: tools/bpv4cdx/ae3.sh.

export PATH=/path/to/q35va/bin:$PATH
export PYTHONPATH=/vla/workspace/my_tbot/src${PYTHONPATH:+:$PYTHONPATH}

export BPV4CDX_REPO=/vla/workspace/my_tbot
# dataset list (egodex is dropped automatically) and the local roots its relative ids resolve under
export BPV4CDX_REPO_FILE=$BPV4CDX_REPO/configs/ds_ids/B200/pretrain-data.txt
export BPV4CDX_DATA_ROOTS=/share
# split manifest (held-out episodes): generated once (`ae3.sh manifest`), then copied to every machine unchanged
export BPV4CDX_MANIFEST=$BPV4CDX_REPO/configs/ds_ids/B200/ae3_split.json
export BPV4CDX_QWEN_PATH=/path/to/q35/base_800m
export BPV4CDX_STATS_ROOT=$BPV4CDX_REPO/norm_stats
# local NVMe: P5 cache (~0.7 TB for ~650k episodes) and the state / action arrays (~30 GB)
export BPV4CDX_CACHE_ROOT=/path/to/nvme/cache
export BPV4CDX_ARRAY_STORE=/path/to/nvme/ae3_arrays
export BPV4CDX_OUT_ROOT=$BPV4CDX_REPO/outputs
export BPV4CDX_NUM_GPUS=8
# cache build: processes per GPU (video decoding is the bottleneck, ViT is batched), decoder threads per process
export BPV4CDX_CACHE_PROCS_PER_GPU=4
export BPV4CDX_CACHE_WORKERS=12
export BPV4CDX_CACHE_EPISODES_PER_BATCH=4
# filled in from the G0 audit (`ae3.sh audit` prints the decision)
export BPV4CDX_AE3_SAMPLER=suf
# checkpoints keep the optimizer state of the newest N saves only (0 = keep all)
export BPV4CDX_KEEP_TRAINING_STATES=2

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=16
# ae3_main learning-rate floor; unset = the ae_main.jsonc value
# export BPV4CDX_AE3_DECAY_LR=2.5e-5
# overlay / xpfs + filelock 3.29: per-rank parquet cache, no fchmod (src/lerobot/datasets/utils.py)
# export LEROBOT_FILELOCK_WORKAROUND=1
