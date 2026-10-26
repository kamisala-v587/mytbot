# bpv4cdx AE-3 paths on this 8 x B200 machine (no internet, no RoboTwin simulator), from env.b200.example.sh.
# `source tools/bpv4cdx/env.b200.sh`. All AE-3 steps: tools/bpv4cdx/ae3.sh.

export CONDA_PREFIX=/home/jovyan/conda-envs/q35va
export PATH=$CONDA_PREFIX/bin:$PATH
# torchcodec needs NPP (normally added by `conda activate q35va`)
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/npp/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# system libstdc++ lacks CXXABI_1.3.15 needed by torchcodec's FFmpeg 7 when pandas/pyarrow load first
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6

export BPV4CDX_REPO=/home/jovyan/workspace/mytbot
# offline gcc 12.4 (conda-forge, /home/jovyan/conda-envs/toolchain) for Triton/fla kernel builds; verified 2026-10-22
export CC=/home/jovyan/conda-envs/toolchain/bin/x86_64-conda-linux-gnu-gcc
export CXX=/home/jovyan/conda-envs/toolchain/bin/x86_64-conda-linux-gnu-g++
# fla on: Qwen3.5 forward + backward (64 x 231 tokens, one B200) 0.24 s with fla vs 1.07 s with the torch fallback.
# to fall back, append $BPV4CDX_REPO/tools/bpv4cdx/nofla to PYTHONPATH
export PYTHONPATH=$BPV4CDX_REPO/src

# pretrain-data.txt + all of agibotworld.txt (egodex is dropped automatically, repos missing locally are skipped)
export BPV4CDX_REPO_FILE=$BPV4CDX_REPO/configs/ds_ids/B200/ae3_repos.txt
export BPV4CDX_DATA_ROOTS=/share
export BPV4CDX_MANIFEST=$BPV4CDX_REPO/configs/ds_ids/B200/ae3_split.json
export BPV4CDX_QWEN_PATH=/home/jovyan/workspace/models/q35/base_800m
export BPV4CDX_STATS_ROOT=$BPV4CDX_REPO/norm_stats
# persistent xpfs (the local root disk is reset on restart, /share is read-only)
export BPV4CDX_CACHE_ROOT=/home/jovyan/workspace/cache
export BPV4CDX_ARRAY_STORE=/home/jovyan/workspace/cache/ae3_arrays
export BPV4CDX_OUT_ROOT=$BPV4CDX_REPO/outputs
export BPV4CDX_NUM_GPUS=8
export BPV4CDX_CACHE_PROCS_PER_GPU=4
export BPV4CDX_CACHE_WORKERS=12
export BPV4CDX_CACHE_EPISODES_PER_BATCH=4
# sampler chosen after the G0 audit (2026-10-22: S-U, misses are mostly < 0.5 s)
export BPV4CDX_AE3_SAMPLER=su
# ae3_main learning-rate floor (5e-5 for the 1e-4 groups); unset = the ae_main.jsonc value
export BPV4CDX_AE3_DECAY_LR=2.5e-5
export BPV4CDX_KEEP_TRAINING_STATES=2
# overlay / xpfs + filelock 3.29: per-rank parquet cache, no fchmod (src/lerobot/datasets/utils.py)
export LEROBOT_FILELOCK_WORKAROUND=1

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=16
