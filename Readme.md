### CONFIG
CONFIG=/vla/workspace/my_tbot/configs/Pro6k/bpv3_plus.jsonc

CONFIG=/home/jovyan/workspace/mytbot/configs/B200/bpvav3.jsonc
CONFIG=/home/jovyan/workspace/mytbot/outputs/bpv3/stage1/frozen_vit/2026-09-29_12-20/checkpoints/002000/pretrained_model/train_config.json

CONFIG=/home/jovyan/workspace/mytbot/configs/B200/bpvav3_plus.jsonc
### B200环境启动
CONFIG=/home/jovyan/workspace/mytbot/configs/B200/bpv3_fixed8_taskcon_stage15.jsonc
CONFIG=/home/jovyan/workspace/mytbot/configs/B200/bpv3_fixed8_taskcon_stage15_vit.jsonc

cd /home/jovyan/workspace/mytbot
conda activate q35va

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=64
# B200 必须：overlay/xpfs + filelock 3.29 下多卡共享 HF datasets 缓存会报 FileNotFoundError
export LEROBOT_FILELOCK_WORKAROUND=1

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train --config_path=$CONFIG

### pro6k 环境启动
cd /vla/workspace/my_tbot
conda activate q35va  # bpv3 # tbot 与 bpv2及以下的模型不用这个环境
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=16

accelerate launch --num_processes=4 -m lerobot.scripts.lerobot_train --config_path=$CONFIG



### Benckmark
accelerate launch --num_processes=8 -m tools.bpva_benchmark.train_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/Pro6k/bpv2.jsonc \
  --warmup-steps=20 \
  --measure-steps=500 \
  --num-workers=12 \
  --output-dir=outputs/bpva_benchmark

