### CONFIG
CONFIG=/vla/workspace/my_tbot/configs/Pro6k/bpv3_plus.jsonc

CONFIG=/vla/workspace/my_tbot/configs/Pro6k/bpv3_plus.jsonc
### B200环境启动
cd /home/jovyan/workspace/mytbot
conda activate q35va

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1
export LEROBOT_TORCHCODEC_CACHE_SIZE=64

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

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

