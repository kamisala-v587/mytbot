### B200环境启动
cd /home/jovyan/workspace/mytbot
conda activate bptbot

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 

### pro6k 环境启动
cd /vla/workspace/my_tbot
conda activate bptbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1


### Tbot 训练脚本
accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/workspace/mytbot/configs/B200/pretrain_config.jsonc



### BPVA 训练脚本
accelerate launch --num_processes=8   -m lerobot.scripts.lerobot_train   --config_path=/home/jovyan/workspace/mytbot/configs/B200/bpva_config.jsonc

accelerate launch --num_processes=4   -m lerobot.scripts.lerobot_train   --config_path=/vla/workspace/my_tbot/configs/Pro6k/bpva_sft_robotwin.jsonc