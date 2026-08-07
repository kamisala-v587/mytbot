### 预训练Tbot
cd /home/jovyan/vla/workspace/mytbot
source /home/jovyan/.conda/envs/tbot/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export LEROBOT_PARALLEL_DATASET_LOAD=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/vla/workspace/mytbot/configs/pretrain_config.jsonc

cat /home/jovyan/vla/workspace/mytbot/.config/pretrain_config.jsonc


### 微调Tbot
cd /home/jovyan/vla/workspace/mytbot
source /home/jovyan/.conda/envs/tbot/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/home/jovyan/vla/workspace/mytbot/.config/finetune_config.jsonc

accelerate launch --num_processes=8 -m lerobot.scripts.lerobot_train --config_path=/home/jovyan/vla/workspace/mytbot/outputs/TBot_SA1/2026-06-26/06-19-44_TBot_SFT_robotwin_v0/checkpoints/154000/pretrained_model/train_config.json


### 防中断（nohup 续训，容器重建后 tmux 不可用）
```
cd /home/jovyan/vla/workspace/mytbot
mkdir -p logs
source /home/jovyan/.conda/envs/tbot/bin/activate
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=1

CKPT="/home/jovyan/vla/workspace/mytbot/outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_ /checkpoints/010000"
accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path="${CKPT}/pretrained_model/train_config.json"

nohup accelerate launch --num_processes=8 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path="${CKPT}/pretrained_model/train_config.json" \
  > logs/resume_$(date +%F_%H%M).log 2>&1 &

# [1] 297 PID
tail -f /home/jovyan/vla/workspace/mytbot/logs/resume_2026-07-03_1014.log

kill PID # 1816

```
--resume=true会覆盖，config_path设定为断点的ckpt即可
resume=true 时代码自动设pretrained_path为 .../008000/pretrained_model/



### 监控

cd /home/jovyan/vla/workspace/mytbot
bash /home/jovyan/vla/workspace/mytbot/launch/watch_training.sh \
  "outputs/TBot_SA1/pretrain_v1/2026-07-03/01-46-07_ "


## BP Tbot
cd /vla/workspace/my_tbot
conda activate mytbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --config_path=/vla/workspace/my_tbot/configs/bp_tbot_pretrain_config.jsonc

或者 
cd /vla/workspace/my_tbot
conda activate mytbot
bash tools/launch_bp_tbot_pretrain.sh

恢复
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 \
  -m lerobot.scripts.lerobot_train \
  --resume=true \
  --config_path='outputs/BP_TBot/pretrain_v1/2026-07-30/22-39-35_BP_TBot——test/checkpoints/011000/pretrained_model/train_config.json' \
  --num_workers=8


### H100 - Tbot -SFT
cd /vla/workspace/my_tbot
conda activate mytbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_PARALLEL_DATASET_LOAD=0

CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1   -m lerobot.scripts.lerobot_train   --config_path=/vla/workspace/my_tbot/configs/tbot_sft_h100.jsonc 