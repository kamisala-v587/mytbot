# Server

```bash
cd /vla/workspace/my_tbot
conda activate bptbot
```

## LeRobot Policy
### CKPTS
CKPT=/vla/workspace/models/tbot_ckpts/sftByOfficial/clean-e9/pretrained_model
CKPT=/vla/workspace/my_tbot/outputs/TBot_SA1/SFT-Robotwin/2026-08-25/15-57-52_tbot_robotwin_clean_e9/checkpoints/080000/pretrained_model
CKPT=/vla/workspace/models/tbot_ckpts/sftByOfficial/rand-20w/200000/pretrained_model
CKPT=/vla/workspace/models/tbot_ckpts/MytbotBase/v2
CKPT=/vla/workspace/my_tbot/outputs/BPVA/SFT-Robotwin/2026-08-28/19-58-02_bpva_train_clean_e9/checkpoints/110000/pretrained_model

CKPT=/vla/workspace/my_tbot/outputs/TBot_SA1/SFT-Robotwin/2026-09-01/17-23-40_tbot_robotwin_clean_e9_v1/checkpoints/080000/pretrained_model

CKPT=/vla/workspace/my_tbot/outputs/TBot_SA1/SFT-Robotwin/2026-09-01/15-15-46_tbot_robotwin_rand_e4_v2/checkpoints/095000/pretrained_model

### CKPTS
**tbot官方权重 + clean数据集**
CKPT=/vla/workspace/my_tbot/outputs/TBot/tbot_base_clean_e9/tbot_base_clean_e9/checkpoints/080000/pretrained_model 
**mytbot v1权重 + clean数据集**
CKPT=/vla/workspace/my_tbot/outputs/TBot_SA1/SFT-Robotwin/2026-09-01/tbot_pretrain_v1_clean_e9/checkpoints/080000/pretrained_model
**mytbot v2权重 + clean数据集**
CKPT=/vla/workspace/my_tbot/outputs/TBot_SA1/SFT-Robotwin/tbot_pretrain_v2_clean_e9/15-57-52_tbot_robotwin_clean_e9/checkpoints/080000/pretrained_model


**tbot官方权重 + robotwina全量数据集**
CKPT=/vla/workspace/models/tbot_ckpts/sftByOfficial/rand-20w/200000/pretrained_model

**tbot v1权重 + robotwina全量数据集**
···
bash /Users/luyi/Documents/Develop/workspace/codes/download_dir_resume.sh \
  b200-luyi \
  /home/jovyan/workspace/models/Tbots/v1/robotwin-sft \
  /Users/luyi/Documents/Develop/workspace/models/Tbots/v1/robotwin-sft

  bash /Users/luyi/Documents/Develop/workspace/codes/upload_dir_resume.sh \
  /Users/luyi/Documents/Develop/workspace/models/Tbots/v1/robotwin-sft \
  pro6000 \
  /home/jovyan/workspace/models/Tbots/v1/robotwin-sft
···
CKPT=/home/jovyan/workspace/models/Tbots/v1/robotwin-sft/pretrained_model

**tbot v2权重 + robotwina全量数据集**
需要周日晚上跑完
CKPT=/vla/workspace/models/tbot_ckpts/sftByMytbot/v2/robotwin-sft-20w/pretrained_model


**mytbot v2权重**
CKPT=/vla/workspace/models/tbot_ckpts/MytbotBase/v2 
### 通用启动
```bash
python server/serve_lerobot_policy_batch.py \
  --ckpt_path $CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

python server/bpva_serve_batch.py \
  --ckpt_path $CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

python server/bpva_serve_history_debug_batch.py \
  --ckpt_path $CKPT \
  --bp_mapping_path /vla/workspace/my_tbot/server/bpva_task_bps.yml \
  --action_mode delta \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
# /vla/workspace/my_tbot/server/bpva_task_bps_rand.yml 
# /vla/workspace/my_tbot/server/bpva_task_bps.yml
```

### 压测
python server/gpu_cuda_stress_test.py \
  --device 0,1,2,3 \
  --duration_sec 60000 \
  --memory_fraction 0.9 