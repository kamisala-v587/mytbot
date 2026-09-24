# Server

```bash
cd /vla/workspace/my_tbot
conda activate bptbot
```

## LeRobot Policy

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
CKPT=/vla/workspace/models/tbot_ckpts/MytbotBase/v2/sft-robotwin_10
CKPT=/vla/workspace/models/BPVA/v2/sft-robotwin_10+10w
CKPT=/vla/workspace/models/BPVA/v2/v2.1.x/robotwin-clean-sft/025000/pretrained_model
### 通用启动

日常改 `server/serve_policy.py` 顶部大写常量（`POLICY`、`CKPT_PATH`、`DEBUG` 等），再启动。命令行只在显式传入时覆盖常量。`POLICY=auto` 时按 checkpoint 的 `type` 选择 tbot / bpvav2 / q35。

```bash
python server/serve_policy.py

python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```

`DEBUG=True` 时：bpvav2 把示教轨迹写到 `server/debug_logs/<run>/bp/`；tbot 和 q35 把每个 client 的第一次请求（任务文本、字段摘要、图像）写到 `server/debug_logs/<run>/vla/`。

tbot / bpvav2 默认要求每个已提供相机带两帧历史 `[past, current]`。q35 只使用当前帧；如果客户端传来多帧，取最后一帧。

新增策略时，在 `server/policy_runtime/` 下增加一个模块并导出 `ADAPTER`。

### v0.1 启动

旧脚本在 `server/v0.1/`。

```bash
python server/v0.1/serve_lerobot_policy_history_batch.py \
  --ckpt_path $CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

python server/v0.1/bpva_serve_batch.py \
  --ckpt_path $CKPT \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

python server/v0.1/bpva_serve_history_debug_batch.py \
  --ckpt_path $CKPT \
  --bp_mapping_path /vla/workspace/my_tbot/server/policy_runtime/bpva_task_bps.yml \
  --action_mode delta \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
# /vla/workspace/my_tbot/server/policy_runtime/bpva_task_bps_rand.yml
# /vla/workspace/my_tbot/server/policy_runtime/bpva_task_bps.yml
```

### 压测
python server/v0.1/gpu_cuda_stress_test.py \
  --device 0,1,2,3 \
  --duration_sec 60000 \
  --memory_fraction 0.9 