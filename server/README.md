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
    
```

### 压测
python server/gpu_cuda_stress_test.py \
  --device 0,1,2,3 \
  --duration_sec 60000 \
  --memory_fraction 0.9 