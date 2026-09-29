###  环境启动
**通用**
cd /vla/workspace/my_tbot
conda activate bptbot

**q35**
conda activate q35va
cd /vla/workspace/my_tbot

### 通用启动
```bash
python server/serve_policy.py

python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --device cuda:0 \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000
```

### CKPT
**q35**
CKPT=/vla/workspace/my_tbot/outputs/Q35/pretrain/2026-09-22/17-33-51_q35-pretrain/checkpoints/020000/pretrained_model

CKPT=/vla/workspace/my_tbot/outputs/Q35/pretrain/2026-09-22/17-33-51_q35-pretrain/checkpoints/010000/pretrained_model

**bpvav3**
CKPT=/vla/workspace/models/BPVA/v3_q35/test/robotwint3-sft-only-stage2

CKPT=/vla/workspace/my_tbot/outputs/BPVAv3/stage1/2026-09-27/22-37-10_bpvav3_stage1_with_language/checkpoints/020000/pretrained_model
conda activate q35va
cd /vla/workspace/my_tbot
python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --device cuda:0 \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8000

CKPT=/vla/workspace/models/BPVA/v3_q35/test/robotwint3-sft-only-stage1
conda activate q35va
cd /vla/workspace/my_tbot
python server/serve_policy.py \
  --ckpt_path $CKPT \
  --policy auto \
  --device cuda:1 \
  --max_batch_size 4 \
  --batch_wait_ms 10 \
  --queue_size 64 \
  --host 0.0.0.0 \
  --port 8001

--------
q35-pt-t3
CKPT=/vla/workspace/my_tbot/outputs/BPVAv3/stage1/2026-09-28/11-12-55_bpvav3_stage1_with_language/checkpoints/020000/pretrained_model
bpv3-pro6k-t3 略
bpv3-pro6k-t3_v2
CKPT=/vla/workspace/models/BPVA/v3_q35/test/bpv3-pro6k-t3_v2
bpv3-pro6k-t3_v2.1
CKPT=/vla/workspace/my_tbot/outputs/BPVAv3/stage1_sft_stage2-frozen/2026-09-28/19-50-20_t3/checkpoints/020000/pretrained_model
bpv3-b200-2.2w-sft
CKPT=/vla/workspace/models/BPVA/v3_q35/test/bpv3-b200-2.2w-sft
bpv3-b200-1.0w-sft
CKPT=/vla/workspace/models/BPVA/v3_q35/test/bpv3-b200-1.0w-sft
bpv3-b200-1.6w-sft
CKPT=/vla/workspace/models/BPVA/v3_q35/test/bpv3-b200-1.6w-sft