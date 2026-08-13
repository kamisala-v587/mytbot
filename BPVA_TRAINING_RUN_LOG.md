# BPVA train2 排查与运行记录

## 目标
- 跑通 `configs/bpva_train2.jsonc` 的 smoke 测试。
- smoke 通过后，用 `launch/bpva_train_watchdog2.sh` 后台启动完整四卡训练。
- 限制为必要的小改动，不进行大规模源码修改。

## 2026-08-12 初始失败基线
- 启动配置：`configs/bpva_train2.jsonc`，GPU `0,1,2,3`，每卡 batch 8。
- 输出目录：`outputs/BPVA/SFT-rand2/2026-08-12/18-50-19_bpva_train`。
- 模型与优化器初始化完成后、首个 loss 之前失败。
- 直接错误：四个 rank 报 `CUDA error: an illegal memory access was encountered`；rank 3 最终收到 SIGABRT，accelerate 报 `ChildFailedError`。
- watchdog 检测到 exit_code=1；无完整 checkpoint。失败循环已停止，当前无训练或 watchdog 进程。
- 注意：`libnccl-net.so`、`libibverbs.so` 缺失时 NCCL 回退 Socket，这些是 INFO，不是本次直接根因。

## 排查与测试记录

### 单卡 smoke（通过）
- 命令覆盖：单卡 GPU 0、`batch_size=1`、`num_workers=0`、`steps=1`、关闭评估/checkpoint/W&B，并启用 `CUDA_LAUNCH_BLOCKING=1`。
- 日志：`logs/bpva_train2_smoke_single_2026-08-12.log`。
- 结果：退出码 0；配置、数据、模型、forward/backward 和 optimizer 单卡路径可运行。

### 四卡 smoke（失败，稳定复现）
- 命令覆盖：GPU 0-3、每卡 `batch_size=1`、`num_workers=0`、`steps=1`、关闭评估/checkpoint/W&B，启用 `CUDA_LAUNCH_BLOCKING=1` 和 rank-device 映射日志。
- 日志：`logs/bpva_train2_smoke_4gpu_2026-08-12.log`。
- 结果：退出码 1。单卡通过而四卡最小配置失败，确认是多卡/DDP 专属问题，与原配置 batch=8 或数据 worker 数无关。

### 根因确认与兼容方案
- 独立 4-GPU 256 MiB NCCL broadcast 在默认 SHM 路径稳定复现 CUDA 700，证明根因属于当前节点 NCCL/容器通信层，而不是 BPVA 源码。
- 单卡原配置 batch 8 smoke 通过。
- 两卡强制 Socket 的独立 NCCL broadcast 通过。
- 两卡完整训练 smoke（batch 1，1 step）通过。
- 两卡等效配置 smoke（每卡 batch 8、梯度累积 2、2 optimizer steps）通过，保持有效 batch `8 x 2 x 2 = 32`。
- 四卡即使关闭 P2P/SHM/cuMem，训练初始 barrier 仍出现 CUDA 700，故不采用不稳定的四卡方案。

### 最小修改
仅修改 `launch/bpva_train_watchdog2.sh`：
- 默认 GPU 从 `0,1,2,3` 改为 `0,1`，进程数从 4 改为 2。
- 增加 `gradient_accumulation_steps=2`，保持原配置有效 batch 32。
- 设置 `NCCL_CUMEM_ENABLE=0`、`NCCL_P2P_DISABLE=1`、`NCCL_SHM_DISABLE=1`、`NCCL_SOCKET_IFNAME=eth0`，使用已验证的 Socket 通信路径。
- 未修改模型、数据集或训练循环源代码；未修改 `configs/bpva_train2.jsonc`。

## 正式训练启动结果（运行中）
- 启动时间：2026-08-12 19:27:04。
- watchdog PID：`609628`；accelerate PID：`609677`。
- 配置：`configs/bpva_train2.jsonc`，未修改配置文件。
- 运行方式：GPU 0/1，2 个进程，每卡 batch 8，梯度累积 2，有效 batch 32，训练步数 17500。
- 输出目录：`outputs/BPVA/SFT-rand2/2026-08-12/19-27-25_bpva_train`。
- watchdog 日志：`logs/bpva_watchdog2_2026-08-12_19-27-04.log`。
- nohup 日志：`logs/bpva_watchdog2.nohup.log`。
- 控制台日志：`logs/bpva_train2_attempt_001_2026-08-12_19-27-04.log`。
- 训练日志：`outputs/BPVA/SFT-rand2/2026-08-12/19-27-25_bpva_train/train.log`。
- loss 日志：`outputs/BPVA/SFT-rand2/2026-08-12/19-27-25_bpva_train/loss.log`。
- 验收状态：已进入训练循环并运行超过 26 step；loss 有限且持续更新，约 0.07–0.27；未出现 CUDA 700、NCCL 失败或 watchdog 重启。
- 现场 GPU：GPU 0/1 显存约 83.7/82.7 GiB，利用率约 72%/69%。

### 查看与停止
```bash
tail -f /vla/workspace/my_tbot/outputs/BPVA/SFT-rand2/2026-08-12/19-27-25_bpva_train/loss.log
tail -f /vla/workspace/my_tbot/logs/bpva_watchdog2.nohup.log
# 优雅停止（watchdog 会清理训练进程组）
kill -TERM 609628
```

## 最终结论
- `bpva_train2.jsonc` 的训练逻辑可运行，单卡与两卡 smoke 均通过。
- 当前节点四卡 NCCL collective 存在稳定 CUDA 700 故障；独立 NCCL 测试可复现，非 BPVA 源码导致。
- 已采用经 smoke 验证的两卡 Socket 兼容方案，并用梯度累积保持原有效 batch 32。
- 正式训练当前正在后台稳定运行，由 watchdog 负责异常恢复。
