# BPVA 吞吐测评工具

本目录提供只读配置、运行时插桩的 BPVA 数据与短程训练测评。工具只修改内存配置副本，不保存 checkpoint、不执行 eval、不启用 W&B，也不修改生产源码。

## 文件说明

- `config_utils.py`：`register_bpva_configs()`，在解析配置前显式 `import lerobot.policies.BPVA.configuration_bpva`，确保 draccus 的 `ChoiceRegistry` 中存在 `"bpva"` 选择项（否则 `TrainPipelineConfig.from_pretrained` 会抛出 `DecodingError: Couldn't find a choice class for 'bpva'`，因为独立脚本没有 `lerobot_train.py` 里那些顺带触发注册的导入）。`data_benchmark.py`/`train_benchmark.py` 的配置加载函数都必须先调用它。
- `data_benchmark.py` / `train_benchmark.py`：CLI 入口，见下方命令。
- `data_instrumentation.py` / `model_instrumentation.py`：运行时插桩（monkey patch / hook），采集数据与模型阶段计时事件。
- `metrics.py` / `reporting.py` / `system_monitor.py`：统计聚合、报告落盘与 GPU/内存采样。
- `tests/`：不依赖 GPU 或完整模型的轻量单测，覆盖 metrics、reporting、instrumentation 与本 README 提到的配置注册修复。

## 命令

单进程数据测评：

```bash
python -m tools.bpva_benchmark.data_benchmark --config-path configs/Pro6k/bpva_sft_robotwin.jsonc --warmup-batches 5 --measure-batches 50
```

若输入配置是 `dist_loading=true`，单进程数据测评会在内存副本中自动改成 `false`，避免 `make_dataset` 拒绝运行；原配置文件不会变化。非 dist DataLoader 经 `accelerator.prepare` 后，H2D 包含在 `next_dataloader`，不会再次显式搬运。`send_to_device` 指标只存在于 dist loading；`--skip-send-to-device` 也只对 dist loading 有效。

四卡数据测评：

```bash
accelerate launch --num_processes 4 -m tools.bpva_benchmark.data_benchmark --config-path configs/Pro6k/bpva_sft_robotwin.jsonc --num-workers 4 --bp-num-chunks 10
```

四卡短程训练：

```bash
accelerate launch --num_processes 4 -m tools.bpva_benchmark.train_benchmark --config-path configs/Pro6k/bpva_intern.jsonc --warmup-steps 10 --measure-steps 100
```

训练测评应用 policy optimizer/scheduler preset、seed、rank device、TF32/cuDNN、DDP `find_unused_parameters` 与 timeout，并按正式训练语义支持 gradient accumulation。`warmup-steps` 和 `measure-steps` 都指真实 optimizer step；CSV metadata 另含 microstep、optimizer step 和 `sync_gradients`。数据和训练 DataLoader 都在存在 `dataset_weights` 时使用 `MultiLeRobotWeightedSampler`。

默认 `--output-dir` 被视为基目录，每次追加 `YYYY-MM-DD/HH-MM-SS-microseconds`，避免覆盖历史结果。只有明确传入 `--exact-output-dir` 才直接写指定目录。

## 指标口径

`elapsed_s` 是 CPU wall time，`device_elapsed_s` 是 CUDA event 设备时间。模型内部 hook 和训练一级 `forward`、`backward`、`grad_clip`、`optimizer` 等阶段都使用 CUDA event。为保持实现简单且适合短测，每个 microstep 末会执行一次 CUDA synchronize 来解析全部 event；这会扰动流水重叠，因此本工具适合阶段归因，不代表无插桩的最高吞吐。尤其异步 backward 的 CPU wall 不应当作 GPU 执行时间，应优先看 `device_elapsed_s`。

模型阶段是嵌套口径，不能相加推导总 forward。`bp_vit` hook 注册到 `key_model_map` 中实际执行的共享 vision model，并按 module identity 去重；方法 wrapper 使用 `method.*` 命名，避免与模块阶段混淆。传入插桩的是 raw policy 的 `model`（BPVA 时即 `BPVAModel`）。

视频插桩只报告整体 `decode_video_frames` / torchvision decode 调用耗时、后端和请求 timestamps。它不能直接观测 open/seek/decode/close 子阶段、实际解码帧数或 amplification；这些指标需要未来对解码器内部做更深插桩，当前报告不会推测或虚构。

## 输出、内存与事件边界

每次输出 `summary.json`、`stages.csv`、`gpu_samples.csv`、`slow_samples.jsonl` 和 `slow_videos.jsonl`。主进程只启动一个全 GPU `nvidia-smi` monitor，避免每 rank 重复采样。内存元数据包含 start、dataset 创建后和 end 三个快照，并尽力记录 RSS、PSS、Private Dirty、`/dev/shm`、cgroup current/limit 以及 `memory.events` 的 oom/oom_kill。

worker 通过有界队列非阻塞上报。collector 不保留全量事件，而是每种 kind 只保留 bounded top-k；`seen`、`retained` 和 `dropped` 计数写入 summary。这里的 dropped 包含未进入 top-k 的事件；队列在生产者侧因满而丢失的数量无法跨进程准确观测。slow 事件也受同一界限约束，长测不会因明细无限增长而 OOM。

短程训练强制关闭保存、eval、W&B 和内建 DA3 同步计时，保留 compile model 与 gradient checkpointing。工具复用现有模型、数据和视频缓存，不主动清理缓存；冷缓存与热缓存应分开实验并记录。所有 runtime patch、collector 和 monitor 都在 `finally` 中清理，分布式汇总前执行 barrier。

## 排障

- `draccus.utils.DecodingError: Couldn't find a choice class for 'bpva'`：说明配置加载函数没有先调用 `config_utils.register_bpva_configs()`。`data_benchmark.py`/`train_benchmark.py` 已经内置该调用；如果你新写了别的加载路径，记得同样先调用它，再解析 `TrainPipelineConfig.from_pretrained`。
- 数据测评单进程报 `dist_loading` 相关错误：确认没有手动覆盖 `disable_dist_loading_for_single_process` 的行为；单进程会自动把内存副本的 `dataset.dist_loading` 改成 `False`。
- CPU/cgroup OOM（常见于 `num_workers` 较大、`prefetch_factor` 较高或 PyAV 顺序解码放大读帧量时）：先用较小的 `--num-workers`、`--sample-rate` 跑一次数据测评，观察 `summary.json` 里的 `memory_*` 快照与 `memory.events` 的 oom/oom_kill 计数，再逐步恢复到生产配置定位具体拐点。
- PyAV 解码变慢/`slow_videos.jsonl` 中出现耗时异常样本：检查对应视频的 GOP 长度与请求 timestamps 是否触发了大范围顺序解码（seek 后到目标帧之间需要解码的帧数越多，单次调用越慢）；工具目前只报告整体调用耗时，无法直接看到 open/seek/decode 子阶段。
