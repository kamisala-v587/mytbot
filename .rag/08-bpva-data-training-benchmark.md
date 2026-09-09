# BPVA 数据与训练性能 Benchmark 知识库

> 截至/最后核验：2026-09-03
>
> 本文是性能测评专题，不替代 `.rag/07-experiment-evidence.md` 的效果消融台账。实现判断以当前代码、配置和原始日志为准；历史运行条件与当前配置快照必须分开记录。

## 事实锚点

- 【用户实验】2026-09-03 CPU 单进程、`batch_size=16`、4 workers、warmup 2、measure 200、跳过 device copy 的 dataloader A/B 中，用户称运行时 backend 为 `pyav`。`in_order=True`：mean 5.70s、p50 0.04s、p90 22.58s、p95 24.66s、p99 29.45s、max 40.72s；out-of-order：mean 5.40s、p50 4.53s、p90 10.80s、p95 13.27s、p99 15.63s、max 16.85s。长尾计数分别为 `>10s 52/28`、`>20s 27/0`、`>30s 2/0`。输出位于 `outputs/bpva_benchmark/train_dataloader/2026-09-03/10-56-29` 与 `.../11-49-09`；本次未重跑或独立读取原始产物。
- 【用户实验】慢样本集中于 `agibot task_498`，其次为部分 `agibot`/`robochallenge`；这是用户观察，需用 repo/task/index 明细复核。
- 【代码事实】当前配置快照为 `video_backend=torchcodec`、`batch_prompt_video_decode=true`、`dist_loading=false`、`bp_num_chunks=8`：`configs/bpvav2_pretrain_test.jsonc:10-14,22`；训练为 `num_workers=16`、`batch_size=16`、`gradient_accumulation_steps=1`：`configs/bpvav2_pretrain_test.jsonc:125-128`。它与历史实验声称的 `pyav` 不一致。
- 【代码事实】BP `_build_prompt` 在 batch decode 开启时调用 `prompt_ds.get_items(..., batch_video_decode=True)`，失败后回退逐项读取：`src/lerobot/datasets/behavior_prompt_dataset.py:487-547`。当前样本随后附加 BP 并 transform：`src/lerobot/datasets/behavior_prompt_dataset.py:613-617`。
- 【代码事实】`LeRobotDataset.get_items` 按 `video_path` 聚合 timestamp 后每文件调用一次 `decode_video_frames`：`src/lerobot/datasets/lerobot_dataset.py:1110-1181`；current obs 经 `_query_videos` 调用同一个 decode 入口：`src/lerobot/datasets/lerobot_dataset.py:1069-1095`。
- 【代码事实】请求 `pyav` 遇到特定无效数据错误时可回退 torchcodec：`src/lerobot/datasets/video_utils.py:49-89`。报告必须区分请求 backend、有效 backend 与 fallback。
- 【代码事实】torchcodec `VideoDecoderCache` 的环境变量为 `LEROBOT_TORCHCODEC_CACHE_SIZE`，默认 4：`src/lerobot/datasets/video_utils.py:190-222`；模块级默认 cache 在进程内复用：`src/lerobot/datasets/video_utils.py:275-305`。因此 4 是每个 DataLoader worker 进程的默认上限，不是全局 workers 合计上限。

## 脚本矩阵

- `train_dataloader_benchmark.py`：保留原训练 dataloader 路径、batch next、可选 H2D、样本索引明细和 `--out-of-order`。其旧 `SampleLoadRecord.pid` 仍是主进程 PID，worker 精确归因应使用增强后的 `train_benchmark.py`。
- `data_benchmark.py` + `data_instrumentation.py`：runtime-only monkey patch 可记录 BP dataset、BP build、LeRobot getitem 与 decode；共享 instrumentation 现在携带 `bp_prompt`/`current_obs` context、requested/effective backend、backend spans 和 pyav→torchcodec fallback。
- `train_benchmark.py`：构建真实 dataset/policy/optimizer，使用 Accelerate，支持默认 `in_order=True` 和 `--out-of-order`；样本 metadata 随 batch 回传后关联唯一 measured microstep，并在 policy/H2D 前递归剥离。记录 data_wait、worker batch envelope、H2D、forward/backward/optimizer 子阶段、`train_compute_wall`、`microstep_wall` 和 `optimizer_step_wall`。
- `metrics.py`/`reporting.py`：默认对 data wait、compute、microstep 及 optimizer-step wall 做同 step 跨 rank max/median/spread/straggler 聚合，并保留逐 rank 原始 StageRecord。

## 指标定义

- **batch/data_wait**：【代码事实/口径】主进程 `next(iterator)` CPU wall；受 DataLoader 预取影响，表示训练线程等待，不等于 worker 总工作量。
- **sample + repo**：【已实现】worker 内返回样本 wall 及 rank、worker PID/id、repo/index；随 batch 到达后才赋予 report step，不给预取生产过程伪造训练 step。
- **BP decode / obs decode**：【已实现/边界】context 显式区分 `_build_prompt` 与 current `_query_videos`，记录 video、timestamp count、requested/effective backend 与 fallback。batch decode 失败后逐项回退可由同一样本中的事件序列诊断，但目前没有独立的“batch-to-item fallback reason”结构化事件。
- **worker/同 step 拖尾**：【已实现/口径】`worker_batch_envelope` 是已交付 batch 内同 worker 样本的 min(start) 到 max(end)，绝不使用样本耗时求和；它是观测包络，不等同于 DataLoader worker 内部完整 batch 调度 wall time。
- **forward/backward**：【代码事实/口径】CPU wall 表示主机提交/同步行为，CUDA event 表示设备区间；GPU 结论优先 event。 当前实现每个 microstep 末尾 resolve events 并同步 CUDA，能获得精确设备区间，但会改变原生异步流水线/overlap；A/B 必须固定相同插桩模式，结果用于归因而不是零插桩吞吐。
- **完整 step**：【已实现】`microstep_wall` 从 next 前到 CUDA event 解析后；`optimizer_step_wall` 从 accumulation cycle 第一个 microstep 的 next 前到最终同步 microstep 的 event 解析后。warmup cycle 完成即重置计时器，测量 optimizer step 使用独立的 0-based `StageRecord.step`。
- **四卡全局 step**：【已实现】microstep 类阶段按唯一 measured-microstep step 对齐；optimizer-step wall 按自身 0-based measured optimizer step 对齐，输出 rank max/median/spread/straggler。不可将 rank 记录直接混池作为全局 step 分布。
- **资源**：【目标】CPU、RSS/系统内存、磁盘/网络 IO、GPU 利用率/显存；当前配置四 rank 最多 64 workers，运行前验证承载。

## 已知限制与剩余验收

1. 【待验证】历史 CPU 两组的 backend/fallback、代码版本、数据顺序和系统负载尚未独立验证，不能直接外推当前 torchcodec 四卡训练。
2. 【代码事实】worker 预取发生时未来训练 step 未知；工具只在 batch 被主进程取出后关联 step。`worker_batch_envelope` 是返回样本时间范围，不是 worker 调度器的完整 batch wall。
3. 【代码事实】decode effective backend 由被调用的 backend child span 推断；能识别 pyav→torchcodec fallback，但不提供 torchcodec cache hit/eviction 计数。
4. 【代码事实】streaming dataset 由 `InstrumentedIterableDataset` 保持 `IterableDataset` 类型和 DataLoader 语义；流式样本的 `index` 是每个 worker 本地 yield sequence，不是全局 dataset index。
5. 【待验证】out-of-order 改变返回顺序，吞吐改善不代表训练效果等价。
6. 【代码事实/限制】CUDA event 每 microstep resolve 会同步并扰动原生流水线；所有 A/B 必须使用相同模式。报告适合瓶颈归因与相对比较，不代表关闭 instrumentation 的正式训练原生吞吐。
7. 【剩余验收】尚未运行真实四 rank 训练；仍需实测 rank step 对齐、存储/CPU 承载、事件体积和 A/B 可重复性。manifest 的代码 commit/dirty diff、配置副本、硬件和实际返回序列仍待补全。

## 四卡实验协议

- 四进程、同配置/checkpoint/代码/硬件/数据/seed/warmup/measure；每 rank workers 固定，A/B 只改变 `in_order`。当前 `num_workers=16` 意味着最多 64 workers，先检查 CPU、内存、IO 和文件句柄承载。
- 每组至少重复 3 次并交错运行；保存逐 rank 原始事件、实际返回顺序和资源曲线。
- 当前可用 `train_dataloader_benchmark.py --out-of-order` 做 dataloader A/B；真实训练 B 组须等待 `train_benchmark.py` 接线完成。命令模板与历史结果见 `tools/bpva_benchmark/README.md`。
- backend A/B 固定 in-order 和其他变量；torchcodec cache A/B 的 4 表示每进程 cache。pyav 组必须统计 fallback，否则会混入 torchcodec。
- 主要判据：全局 step rank-max、data_wait p95/p99/max、`>10/20/30s`、最慢 worker/sample/repo、BP/obs decode 分布、forward/backward CUDA event、完整 step wall 与 CPU/内存/IO/GPU。

## 查询关键词

`BPVA benchmark`、`dataloader`、`train_benchmark`、`train_dataloader_benchmark`、`data_instrumentation`、`in_order`、`out-of-order`、`队头阻塞`、`四卡`、`rank max`、`straggler`、`worker pid`、`task_498`、`BP decode`、`current obs decode`、`batch_prompt_video_decode`、`pyav fallback torchcodec`、`VideoDecoderCache`、`LEROBOT_TORCHCODEC_CACHE_SIZE`、`CUDA event`、`data_wait`、`gradient accumulation`。
