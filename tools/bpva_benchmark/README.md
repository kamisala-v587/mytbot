# BPVA Dataloader 与训练吞吐测评记录

> 最后核验：2026-09-03。本文区分【用户实验】、【代码事实】、【待实现】与【待验证】；历史日志中的运行条件不能由当前配置文件反推。

## 范围与证据说明

本文保留 2026-09-03 的 CPU dataloader A/B 原始记录，并定义四卡真实训练 benchmark 的指标、协议和验收边界。训练侧增强已实现并通过轻量测试；本文中的历史数值仍只代表用户已有实验，不代表本次执行了四卡训练。

- 【用户实验】数值、输出目录、运行时 `backend=pyav` 及慢样本分布来自用户实验记录；当前会话未重跑训练或读取原始输出验证。
- 【代码事实】脚本能力和限制以当前源码为准，关键路径见本文及 `.rag/08-bpva-data-training-benchmark.md`。
- 【代码事实】当前配置快照不是历史实验配置：`video_backend=torchcodec`、`batch_prompt_video_decode=true`、`dist_loading=false`、`bp_num_chunks=8`、`num_workers=16`、`batch_size=16`、`gradient_accumulation_steps=1`（`configs/bpvav2_pretrain_test.jsonc:10-14,22,125-128`）。新实验必须在输出中固定并记录实际配置、请求 backend、有效 backend 与 fallback；不能把当前配置写成 `pyav`。

## 已完成的 CPU dataloader A/B（用户实验）

### 历史运行条件

- 配置入口：`configs/bpvav2_pretrain_test.jsonc`
- 实验声称的运行时视频后端：`pyav`（与当前文件中的 `torchcodec` 不一致）
- 单进程 CPU；`batch_size=16`
- 命令覆盖 `num_workers=4`
- warmup 2 batches，measure 200 batches
- `--skip-send-to-device`

基线命令：

```bash
python -m tools.bpva_benchmark.train_dataloader_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/bpvav2_pretrain_test.jsonc \
  --warmup-batches=2 \
  --measure-batches=200 \
  --num-workers=4 \
  --skip-send-to-device
```

out-of-order 组在相同命令末尾增加：

```bash
  --out-of-order
```

### 原始结果

| 模式 | 输出目录 | mean | p50 | p90 | p95 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 默认 `in_order=True` | `outputs/bpva_benchmark/train_dataloader/2026-09-03/10-56-29` | 5.70s | 0.04s | 22.58s | 24.66s | 29.45s | 40.72s |
| `--out-of-order` | `outputs/bpva_benchmark/train_dataloader/2026-09-03/11-49-09` | 5.40s | 4.53s | 10.80s | 13.27s | 15.63s | 16.85s |

| 模式 | `>10s` batch | `>20s` batch | `>30s` batch |
| --- | ---: | ---: | ---: |
| 默认 `in_order=True` | 52 | 27 | 2 |
| `--out-of-order` | 28 | 0 | 0 |

【用户实验观察】慢样本主要集中在 `agibot task_498`，其次是部分 `agibot` 与 `robochallenge`。这不是当前代码事实；复核时需以样本明细、repo/task/index 和对应运行 manifest 为证据。

## 可支持的结论及边界

- 【用户实验】在这次单进程、4-worker、CPU、200 个测量 batch 的运行中，out-of-order 将 `max` 从 40.72s 降至 16.85s，并将 `>20s` 从 27 个降至 0 个；长尾明显收窄。
- 【用户实验】均值只从 5.70s 变为 5.40s，不能据此声称视频解码本身显著加速。`in_order=False` 改变的是 DataLoader 交付顺序与队头阻塞，不是单样本解码算法。
- 【边界】两次输出对应的实际 backend、fallback、数据顺序、系统负载及代码版本尚未由本文独立核验；历史结果不能直接外推到当前 `torchcodec` 配置或四卡训练。
- 【高置信推断】DDP 同步会放大任一 rank 的拖尾影响，但本次方案是四卡，不把八卡写成本次实测或执行方案。多卡完整 step 应按相同 step 的 rank 最大值汇总；把所有 rank 的记录直接混成一组均值会低估同步等待。
- 【边界】out-of-order 会改变样本到达顺序。A/B 必须固定 seed 与采样输入，并保存实际返回序列；若顺序影响训练语义，还需单独评估模型效果，不能只比较吞吐。

## 现有脚本能力清单

### `train_dataloader_benchmark.py`

已存在并应保留。当前可测量 batch `next(iterator)`、可选 `send_to_device`、每样本 `__getitem__`，记录 repo/global/local/dataset index，输出分 rank 明细与汇总，并支持 `--out-of-order`（`tools/bpva_benchmark/train_dataloader_benchmark.py:47-70,75-98,250-288,397-419,530-540,543-594,603-677`）。

限制：样本记录中的 `pid=os.getpid()` 在主进程抽取 batch 时写入，因此它不是执行 `__getitem__` 的 worker PID（`tools/bpva_benchmark/train_dataloader_benchmark.py:397-419`）。当前也没有 worker batch 起止事件，不能准确计算“每个 worker 加载一个 batch 的耗时”。

### `data_benchmark.py` + `data_instrumentation.py`

可在 DataLoader worker 内记录 `bp_dataset_getitem`、`bp_build_prompt`、`lerobot_getitem`、整个 `decode_video_frames` 调用以及 `worker_id/pid`，并测 `next_dataloader`、可选 `send_to_device` 和系统监控（`tools/bpva_benchmark/data_instrumentation.py:73-150,249-340,357-374`；`tools/bpva_benchmark/data_benchmark.py:210-249,275-305,340-369`）。

增强后的共享 instrumentation 会记录整次 decode、`bp_prompt`/`current_obs` context、requested/effective backend 与 fallback。训练脚本通过随样本回传的 metadata 在交付后关联 step；独立 `data_benchmark.py` 仍主要提供队列 Top-K 诊断，不生成训练 step。

### `train_benchmark.py`

脚本已经存在，不是待新建。它构建真实 dataset/policy/optimizer，使用 Accelerate 准备 DDP，测 `data_wait`、条件式 H2D、forward、backward、grad clip、optimizer、scheduler、zero grad，带模型子阶段 CUDA event 和 GPU/system monitor（`tools/bpva_benchmark/train_benchmark.py:180-269,287-399,427-478`；`tools/bpva_benchmark/model_instrumentation.py:23-91,97-220`）。

当前增强包括 `--out-of-order`、runtime-only data instrumentation、BP/current decode 归因、样本/worker 与交付 step 关联，以及 microstep/optimizer-step 完整 wall records。streaming 数据集保持 `IterableDataset` 类型；其 index 是 worker 本地 yield sequence，不代表全局索引。 weighted multi-repo 数据集会将未包装的原始 `MultiLeRobotDataset` 交给 `MultiLeRobotWeightedSampler`，而 DataLoader 仍读取插桩包装器，避免 sampler 严格类型检查失败。

## 四卡训练 benchmark 目标与指标定义

四卡目标是定位“训练 step 为什么慢”，而非只生成总体均值。所有指标保留 rank、optimizer step、microstep、worker、PID、batch/sample/repo 关联，并报告 count、mean、p50/p90/p95/p99/max 和 Top-K 拖尾。

1. **整体 batch / data wait**：`next(iterator)` 的主进程 CPU wall time；明确预取使它表示训练线程阻塞时间，不等于 worker 内全部工作总和。
2. **单样本 + repo**：worker 内 `__getitem__` CPU wall time，关联 repo、task（可得时）、global/local/dataset index、rank、worker PID、batch/step。
3. **BP decode**：`BehaviorPromptLeRobotDataset._build_prompt` 发起的解码调用，记录请求/有效 backend、fallback、video path、timestamp 数、耗时及 batch decode 是否成功或回退逐项。
4. **current obs decode**：`LeRobotDataset._query_videos` 路径的解码调用，与 BP decode 使用显式上下文标签隔离。
5. **每 worker / 同 step 拖尾**：每个 worker 的 sample/batch wall time；同一 step 内输出最慢 worker、最慢样本、最慢 repo，以及 `rank max - rank median`。不能用主进程 PID冒充 worker PID。
6. **forward / backward**：同时记录 CPU 提交 wall time 与 CUDA event device time；GPU 阶段比较以 CUDA event 为主，event 解析前进行明确同步。
7. **完整 step**：增加独立 CPU wall-clock `StageRecord`，边界明确为一个 microstep 或一个 optimizer step。报告必须注明 data wait 是否能与前一步训练重叠，以及 gradient accumulation 的 microstep/`sync_gradients` 边界。
8. **四卡全局 step**：对同一 `(optimizer_step, microstep)` 取四个 rank 的 wall-time 最大值，再在 step 维度统计分位数；同时保留逐 rank 分布，禁止直接混合 rank 样本冒充全局 step 分布。
9. **资源侧指标**：同步记录 CPU 利用率、RSS/系统内存、磁盘/网络 IO（依数据位置）、GPU 利用率与显存。16 workers/rank 在四卡下最多 64 个 worker，实验前先确认 CPU、内存、文件句柄和存储吞吐承载能力。

计时口径：数据加载、Python 调度、H2D 发起和完整 step 边界使用 CPU monotonic wall clock；CUDA kernel 阶段使用 CUDA events。异步 H2D 若要声称“传输完成耗时”，必须用 event 或同步边界，不能只计 `send_to_device` 的 Python 返回时间。

## 四卡 out-of-order A/B 严格协议

### 固定项

- 4 个进程，同一四张 GPU、CPU/内存/存储环境和同一时间窗口。
- 同一代码 commit/工作树快照、配置快照、checkpoint、dataset/repo 列表、seed、batch size、每 rank workers、prefetch、persistent workers、backend/cache、warmup 和 measure。
- 当前配置基准为每 rank `batch_size=16`、`num_workers=16`、accumulation 1；四 rank 最多 64 workers。机器不能承载时，应先确定一个两组共同的新 worker 数，不能只改一组。
- 保存采样输入序列和实际返回序列；两组唯一预期差异是 `in_order`。
- 每组至少重复 3 次并交错运行（如 A/B/B/A），避免缓存热度与系统负载单向偏置。

### 当前可运行的 dataloader 四卡模板

A：默认 `in_order=True`

```bash
cd /vla/workspace/my_tbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_TORCHCODEC_CACHE_SIZE=4
accelerate launch --num_processes=4 -m tools.bpva_benchmark.train_dataloader_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/bpvav2_pretrain_test.jsonc \
  --warmup-batches=20 \
  --measure-batches=200 \
  --num-workers=16 \
  --output-dir=outputs/bpva_benchmark/dataloader_4gpu_in_order
```

B：只增加 `--out-of-order`，并使用不同输出目录：

```bash
cd /vla/workspace/my_tbot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LEROBOT_TORCHCODEC_CACHE_SIZE=4
accelerate launch --num_processes=4 -m tools.bpva_benchmark.train_dataloader_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/bpvav2_pretrain_test.jsonc \
  --warmup-batches=20 \
  --measure-batches=200 \
  --num-workers=16 \
  --output-dir=outputs/bpva_benchmark/dataloader_4gpu_out_of_order \
  --out-of-order
```

### 真实训练四卡模板

A：默认 `in_order=True`：

```bash
cd /vla/workspace/my_tbot
export LEROBOT_TORCHCODEC_CACHE_SIZE=4
accelerate launch --num_processes=4 -m tools.bpva_benchmark.train_benchmark \
  --config-path=/vla/workspace/my_tbot/configs/bpvav2_pretrain_test.jsonc \
  --warmup-steps=20 \
  --measure-steps=200 \
  --num-workers=16 \
  --output-dir=outputs/bpva_benchmark/train_4gpu_in_order
```

真实训练 B 组使用同一命令，仅更换输出目录并增加 `--out-of-order`。脚本会把有效 `dataloader_in_order` 写入 manifest/summary；若当前 PyTorch 不支持 `DataLoader(in_order=...)`，out-of-order 模式会明确报错。

## torchcodec A/B 与 cache 语义

- 当前配置请求 `torchcodec`；历史 CPU 实验声称运行时为 `pyav`。每次实验必须把解析后的配置和运行 manifest 一起保存。
- `VideoDecoderCache` 读取 `LEROBOT_TORCHCODEC_CACHE_SIZE`，默认值已经是 4（`src/lerobot/datasets/video_utils.py:190-222`）。模块级默认 cache 在进程内创建并复用（`src/lerobot/datasets/video_utils.py:275-305`）；DataLoader worker 是独立进程，因此每个 worker 各自最多缓存 4 个 decoder。`export ...=4` 是显式固定默认值，不是所有 rank/worker 合计仅 4 个。
- 请求 `pyav` 时，特定 `InvalidDataError` 可回退到 torchcodec（`src/lerobot/datasets/video_utils.py:49-89`）。报告应分别记录 requested backend、effective backend、fallback 次数/原因，不能只抄配置值。
- backend A/B 固定 `in_order` 和其他全部变量，分别运行 `pyav` 与 `torchcodec`；torchcodec cache A/B 再固定 backend，仅改变 cache size（例如 1/4/8），同时观察进程 RSS、打开文件数、命中/淘汰和吞吐。当前代码没有 cache hit/eviction 计数，相关结论属于【待实现/待验证】。

## 实施缺口与验收标准

- [x] `train_benchmark.py` 支持 `--out-of-order`，manifest/summary 记录有效 `dataloader_in_order`；默认顺序交付。
- [x] worker runtime instrumentation 已接入真实训练 DataLoader，并提供可 picklable 的 worker-init 组合器。
- [x] decode 事件用上下文区分 `bp_prompt` 与 `current_obs`，记录 requested/effective backend、backend spans 与 pyav→torchcodec fallback。
- [x] sample 元数据由 worker 随 batch 回传，`next()` 后诚实关联 report step；policy/H2D 前递归剥离 benchmark 字段。
- [x] `worker_batch_envelope` 定义为已返回 batch 内同 worker 样本的 min(start) 到 max(end)，不把样本耗时求和冒充 wall time。
- [x] 记录 `microstep_wall`、`train_compute_wall` 与 `optimizer_step_wall`，保留 accumulation、`sync_gradients` 和 CUDA 同步口径。
- [x] 汇总器按同 `(stage, report_step)` 输出 rank max、median、spread、straggler rank，同时保留所有 rank 原始 `StageRecord`。
- [ ] manifest 固定代码版本、工作树状态、配置副本、解析后参数、seed、backend/fallback、cache、硬件、数据顺序、warmup/measure 和资源监控状态。
- [ ] 用小规模 smoke test 验证四 rank step id 对齐、事件不串步、默认路径行为不变，再执行耗时 A/B。

## 后续实验记录模板

```text
实验 ID / 日期：
证据类型：【用户实验】或【实测日志】
代码 commit + dirty diff：
命令 / 输出目录：
硬件：GPU×4、CPU、内存、存储：
配置副本与解析后关键值：
world_size / batch_size(per-rank) / workers(per-rank) / accumulation：
seed / sampler 输入序列 / 实际返回序列：
in_order：
requested backend / effective backend / fallback：
torchcodec cache（每进程）：
warmup / measure / 重复次数：
全局 step rank-max：mean/p50/p90/p95/p99/max
逐 rank data_wait：
worker/sample/repo Top-K 拖尾：
BP decode / obs decode：
forward/backward：CPU wall + CUDA event
完整 step 边界及 data overlap 说明：
CPU/内存/IO/GPU 资源摘要：
异常、失败样本与日志：
可支持结论：
不可支持结论 / 待验证：
```


## 训练 benchmark 输出（实现状态）

除 `stages.csv` 与 `summary.json` 外，训练 benchmark 还生成 `sample_loads.jsonl`、`step_stragglers.json/csv` 及慢样本/慢视频 JSONL。精确 step 关联来自随样本回传的 metadata；队列 Top-K 仅用于慢事件诊断。worker 预取发生时训练 step 尚未知，因此报告不会给生产事件伪造 step。`microstep_wall` 从 `next()` 前覆盖到 CUDA event 解析完成，包含该解析引入的同步；`train_compute_wall` 从 metadata 剥离/H2D 前覆盖到同一同步点。

注意：当前 `DeviceStageTimer` 的 CUDA events 在每个 microstep 末尾统一 resolve，resolve 会执行 CUDA 同步。该同步让阶段边界可精确落地，但会扰动原生异步流水线和 CPU/GPU overlap。因此 in-order/out-of-order A/B 必须使用完全相同的 instrumentation 与 resolve 模式；这些结果主要用于瓶颈归因和相对比较，不应直接当作“零插桩正式训练”的原生吞吐。若需要原生吞吐基线，应另做关闭详细插桩的独立实验。
