# Stage B v2 profile

唯一配置 `configs/bpv5/stage_b/v2/test.jsonc`，须由调用方创建；不接受训练覆盖参数。
要求 `policy.type=bpv5_stageb_v2,batch_size=2,gradient_accumulation_steps=1,steps=40,
policy.gradient_checkpointing=false,resume=false`。原 train() 创建真实模型/数据/Accelerator/DDP；
入口仅 monkeypatch 独立诊断方法并对 train 帧局部 settrace。不修改训练/模型源文件。
禁用保存、eval/HOM、W&B、Hub 上传，训练日志也写入 BPV5_PROFILE_DIR。

## 启动命令（实现期间未执行）

```bash
cd /vla/workspace/my_tbot
source tools/bpv4cdxb/env.local.sh
export LD_LIBRARY_PATH=/vla/workspace/shiliangchao/env/q35va/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export BPV5_DATA_ROOT=${BPV5_DATA_ROOT:-/vla/workspace/cache/bpv5_cleaned_root}
export BPV5_STATS_ROOT=${BPV5_STATS_ROOT:-/vla/workspace/my_tbot/norm_stats_bpv5_pt/stage_a_pretrain_mixed_80d}
export CUDA_VISIBLE_DEVICES=0,1,2,3
export BPV5_PROFILE_DIR="$PWD/outputs/BPv5/bv2/profile_$(date +%Y%m%d_%H%M%S)"
/vla/workspace/shiliangchao/env/q35va/bin/accelerate launch \
  --multi_gpu --num_processes=4 --main_process_port=29739 --mixed_precision=no \
  tools/bpv5/profile_stage_b_v2.py
```

每次使用新目录，已有 rank 子目录会拒绝覆盖。mixed_precision=no 与原启动器默认一致，
模型 dtype 来自 test.jsonc。保留真实训练所需环境，不使用 train_stage_b.sh。
可选 `BPV5_PROFILE_MODULES='model.demo_encoder,model.core.backbone.*,model.action_head'`
用逗号分隔 glob 限定记录 CUDA event 的模块；所有模块仍维护 saved tensor 归因栈。

## 分段与计量语义

- 四卡每卡 batch 2、accum 1、GC off：全局每步 8 条轨迹、32 segments。
- preflight 保持开启：正式 40 步前，对同一真实 batch 执行两次 LR=0 的 forward/backward/
  optimizer step，之后清空 optimizer state、恢复 LR/train_step；不算入正式步数或稳态窗口。
- 1–10 warmup；11–30 无 profiler、模块 hooks、阶段 wrappers 或新增 H2D 同步：每 rank
  同时记录同步 update_policy 墙时和完整 iteration_wall_s，后者含 next(cycle(dataloader))、
  Accelerate 隐式 H2D、update、原训练器每步 GPU 遥测、跨卡 loss 同步、日志和 pbar。
  正式 trainer 只有 rank 0 执行 nvidia-smi；不向其他 rank 注入额外遥测。
- 使用 AST 精确定位 train 的 while step < cfg.steps 首条语句（当前 1250）与循环后首条语句
  （当前 1547），仅对 train 帧使用 sys.settrace。相邻循环起点闭合整步，末步在退出 while
  后立即闭合，排除训练结束日志/分布式清理。遇到源码漂移或已有 tracer 会明确拒绝。
  不在边界添加 barrier；11–30 是连续 20 步窗口。计时包含轻量 Python instrumentation 开销。
- 31–34 同步阶段诊断：forward、backward_ddp、clip、optimizer、scheduler、zero_grad、
  可选 policy_update。记录 baseline/delta、绝对阶段峰值和累计全步峰值。
- 35–38 再加模块 inclusive CUDA event/record_function，以及前向末端 saved tensor 快照。
- 39–40 CPU+CUDA profiler，分别导出 trace 和 self device time 算子排行；保留模块范围和
  阶段诊断和 saved tensor hooks（35–40 均开启，与实际实现一致）。诊断墙时有同步/盘点开销，不能替代稳态时间。

阶段峰值计数重置前合并 whole-step high water，全步峰值仍覆盖所有阶段及阶段间隙。
phase_peak 是绝对 allocator 高水位，不是增量；禁止相加阶段或模块峰值。
module_calls 是 inclusive 流时间，父子/并行范围可重叠，禁止跨模块相加。CPU-only 模块的
event 时间也可能包含其他排队工作，不应当作 exclusive kernel 时间。

saved_tensors_hooks 返回原 tensor 给 autograd；工具仅持有 weakref，不额外保留 tensor。
前向末端按底层 storage 去重，排除参数、buffer、batch 输入 storage；共享 storage
归首个存活 save 的最深执行模块。该数据是 live saved tensor storage 快照，**非峰值分解**；
不含 workspace、DDP buckets、未保存 activation、allocator slack，wrapper 弱引用失效也可
低估。不能与参数/梯度拼成“全步峰值组成”。

## 输出 schema

根目录 `training/` 为原训练日志；`rank_00/` … `rank_03/` 各自包含：

- `metadata.json`：schema_version=3/rank/world_size/torch_version/config/resolved_config/schedule/units。
  resolved_config 为 housekeeping override 后、trainer cfg.validate 前的配置。
- `device.json`：device/name/total_memory_bytes/storage_before_first_update。
- `steps.jsonl`：一行一个完整循环完成的 iteration（不是 update 返回时立即落盘）。公共字段：step/rank/mode/update_wall_s/loss/
  loss_components/before/after/whole_step_peak。内存对象字段 allocated_bytes/reserved_bytes。
  非有限 loss 写 null；不做跨 rank loss 平均。
- 诊断行：phases[] 含 name/status/before/after/wall_s/allocated_delta_bytes/phase_peak/
  whole_step_peak_so_far；另含 storage_after_backward（zero_grad 前）和 storage_after_update。
  storage 对象 groups_bytes[module][category:device] 为真实 untyped storage 字节数，
  全局去重，parameter > gradient > optimizer_state、共享 storage 首 owner 分配；另有
  unsupported_tensor_counts/deduplicated_alias_counts。包含真实 CPU optimizer state，
  gradient=None 不编造字节。排除 buffers/DDP buckets/workspace/activation/allocator slack。
- 35–38 行：saved_activations_forward_end 含 groups_storage_bytes/total_storage_bytes/
  live_tensor_refs/saved_pack_calls/scope/exclusions/attribution。
- 35–40 行：selected_modules/module_calls[{name,inclusive_cuda_ms}]/module_timing_semantics。
- `step_39.trace.json`、`step_40.trace.json`：Chrome/Perfetto CPU+CUDA trace。
- `step_39.operators.json`、`step_40.operators.json`：operators[] 按 self_device_time_us 降序；
  行字段 name/count/self_device_time_us/self_cpu_time_us/source_metric；high_cost_top_30
  标出高开销项。仅允许旧版同义 self_cuda_time_total fallback，不用 inclusive 时间替代。
- `summary.json`：完整成功 40 步后生成，含 11–30 update_wall_s 的 mean/median/min/max/
  p95_nearest_rank、iteration_wall_s 的 mean/median/min/max、连续窗口
  steady_window_elapsed_s/start_monotonic_s/end_monotonic_s、最大单步 update 内 allocated/
  reserved 峰值、逐步 losses。whole_step_peak 历史字段仍指 update 内峰值，不冒充 E2E 峰值。
- 根目录 `e2e_summary.json`：最后完成的 rank 读取四份原子发布的完整 summary 后生成；
  rank_window_elapsed_s、max_rank_window_elapsed_s、global_updates_per_s、
  global_trajectories_per_s、global_segments_per_s。若 rank 在发布前异常退出，根摘要不会生成。
  分母为 **max_rank(该 rank 步 11 起点到步 31 起点的连续 elapsed)**；分子分别为
  20、20×8、20×32。绝不可用 sum_step(max_rank(iteration_wall_s)) 替代该窗口。
  不比较不同进程的绝对时钟，仅比较各自时长；无新增 gather/barrier。
- `failure.json`：status=oom/error、rank/last_step/error/traceback/partial_record/memory/
  whole_step_peak。异常处理无额外 collective、sync、empty_cache；读取设备失败时降级为
  failure_minimal.txt。其他 rank 可能被 launcher 杀掉，不保证每 rank 都有失败文件。

无额外 gather/barrier。同一步 rank-max 仅作逐步 straggler 诊断，不作为窗口吞吐分母。
分别展示每卡峰值；不得将各卡峰值相加表示单卡显存。原 trainer 自带每步统计/日志保留。
旧 report_stage_b_v2_profile.py 仅支持 schema 1/2 的 update-only 报告，会拒绝 schema 3；
本次 E2E 结果以根 e2e_summary.json 和逐 rank 原始记录为准，不复用旧报告吞吐口径。
静态验证不启动训练；实际数据/trace 要等用户执行后产生。

## E2E 新字段（schema 3）

- `iteration_start_monotonic_s` / `iteration_end_monotonic_s` / `iteration_wall_s`：同 rank
  相邻边界的 perf_counter 值和差。落盘在边界之后，开销归入下一步/窗口；不删除真实日志开销。
- `dataloader_next_wall_s`：包装 trainer.cycle 产出的迭代器，next() 返回整体耗时，包含
  Accelerate 的预取/隐式 send_to_device，不是纯磁盘等待。
- `h2d[]`：包装 accelerate.data_loader.send_to_device，只计递归最外层调用；
  `enqueue_wall_s` 是 CPU enqueue 墙时，**不等于 GPU DMA 时间**；31–40 额外先同步再调用、
  返回后同步，记录 `synchronized_wall_s` / `completion_wait_s` / `diagnostic_sync=true`，
  这是含 Python 递归与传输完成等待的诊断墙时，不冒充纯硬件带宽。11–30 不新增传输同步。
  Accelerate 可能在一次 next 内搬运多个 batch，故用数组，不假定每次仅一个调用。
- `worker_timing`：读取原 `_pop_worker_timing` 返回值（getitem_s/getitem_max_s/cache_s/
  frames_s/xform_s 等），`worker_timing_sample_count` 为该 batch timing tensor 的元素数。
  worker 是预取工作量，通常与主循环并行，**不可与 next/update/iteration 墙时直接相加**。
  若真实 batch 没有 _timing，则为 null，不伪造零耗时；count 也可能为 null。
- `gpu_telemetry[]`：原 `_collect_gpu_stats` 调用的 wall_s/stats（含 nvidia-smi）；非主 rank
  通常为空数组。`post_update_wall_s` 为完整 update wrapper 返回到下一循环边界，含
  GPU 遥测、pbar、loss collective 和日志等，不能只归为日志；遥测是它的子集。
- metadata 新增 batch/preflight/telemetry/h2d/worker_timing/update_boundary，记录真实口径。
- ViT 诊断保留 `function/prefix/vision`、`function/prefix/build_prefix`；额外
  `function/vision/visual_only` 精确包装 vlm_visual.forward，仍是 inclusive CUDA event，
  不能与父级 prefix/module 范围相加。

### Stage B worker timing 补充

Stage B 重写 __getitem__，当前没有保留父类 QFormer 的 `_timing`。独立工具在 Linux fork
worker 中临时包装 Stage B 方法，注入兼容原 collate/_pop_worker_timing 的字段：
cache_s = _demo_episode（P5 cache load），frames_s = current_images（视频解码、resize、processor
整体），xform_s = target + tokens（动作变换与语言 tokenize）；getitem_s 为完整取样含重试。
这些分类不是全覆盖：parquet episode 读取等未拆分工作仍包含于 getitem_s，frames_s 包含图像
processor，不可称为纯解码；不要用 cache+frames+xform 冒充 getitem 全部或主循环等待。
只在缺少原生 _timing 时注入，未来数据集原生字段优先；正式数据集源码不修改。
依赖当前 Linux 默认 fork DataLoader worker 继承 profile patch；若改为 spawn，需另行支持。
