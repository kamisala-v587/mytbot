# 批量测评 ex_plus

主入口是 `run.sh`，单个 YAML 定义完整实验：`models × checkpoints × scenes × variants`。
控制器逐个启动模型服务，等待 `/healthz`，运行 RoboTwin 任务并发调度器，再关闭本次服务。
同一时刻只有一个实验使用模型服务；一个实验内部按 GPU 池并行运行任务。

## 运行

```bash
conda activate q35va
cd /vla/workspace/my_tbot
# 先检查匹配到的 checkpoint、任务和实验矩阵，不启动测评
bash launch/ex_plus/run.sh launch/ex_plus/experiment.yml --dry-run
# 正式执行；无需另外启动服务或客户端
bash launch/ex_plus/run.sh launch/ex_plus/experiment.yml
# 中断后，执行同一命令续评
# 仅重新生成 CSV
bash launch/ex_plus/run.sh launch/ex_plus/experiment.yml --summarize-only
```

控制器依赖 Python 3.10+ 和 PyYAML。通过 `conda run -n` 分别启动 q35va 和 myrobotwin，
不需要在两个终端手动激活环境。也可用 `server.python` / `client.python` 指定环境解释器绝对路径，
这时应自行保证所需环境变量已配置。客户端自动设置 SAPIEN_DIR（未显式设置时）。
启动前应停掉原来占用 8000 的服务，或修改 `server.args.port`，客户端端口会自动同步。
运行日志分别写到实验目录的 `server.log`、`client.log`。

## 常用修改

1. `name`：整个实验组的名字。更改任务、测评次数、场景参数或 BP 配置后，请换一个新名字。
2. `models`：支持多个训练目录，每个模型使用不同 name。
   `checkpoint_glob` 自动发现 checkpoint，按数字排序；可用 `steps: [10000, 20000]` 筛选。
   也可用 `checkpoints: [/absolute/path/to/pretrained_model, ...]` 显式选择。
   两种写法同时使用时取并集。每个路径必须包含 `config.json`。
3. `client.tasks`：预设名 `TASK_BP` / `FULL_TASKS` / `TASK_BP_FULL` / `TASK_BP_SMOKE` /
   `SMOKE_TASKS_CLEAN` / `SMOKE_TASKS_RAND`；或任务列表；或 `{adjust_bottle: 16, place_fan: 24}`。
   列表自动从 fast.py 的 FULL_TASKS 查动作执行步数；自定义任务请用字典。
4. `client.gpu_ids`、`tasks_per_gpu`、`max_retries`：仿真 GPU、每卡并发和任务最大尝试次数。
5. `scenes`：用 `overrides.task_config` 指定场景，
   `overrides.task_config_overrides.episode_num` 指定每个任务评测次数。
6. `variants`：保留需要的消融条件即可；仅看训练曲线时只保留 normal。

配置示例：仅测两个 checkpoint 的正常表现：

```yaml
models:
  - name: model_a
    checkpoint_glob: /absolute/training/checkpoints/*/pretrained_model
    steps: [10000, 20000]
variants:
  - name: normal
    server_args: {input_masks: ""}
```

测一个或多个不同模型的 BP 消融时，给 models 添加多个条目，并保留三个 variants。
模型、场景、消融均可指定 `server_args`，优先级为：
`server.args < model.server_args < scene.server_args < variant.server_args`。
客户端参数按 `base_config < client.overrides < scene.overrides` 递归合并。
`eval_name`、`ws_host`、`ws_port` 由控制器生成。

## BP 消融语义与可重复性

- normal：`input_masks: ""`，使用配置中的正常任务 BP。
- mask_bp：`input_masks: behavior_prompt`，调用现有服务的整段 BP 屏蔽，
  内容置零、chunk/image mask 失效、action_is_pad=True，保留视觉网格布局。
  服务仍会先读取 BP，再屏蔽，因此仍需要有效 BP 数据集；不是跳过 BP 编码的加速模式。
- wrong_bp：只切换示教映射，保留原任务名和语言指令。
  默认使用现有 `bpva_task_bps_clean_wrong.yml`（任务表错位 25 项），也可换自己的映射。
  若要使用随机打乱的错误 BP，请预先固定并保存映射，避免每次 checkpoint 都随机变化。

默认两个场景都使用 clean BP，仅改变环境。若要测 randomized 环境 + randomized BP，
在相应 scene 的 server_args 覆盖 bp_mapping_path；wrong_bp 的 variant 参数优先级更高。
所有 checkpoint 共享固定 debug_dir 下的现有 BP 缓存，正确/错误示教按现有缓存规则区分。
缓存首次未命中仍按原服务逻辑随机选示教，之后复用；实验期间不要清空缓存。
`seed_reuse_file` 沿用已有种子表：只有表中配置的任务使用指定种子序列，其余沿用客户端原逻辑。
这有助于对比，但不保证 GPU 数值运算或仿真完全确定。

## 结果与续跑

原始结果仍写入：

```text
/vla/workspace/my_robotwin/eval_result/
  {name}__{model}__{step}__{scene}__{variant}/
    _summary.txt
    {task}/_result.txt
    {task}/episode*.mp4
```

实验配置、命令日志和总汇总写入 `output_root/{name}/`：

- `summary.csv`：每个实验一行，包含 model、step、scene、variant、已完成任务数、
  complete、macro_success_rate（任务成功率平均）、micro_success_rate（总成功次数/总次数）。
- `tasks.csv`：每个实验的每个任务一行，包含成功次数、总次数、成功率。
- 每个实验的 `manifest.json`、`eval.yml`、`job.yml`、`server.log`、`client.log`。
- `error.txt`：失败原因；成功续跑后删除。

比较曲线时只使用 `complete=True` 的实验；未完成实验的均值只覆盖已完成任务。
成功率来自计数重新计算，避免原文本四位小数舍入误差。
测评完成与机器人成功是两个概念：成功率为 0 的已完成任务同样会被跳过。

重复执行会先核对配置指纹（含任务、场景文件、种子表、BP 映射和 checkpoint config），
已完成任务跳过，in_progress 任务交给原客户端续测。指纹不一致会拒绝复用结果。
指纹不包含模型权重、所有代码文件和 BP 缓存内容；不要原地替换同一路径的模型权重。
可以在原训练目录新增 checkpoint，重复运行时会只补充新实验。
同一实验组使用文件锁防止并发运行；不同实验组仍应避免使用相同服务端口/GPU。

Ctrl+C 会停止本次客户端和服务，客户端负责清理其任务子进程。
`continue_on_error: true` 时单个实验失败仍继续后续实验，最终退出码为 1；
设为 false 则在首次失败后停止。中断退出码为 130。

## 验证

```bash
python launch/ex_plus/test_runner.py
```

测试使用临时目录及轻量 HTTP 服务，不加载模型、不运行 GPU 仿真。
