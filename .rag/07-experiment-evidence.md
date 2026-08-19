# BPVA 实验证据台账

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：本文记录用户提供的实验快照，可能滞后于最新 checkpoint、配置和评估脚本。实现解释以当前代码、实际配置、checkpoint 元数据和原始日志为准；如有冲突，应更新本文。用户实验不自动等同于代码事实。

## BPVA-ablation-2026-08-19-01

- 证据类型：【用户实验】
- 原始截图：`/root/.cursor/projects/vla-workspace-my-tbot/assets/image-a58c5e18-444d-447b-a38c-b6c84bd8261b.png`
- 截图条件：四列均标注 `1 epoch, bad obs`。
- 任务顺序：`adjust_bottle ac=16`、`handover_block ac=16`、`move_can_pot ac=16`、`place_bread_skillet ac=24`、`place_dual_shoes ac=16`、`place_object_scale ac=24`、`press_stapler ac=30`、`scan_object ac=24`、`stack_bowls_three ac=24`、`turn_switch ac=24`。

### 原始值

- TBot-base-clean：`100, 50, 30, 50, 40, 90, 80, 40, 90, 30`；宏平均 `60%`。
- BPVA 正常 BP：`90, 10, 30, 70, 10, 70, 80, 30, 100, 50`；宏平均 `54%`。
- BPVA without valid BP tokens：`100, 40, 30, 70, 50, 70, 80, 20, 100, 40`；宏平均 `60%`。
- BPVA black BP tokens：`90, 40, 50, 70, 30, 70, 90, 10, 80, 30`；宏平均 `56%`。

### 可支持的结论

- 【用户实验】同 BPVA 条件下，without valid BP tokens 比正常 BP 高 `+6pp`，black BP tokens 高 `+2pp`。
- 【推断】真实 BP 内容没有显示正贡献，且存在负干扰信号；当前 BP 条件路径或训练方式存在实质问题，值得暂停结构扩张并先隔离根因。
- 【用户实验】任务级响应不一致：normal→no-valid 在 `handover_block` 为 `+30pp`、`place_dual_shoes` 为 `+40pp`；normal→black 在 `stack_bowls_three`、`scan_object`、`turn_switch` 均出现 `-20pp`。

### 不可支持的结论与未知项

- 不能仅凭该实验断言 BP encoder 架构本身已被证伪；训练目标、BP episode 质量/选择、数据、mask 实现与 `bad obs` 都是竞争解释。
- TBot 与 BPVA 是不同 checkpoint，`60%` 对 `54%` 不是纯架构因果对照。
- 【待确认】每任务 rollout 数。百分比为 10% 步长，可能每任务 10 次，但不能据截图确认。
- 【待确认】是否固定同一环境 seed/初态、是否逐 rollout 配对、是否为同一 BPVA checkpoint、`bad obs` 的定义、without-valid/black 的精确 mask 与张量实现、成功率置信区间。

### 下一次最小复现实验

固定同一 BPVA checkpoint、环境 seed/初态、query observation 和评估脚本，仅改变 BP：正确 BP、`mask=False`、`mask=True` 黑/零 BP、错任务 BP。至少使用 3 个 seeds 或增加每任务 rollout 数；记录每次 rollout 的结果、exact BP source dataset/episode、mask 参数和 checkpoint hash。报告每任务与宏平均的 Wilson 置信区间；若逐 rollout 配对，使用 McNemar 检验。
