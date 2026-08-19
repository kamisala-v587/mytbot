# BPVA 架构评估与优化路线

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：以下评估基于当前代码和公开设计，未运行的结果均标为【推断】或【待验证】。若代码与本文不符，以当前代码为准并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## 总体判断

- 【推断】BPVA 的核心价值是把任务语义与动作模式编码进具身示范，使策略能从“如何做”的轨迹条件化，而不只依赖 instruction 文本。
- 【推断】当前最大科学风险不是容量不足，而是 BP 选择带来的任务标签捷径、episode 泄漏，以及训练随机 BP / 推理固定首 BP 的分布差异。先建立可信消融，再增加模型复杂度。

## 优势

- 【代码事实】BP 同时携带三视角、state 与未来动作，覆盖几何、接触过程和控制风格：`src/lerobot/datasets/behavior_prompt_dataset.py:297-337`。
- 【代码事实】K 个 token 是固定小预算 prefix，能直接接入现有 2048 维 MoT，不改 middle/suffix 接口：`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`。
- 【推断】同任务不同 episode 的示范可能提供实例变化下的动作先验，比固定文本更直接。
- 【代码事实】保留 Cosmos、DA3 与 flow matching 目标，使 BP 条件化并未删除世界/动作学习路径：`src/lerobot/policies/BPVA/modeling_bpva.py:1922-1953`。
- 【代码事实】但当前没有专门约束 BP 表征或 BP—action 对齐的 loss；总 loss 仅为 action + 加权 Cosmos + 加权 DA3：`src/lerobot/policies/BPVA/modeling_bpva.py:2694-2708`。

## 缺点与风险

### 1. 泄漏与捷径

- 【推断】若训练/验证按 frame 而非 episode/task 隔离，BP 可能来自验证 episode，甚至在 `avoid` 无候选时回退同 episode；模型可记忆场景或动作答案。
- 【代码事实】`avoid` 不是严格隔离，只有 `forbid` 在无不同 episode 时失败：`src/lerobot/datasets/behavior_prompt_dataset.py:247-255`。
- 【推断】`task_type → 数据集 → 固定 episode` 本身是强任务标识。即使无文本，模型也可能从 BP 外观/动作模板推断 task，而非理解当前场景。

### 2. 训练—推理 prompt 分布差异

- 【代码事实】训练对同任务 episode 做 RNG 选择；推理固定缓存首个可读 episode：`src/lerobot/datasets/behavior_prompt_dataset.py:241-256`；`server/bpva_serve.py:272-295`。
- 【推断】训练见到 prompt 多样性，推理却被单一 episode 的质量、初态和执行风格锁定，容易产生方差和系统性偏置。

### 3. 压缩瓶颈

- 【代码事实】每块 3 图像 + 32D state + 50×32D action 最终压成单个 2048D token：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:230-267`。
- 【推断】单 token MLP fusion 可能丢失空间对应、动作阶段与跨块时序；action flatten 对变速/不同长度轨迹尤其脆弱。

### 4. prompt 选择鲁棒性

- 【推断】首可读不等于最相关/最高质量。相机标定、机器人形态、对象布局、动作速度或成功标签错配会误导策略。
- 【代码事实】当前映射粒度仅为 `task_type → dataset_path`，未表达 episode 质量、embodiment、场景属性或多候选：`.config/bpva_task_bps.yml:1-8`。

### 5. schema 静默回退与缓存误读

- 【代码事实】缺失 `task_index` 会静默按 0 处理；若对应 task 无候选，再回退全部 episodes：`src/lerobot/datasets/behavior_prompt_dataset.py:232-246`、`278-286`。应将 schema 缺失视为错误，而非正常泛化。
- 【代码事实】服务缓存 raw BP 张量，不缓存 ViT/BP encoder 特征：`server/bpva_serve.py:272-283`、`297-350`。

### 6. “无文本”不等于“无语言/VLM”

- 【代码事实】数据变换不注入 instruction 文本：`src/lerobot/transforms/core_bp.py:236-270`。
- 【代码事实】当前图像依旧经 Qwen3-VL visual，且 und/gen/act 主干保留：`src/lerobot/policies/BPVA/modeling_bpva.py:1062-1094`、`395-465`。
- 【推断】若目标是证明 behavior prompting 替代语言，必须加入带/不带 instruction、冻结/移除 language transformer 等受控对照，当前代码不足以支持该宣称。

## 已有初步消融结果（用户提供，2026-08-19）

- 【用户实验】实验 ID：`BPVA-ablation-2026-08-19-01`；原始截图：`/root/.cursor/projects/vla-workspace-my-tbot/assets/image-a58c5e18-444d-447b-a38c-b6c84bd8261b.png`。
- 【用户实验】实验条件均标注为 `1 epoch, bad obs`：TBot-base-clean 为 `60%`；BPVA 正常 BP 为 `54%`；同一 BPVA 条件下 without valid BP tokens 为 `60%`；black BP tokens 为 `56%`。
- 【用户实验】十任务原始成功率，顺序均为 `adjust_bottle / handover_block / move_can_pot / place_bread_skillet / place_dual_shoes / place_object_scale / press_stapler / scan_object / stack_bowls_three / turn_switch`：
  - TBot-base-clean：`100 / 50 / 30 / 50 / 40 / 90 / 80 / 40 / 90 / 30`。
  - BPVA 正常 BP：`90 / 10 / 30 / 70 / 10 / 70 / 80 / 30 / 100 / 50`。
  - BPVA without valid BP tokens：`100 / 40 / 30 / 70 / 50 / 70 / 80 / 20 / 100 / 40`。
  - BPVA black BP tokens：`90 / 40 / 50 / 70 / 30 / 70 / 90 / 10 / 80 / 30`。
- 【用户实验】同名 BPVA 实验条件范围内，屏蔽有效 BP 为 `60%`，比正常 BP 高 `+6pp`；黑色有效 BP 为 `56%`，高 `+2pp`。真实 BP 内容当前没有显示正贡献，并出现负干扰信号，因此 BP 条件路径或训练存在实质问题。
- 【证据边界】TBot `60%` 对 BPVA `54%` 混合了不同 checkpoint 与训练差异，因果力度弱于 BPVA 内部消融。现有结果不能把根因单独归因于 BP encoder 架构；训练目标、BP 选择、数据质量、mask 语义与 `bad obs` 条件均可能参与。
- 【待确认】截图数值均以 10% 为步长，可能是每任务 10 次 rollout，但尚未确认。还需确认每任务 rollout 数、是否共享环境 seed/初态、置信区间、`bad obs` 的准确含义、without-valid 与 black 的精确定义/实现参数，以及三项 BPVA 是否确为同一 checkpoint。

## 必做评估矩阵

### 基线与 BP 消融

- 【待验证/P0】固定 checkpoint、当前观测、环境 seed 与 query，逐项比较：正确 BP；mask=False 的无 BP；mask=True 的零值有效 BP；错任务 BP；同任务不同 episode BP；只乱 action；只乱 image；chunk 顺序反转；固定当前观测只更换 BP。仓库已有 mask/零值服务入口但没有结果：`server/bpva_serve_mask_batch.py:1-9`、`117-173`。
- 【方法要求】“错误 BP 仍配原 query action label”用于训练时，最直接的优化压力可能是让模型忽略不可靠 BP，而非证明模型能识别错误 BP。应把它作为 robustness/BP-dropout 训练，并在固定模型的反事实推理矩阵中单独测 BP sensitivity。
- 【待验证】补充 TBot 原始 instruction 基线、仅图像 BP、仅 state+action BP、仅 action BP；K ∈ {1,2,4,10,20}；每块 1 token 对比多 token/Perceiver。

### 选择与分布

- 【待验证】推理首 episode、随机 episode、多种子 episode、按当前观测检索 top-k、top-k ensemble/attention pooling。
- 【待验证】seen-task/seen-scene、seen-task/new-scene、new-task、同 embodiment/跨 embodiment 分层报告。
- 【待验证】严格 episode split，记录 BP episode 与 query episode 的去重 hash；跨 split 物体布局/背景近重复检查。

### 指标

- 【待验证】成功率均值与置信区间、按任务最差分位、prompt seed 方差、错配性能降幅、无 BP 降幅、推理延迟/显存、BP cache 建立耗时。
- 【待验证】通过 attention/gradient attribution 或 BP shuffle sensitivity 检查模型是否真正利用 BP；避免只看平均成功率。

## 优化优先级

### P0：先保证结论可信

1. 强制 train/val/test 与 BP episode 白名单隔离；训练/评估使用 `same_episode_policy=forbid`，并让缺失 `task_index` 直接失败，记录 source episode。
2. 下一轮先做最小配对矩阵：正确 BP、`mask=False`、`mask=True` 黑/零、错任务 BP；固定同一 checkpoint、环境 seed/初态及其他变量。至少 3 个 seeds 或提高每任务样本量，报告 Wilson CI；若逐 rollout 配对则使用 McNemar 检验，并记录 exact BP source episode 与 mask 参数。
3. 统一训练与推理选择分布：至少支持按种子确定性轮换多个 episode，并输出缓存 provenance。
4. 为 YAML 增加 dataset revision、embodiment、成功/质量过滤及 episode allowlist。

### P1：提高选择鲁棒性

1. 用当前 Qwen/独立视觉 embedding 做相似度检索，取得同任务 top-k BP，再以距离、相机有效性、轨迹长度和质量分重排。
2. 训练时加入 10%–30% null/mismatched BP 与 BP dropout，降低绝对依赖和 task-label shortcut。
3. 多 BP ensemble 或 token-level gating；报告 prompt seed 的均值、方差和最差值。
4. 将确定性 raw BP cache 扩展为带 provenance 的 ViT/BP encoder 特征缓存；须按 checkpoint、归一化、相机映射和设备/dtype 失效，避免陈旧特征。

### P2：缓解压缩瓶颈

1. 将 action chunk 切成多个 action 子段 token，并用 temporal encoder 建模块内/跨块顺序；图像采用 spatial resampler 保留区域结构，而不是每相机只取一个全局 token。
2. 使用真实 `source_time_ratio`/相对时间编码，而非仅 pad 后 chunk index；当前元数据在统一输入时被移除：`src/lerobot/transforms/core_bp.py:274-309`。
3. 比较单 token、action 子段 token、spatial resampler 与 temporal encoder 的收益—延迟曲线。
4. 增加 BP—action 对比/一致性目标，使正确 BP 与 query action 表征更接近、错配更远；同时使用 BP encoder、主干、action head 的分组学习率，避免新模块或预训练主干更新失衡。

### P3：长期研究

1. instruction 与 BP 双条件训练，并做 modality dropout，支持文本、示范或二者兼有。
2. 成功度/不确定性估计驱动 BP 选择；低置信时回退 null BP 或多候选。
3. 将 BP retrieval、world prediction 与动作成功联合优化，但必须保持可解释的离线索引和泄漏防护。

## 决策门槛

- 【推断】只有“正确 BP 显著优于 null BP，跨任务错配显著更差但不灾难性，多个合法 BP 方差可控，严格 split 下仍有效”同时成立，才能说明模型利用的是可迁移行为信息，而非任务标签或 episode 记忆。
