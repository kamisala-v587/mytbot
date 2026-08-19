# my_tbot 机器人策略知识库索引与维护协议

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：本知识库是截至上述日期的快照。涉及本仓库行为时，当前可执行代码、实际配置与 checkpoint 元数据优先于本文；如代码与文档不符，以当前代码为准，并在同一变更中更新 `.rag/`。公开论文只用于解释来源设计，不能覆盖本地实现事实。

## 使用入口

1. `01-internvla-a1.md`：InternVLA-A1 的论文与官方实现基础，明确区分 InternVL/InternVL3 与 InternVLA-A1。
2. `02-tbot-sa1-architecture.md`：官方 TBot-SA1 与本地 TBot 主干源码架构。
3. `03-bpva-current-implementation.md`：BP 数据、训练采样、推理缓存、编码与主干连接的当前代码事实。
4. `04-bpva-assessment-roadmap.md`：优势、风险、消融矩阵与分优先级路线。
5. `05-lineage-audit.md`：TBot / 本地 InternVLA_A1_3B / BPVA 的逐行同构与溯源边界。
6. `06-internvla-evolution.md`：A1.5、M1、N1 的定位、关系与不可混淆边界。
7. `07-experiment-evidence.md`：用户实验原始值、证据边界与最小复现实验。
8. [`07-session-handoff-2026-08-19.md`](07-session-handoff-2026-08-19.md)：本次会话截至 2026-08-19 的完整可恢复交接快照（不是实验结果）。
9. `AI_PROMPT.md`：供后续 Agent 复制使用的短提示词。

## 证据等级

- 【代码事实】：直接由本仓库当前源码/配置支持；必须给出 `路径:起始行-结束行`。
- 【公开资料】：论文、作者项目页或官方仓库；必须给出可点击原始 URL，并记录核验日期。
- 【高置信推断】：多处独立代码证据一致支持，但缺少精确历史/实验闭环；不得升级为法律或性能事实。
- 【推断】：从代码事实或公开资料推导，不得写成已验证结论。
- 【待验证】：缺少实验、运行证据或版本固定信息；应附最小验证办法。
- 【用户实验】：用户提供的截图、日志或口述结果；必须记录来源、条件、未知项，且不得自动升级为代码事实。

## 检索与回答协议

- 先按问题定位本索引，再读相关专题；涉及实现细节必须回查源码，而不是只引用知识库。
- 区分“官方 InternVLA-A1 / 官方 TBot-SA1 / 本地 TBot-SA1 / 本地 BPVA”；禁止把 InternVL（MLLM 家族或 understanding backbone）简称成 InternVLA-A1。
- “BP/prompt”默认指行为示范轨迹，不是普通 instruction 文本。若讨论文本 prompt，必须显式写“instruction 文本”。
- 不得声称 BPVA 完全摆脱 language/VLM：当前数据变换不注入 instruction 文本，但 `task_type` 仍参与 BP 选择，当前图像仍经 Qwen3-VL 视觉编码，understanding expert 仍是 Qwen 模型路径。
- 每条重要结论保留证据标签；不确定就标【推断】或【待验证】，不要用肯定语气补齐空白。

## 维护协议

1. 修改 BP schema、采样、映射、prefix 拼接、注意力、loss、冻结策略、服务端输入或关键配置时，同步更新对应专题与本索引。
2. 每次更新写新的“截至/最后核验”日期；保留稳定的小标题，便于 RAG 分块。
3. 行号漂移时重新核验引用；引用应指向最窄、能完整支持结论的范围。
4. 新增公开资料优先级：论文/arXiv → 作者官方仓库/项目页 → 模型卡；二手文章只能作为线索。
5. 评估结论必须记录数据划分、BP 来源 episode、task 映射、随机种子、checkpoint 与无/错配 BP 对照。
6. 发现文档与代码冲突：以当前代码为准，先标记冲突，再立即更新滞后文档；不得通过修改模型源码来“迁就文档”。

## 当前最重要的事实锚点

- 【代码事实】训练 BP 是同任务优先、默认尽量避开同 episode 的轨迹采样：`src/lerobot/datasets/behavior_prompt_dataset.py:232-256`。
- 【代码事实】每个 BP 块由三相机当前帧、state 与未来 action chunk 构成：`src/lerobot/datasets/behavior_prompt_dataset.py:278-337`。
- 【代码事实】推理按 `task_type` 查映射，确定性扫描并缓存首个可读 episode：`server/bpva_serve.py:272-295`。
- 【代码事实】每块压成一个 2048 维 token，K 块接到当前 Qwen prefix 后：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:252-282`；`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`。
