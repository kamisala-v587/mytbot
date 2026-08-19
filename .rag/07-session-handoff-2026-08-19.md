# 会话交接快照：BPVA / TBot-SA1（2026-08-19）

> 文档类型：会话交接快照，不是实验结果。
>
> 截至/最后核验：2026-08-19
>
> 本快照只记录本次会话截至 2026-08-19 已阅读、已核验或明确提出的上下文；没有虚构任何未执行实验。后续 AI 应先读 `.rag/README.md` 与本文件，再回查当前源码、配置、checkpoint 元数据和运行日志。

## 1. 用户目标与研究边界

- 【用户目标】在官方 TBot-SA1 上增强 BP（behavior prompting，行为示范提示）能力，主要依赖行为 prompting，而不是依赖 language/instruction 文本。
- 【用户目标】一般不修改 TBot 模型源码；利用 InternVLA 的公开资料建立 RAG，用于理解上游设计、定位代码谱系和指导验证。
- 【边界】“无 instruction 文本输入”不等于“完全去除 language/VLM”。当前 BPVA 仍使用 Qwen VLM/current vision、TBot 的 understanding/generation/action 主干，并由 `task_type` 参与 BP 选择。任何“主要依赖 BP 已被证明”的表述都必须等待反事实实验。

## 2. 名称、实现对象与边界

- `InternVL` / `InternVL3`：MLLM/understanding backbone 名称；不能简称为 InternVLA-A1。
- `InternVLA-A1`、`InternVLA-A1.5`、`InternVLA-M1`、`InternVLA-N1`：不同公开模型/系列；A1/A1.5 主要涉及 manipulation，M1 是 spatial grounding 双系统操控参考，N1 是 navigation，不是 manipulation。
- 官方 `TBot-SA1`：上游公开 World-Spatial-Action 机器人模型；本地代码中的 `TBot_SA1` 是当前实际实现对象，不能用官方 README 替代本地源码事实。
- 本地 `BPVA` / `bptbot`：行为提示扩展实现。`bptbot` 更多是环境、脚本或内部称呼；当前 canonical policy 名称是 `bpva`，后续文档和代码引用优先使用 BPVA。
- `TBot_SA1_Wan`：名称边界中的相关实现/实验称呼；不要把它与 A1.5 的 WAN latent foresight 直接等同，也不要把 WAN 路径自动归入当前 canonical BPVA。

## 3. 已核验源码事实（路径与精确行号）

### 3.1 TBot 三专家与视觉/世界/动作路径

- 【代码事实】TBot 的 `current / middle / suffix` 分别对应 `und / gen / act`：current/prefix 进入 understanding，middle 进入 generation，suffix 进入 action；`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:394-402`。
- 【代码事实】`und` 从 Qwen3-VL pretrained 初始化；`gen`/`act` 由 Qwen3-VL text config 构造并移除 token embedding/lm head，形成独立 transformer：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:404-467`。
- 【代码事实】三专家逐层进入统一 attention 后分别回写，最终各自 norm 并返回 prefix/middle/suffix：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:542-613`。
- 【代码事实】当前图像经 Qwen3-VL visual encoder 替换 image placeholder：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1013-1033`。
- 【代码事实】Cosmos 图像 tokenizer 路径将图像 resize 到 256×256、归一化到 [-1,1] 后编码：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1035-1044`；gen 输出训练未来 Cosmos latent，受 `lambda_gen` 控制：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1834-1845`；配置见 `src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:273-276`。
- 【代码事实】DA3 future 3D queries 追加到 middle；Cosmos visual 子块双向，3D query 可读对应 visual，query 组内/组间遵守实现中的 attention 约束：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1046-1131`。DA3 teacher 对齐配置与层选择见 `src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:278-297`。
- 【代码事实】suffix 使用 state、带噪 action 和时间编码；一次预测 `chunk_size` action，默认 50，推理默认 10 次去噪：`src/lerobot/policies/TBot_SA1/configuration_tbot_sa1.py:200-218`。
- 【代码事实】BPVA 保留 flow matching：`x_t=t*noise+(1-t)*actions`，目标速度为 `noise-actions`，对 action head suffix 输出做 MSE：`src/lerobot/policies/BPVA/modeling_bpva.py:1844-1857`、`1937-1946`。
- 【代码事实】外部 Cosmos/DA3 teacher 与 BP ViT 的冻结、以及 v3 训练边界见 `src/lerobot/policies/BPVA/modeling_bpva.py:874-934` 和 `configs/bpva_train_v3.jsonc:69-89`；v3 的 `freeze_vision_encoder=false`、`train_expert_only=false`、`train_vlm_only=false`、`lambda_gen=lambda_3d=0.01` 见 `configs/bpva_train_v3.jsonc:86-115`。

### 3.2 BP 数据采样与 schema

- 【代码事实】训练同时构建 current dataset 与 prompt dataset；prompt dataset 为 action 配置未来 T 步 delta timestamp：`src/lerobot/datasets/factory.py:977-1007`。
- 【代码事实】建立 episode→frame、task→episodes 索引；优先当前 `task_index` 的 episodes，找不到再退回全部 episodes：`src/lerobot/datasets/behavior_prompt_dataset.py:222-246`。
- 【代码事实】缺失 `task_index` 会以 `0` 静默回退；样本读取处同样回退为 0：`src/lerobot/datasets/behavior_prompt_dataset.py:232-246`、`278-286`。这可能掩盖 schema 错误并导致跨任务 BP。
- 【代码事实】`same_episode_policy` 支持 `avoid / allow / forbid`，默认 `avoid`；有不同 episode 时排除当前 episode，无不同 episode 时 `avoid` 允许同 episode，只有 `forbid` 抛错：`src/lerobot/datasets/behavior_prompt_dataset.py:241-256`。
- 【代码事实】选择 episode 后按轨迹长度和 action chunk size 决定最多 K 块，并在整轨迹上 linspace 取关键帧；不足 K 的块 padding 并写 mask：`src/lerobot/datasets/behavior_prompt_dataset.py:258-276`；`src/lerobot/transforms/core_bp.py:28-65`。
- 【代码事实】每个 BP chunk 包含三相机当前帧、当前 state、未来 action chunk 及 padding/mask：`src/lerobot/datasets/behavior_prompt_dataset.py:278-337`。
- 【代码事实】BP schema 为 `images[B,K,3,H,W]`、`state[B,K,D]`、`action[B,K,T,D]`，编码输出 `[B,K,2048]`：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:84-101`。
- 【代码事实】三相机各经 timm ViT 形成 768 维 token，state 与整段 future action 各投影为 768 维 token；五个 modality token 融合成一个 2048 维 chunk token：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:103-187`、`230-282`。默认 K=4，但 v3 训练配置 K=10：`src/lerobot/policies/BPVA/configuration_bpva.py:148-166`、`configs/bpva_train_v3.jsonc:61-78`。

### 3.3 BP 与当前观测、服务端和推理限制

- 【代码事实】当前数据变换生成三相机 Qwen 图像 tokens，不附加 instruction 文本：`src/lerobot/transforms/core_bp.py:236-270`。
- 【代码事实】当前图像仍经 Qwen3-VL visual encoder；`lang_tokens` 仍承担 image placeholder 位置结构：`src/lerobot/policies/BPVA/modeling_bpva.py:1062-1094`。
- 【代码事实】K 个 2048 维 BP token 直接拼到当前 Qwen 图像 prefix 后，形成 `[current Qwen image tokens ; BP tokens]`；BP token 不是 Qwen image token，而是使用伪 token id 复用 RoPE：`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`。
- 【代码事实】请求必须提供非空 `task_type`；服务端通过 YAML 映射到 BP 数据集：`server/bpva_serve.py:492-496`；映射见 `.config/bpva_task_bps.yml:1-6`。
- 【代码事实】服务端按 episode 0 起确定性扫描，跳过不可读项，缓存每个 task 的首个可读 episode，后续请求复用：`server/bpva_serve.py:272-295`。
- 【代码事实】服务端 episode 内 chunk 数为 `min(K, ceil(len/T))`，帧位置 linspace；再执行相机映射、pad/sample、resize、delta action、normalize 与维度 padding：`server/bpva_serve.py:297-350`。
- 【代码事实】已有 mask ablation server 可生成全零 images/state/action；`mask=False` 表示无 BP，`mask=True` 表示零值但有效的 BP：`server/bpva_serve_mask_batch.py:1-9`、`117-159`。
- 【代码事实】生产 `BehaviorPromptCache` 缓存的是 transform 后 raw BP 张量，不是 ViT/BP encoder 特征；每次请求仍进入 BP encoder：`server/bpva_serve.py:272-283`、`297-350`。
- 【代码事实】BPVA causal inference 当前不支持；causal 工具仍保留，但入口明确报错并要求 `attention_mask_mode='default'`：`src/lerobot/policies/BPVA/modeling_bpva.py:1259-1350`、`1976-2075`，尤其 `2072-2075`。

## 4. 已核验结论与证据边界

- 【已核验结论/代码事实】BPVA 的当前数据变换移除了 instruction 文本输入，但这不能证明模型主要依赖 BP。它仍依赖 `task_type`（选择 BP）、Qwen VLM/current vision，以及 TBot 的 `und/gen/act` 主干。
- 【已核验结论/代码事实】没有 BP-specific loss；当前总 loss 是 action flow matching、Cosmos `loss_gen` 与 DA3 `loss_3d`：`src/lerobot/policies/BPVA/modeling_bpva.py:1922-1953`、`2694-2708`。
- 【高置信推断】训练按 seeded RNG 选同任务 BP，推理固定缓存首个可读 episode；这造成训练随机 prompt 与推理固定 prompt 的分布偏移：训练 `src/lerobot/datasets/behavior_prompt_dataset.py:241-256`，推理 `server/bpva_serve.py:272-295`。
- 【高置信推断】每个 chunk 的三图像+state+整段 action 被压成单个 2048 token，存在空间、动作阶段和跨 chunk 时序信息瓶颈：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:230-282`。
- 【用户实验/已有台账】`BPVA-ablation-2026-08-19-01` 的 `1 epoch, bad obs` 截图记录：TBot-base-clean 60%，正常 BPVA 54%，无有效 BP token 60%，黑 BP token 56%。同名 BPVA 内部无有效 BP 比正常高 +6pp，黑 BP 高 +2pp；但未知 rollout 数、环境 seed/初态、bad obs 定义、mask 实现细节和 checkpoint 是否相同。详见 `.rag/07-experiment-evidence.md`。
- 【证据边界】已有训练日志只能证明 loss/step 正常运行，不能证明 BP 增益、BP 被模型使用或行为 prompting 优于 instruction。没有已执行实验可填补该因果空白。

### 当前输出目录与配置事实

- 【代码/配置事实】已见输出目录：`outputs/TBot`（`configs/tbot_sft_h100.jsonc:95`）、`outputs/BP_TBot/SFT-rand`（`configs/bp_tbot_train.jsonc:105`）、`outputs/BPVA/SFT-pro6k-submit`（`configs/bpva_train.jsonc:127`、`configs/bpva_train_v3.jsonc:126`）、`outputs/BPVA/SFT-Robotwin-smoke`（`configs/bpva_sft_smoke.jsonc:126`）、`outputs/BPVA/SFT-Robotwin`（`configs/bpva_sft_robotwin.jsonc:126`）、`outputs/TBot_SA1/pretrain_v1`（`configs/pretrain_config.jsonc:90`）。
- 【代码/配置事实】这些是配置声明的 output_dir，不是本快照对实际磁盘 checkpoint、训练完成状态或指标的保证；实际运行时应重新检查目录、checkpoint hash、配置快照和日志。

## 5. InternVLA 公开资料结论与 URL

- 【公开资料】InternVLA-A1 是“理解—视觉前瞻—动作”统一模型，采用 understanding、generation、action 三专家 MoT；Cosmos latent foresight 预测未来视觉 latent，action expert 用 flow matching 预测连续 action chunk：[论文](https://arxiv.org/html/2601.02456v2)，[官方 A1 分支](https://github.com/InternRobotics/InternVLA-A-series/tree/InternVLA-A1)。
- 【公开资料】A1 论文区分 2B（InternVL3 + Qwen2.5 gen/act）与 3B（Qwen3-VL + Qwen3 gen/act），因此 InternVL/InternVL3 不能与 InternVLA-A1 混名：[论文 §3.4](https://arxiv.org/html/2601.02456v2#S3.SS4)。
- 【公开资料】A1.5 训练期用 WAN latent foresight（冻结 WAN2.2-5B 监督 task-relevant future dynamics），action-only 推理跳过 WAN 视频分支，再用 flow matching 输出 action：[官方系列仓库](https://github.com/InternRobotics/InternVLA-A-series)。这不是当前 TBot/BPVA 的 Cosmos 路径，也不是 BP prefix。
- 【公开资料】M1 是 spatial grounding 双系统，整合 language head 与 action head，并做 image QA/spatial grounding：[官方仓库](https://github.com/InternRobotics/InternVLA-M1)。它不是 TBot 的 und/gen/act 三专家，也不是三相机+state+future action 的 BP schema。
- 【公开资料】N1 面向 embodied navigation/VLN/路径规划，是导航模型而非 manipulation policy：[InternNav 官方仓库](https://github.com/InternRobotics/InternNav)，[技术报告](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)。
- 【代码事实】本地 `InternVLA_A1_3B` 与 TBot 的三专家逐层联合 attention、Qwen current prefix、Cosmos middle、state/noisy-action suffix、flow-matching action loss/sampling 存在大量同构证据：`src/lerobot/policies/InternVLA_A1_3B/modeling_internvla_a1.py:131-222`、`242-454`、`619-910`；TBot 对应 `src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:281-352`、`394-613`、`1013-1044`、`1671-1864`。TBot 另加入 3D/DA3：`src/lerobot/policies/TBot_SA1/modeling_tbot_sa1.py:1046-1227`、`1371-1466`。
- 【方法局限】静态逐行相似和 SequenceMatcher 结果（约 910 个 matching lines、ratio≈0.464879）只能支持代码同构/谱系线索，不能证明精确上游 commit、传播方向、原创性、许可证兼容或法律结论；需固定 hash、完整 git history、commit-aware blame、AST/语义 diff 和许可证清单才能进一步核验。详见 `.rag/05-lineage-audit.md`。

## 6. 已创建交付物与仓库注意事项

- `.rag/README.md`：知识库索引、证据等级、检索回答协议、维护协议。
- `.rag/01-internvla-a1.md` 至 `.rag/06-internvla-evolution.md`：A1 基础、TBot 架构、BPVA 当前实现、评估路线、谱系审计、A1.5/M1/N1 边界。
- `.rag/07-experiment-evidence.md`：用户提供的实验截图数值与证据边界。
- `.rag/AI_PROMPT.md`：后续 AI 使用的短提示词。
- Canvas 绝对路径：`/vla/workspace/my_tbot/.rag/BPVA-architecture-assessment.canvas.tsx`。此前已核验无 lint/type 错误；本任务不修改 Canvas。
- `.rag/` 由仓库 `.gitignore:14` 忽略。即使文件存在，普通 `git status` 也可能不显示；本快照未修改 `.gitignore`。如需版本化，应由用户明确决定是否强制添加，而不是在本任务中改变忽略规则。

## 7. 下一步行动清单

### P0：先建立 BP 因果证据

在固定 checkpoint、当前观测、环境 seed/初态、query action label、评估脚本和推理参数下，完整运行并逐 rollout 记录 provenance：

1. 正确 BP；
2. `mask=False`（无 BP）；
3. `mask=True` + 全零 BP（零值但有效）；
4. 错任务 BP；
5. 同任务不同 episode BP；
6. 只乱 action（image/state 不变）；
7. 只乱 image（state/action 不变）；
8. chunk 顺序反序；
9. 固定当前观测，只替换 BP（最关键的 BP counterfactual）。

同时执行：直接 BP 输入而非只依赖 `task_type` 的接口设计；严格 episode split；训练/评估 `same_episode_policy=forbid`；缺失 `task_index` 直接失败；为每个 BP 保存 dataset revision、task、episode、frame、embodiment、相机映射、mask、checkpoint hash 等 provenance；报告成功率及 Wilson CI、按任务最差分位、BP seed 方差、错配降幅、无 BP 降幅、延迟/显存与 cache 建立时间；缓存 ViT/BP encoder 特征（按 checkpoint、归一化、相机映射、设备/dtype 失效），而不是只缓存 raw BP。

### P1：选择与鲁棒性

- 推理首 episode、随机 episode、多 seed、当前观测检索 top-k 及 ensemble 对照。
- 加 null/mismatched BP 与 BP dropout；增加 BP sensitivity / shuffle attribution 检查。
- YAML 增加 dataset revision、embodiment、质量/成功过滤和 episode allowlist。

### P2：缓解压缩瓶颈

- action chunk 切成多个 temporal tokens；使用 temporal encoder 保留模块内/跨块顺序。
- image 使用 spatial resampler 保留区域结构；比较单 token、action 子段 token、spatial resampler、temporal encoder 的收益—延迟曲线。
- 保留真实 `source_time_ratio`/相对时间编码；目前统一输入会移除相关元数据：`src/lerobot/transforms/core_bp.py:274-309`。

## 8. 维护协议与恢复顺序

- 【维护协议】每次维护记录“截至/最后核验”日期；本文件当前为 2026-08-19 会话快照。
- 【维护协议】代码优先：代码、实际配置、checkpoint 元数据和运行日志优先于 `.rag` 文字；代码与文档冲突时更新文档，不修改模型源码来迁就文档。
- 【维护协议】重要陈述必须带证据等级标签：`【代码事实】`、`【公开资料】`、`【高置信推断】`、`【推断】`、`【待验证】`、`【用户实验】`；不要把推断写成事实。
- 【维护协议】公开资料给原始 URL 和核验日期；本快照的公开 URL 沿用专题文档，后续应重新检查页面和版本。
- 【恢复顺序】先读 `.rag/README.md` → 本文件 → 相关专题 `01`–`06` 与 `07-experiment-evidence.md` → 当前源码/配置 → checkpoint/log provenance → 再决定是否修改实现或运行实验。
- 【最终提醒】这是交接快照，不是实验报告；截至本文件日期没有任何未执行实验被写成已完成结果，也没有因本任务修改模型源码、配置、现有项目文档、`.gitignore` 或 Canvas。
