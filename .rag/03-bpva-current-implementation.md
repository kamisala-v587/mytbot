# BPVA 当前实现

> 截至/最后核验：2026-08-19
>
> 时效与代码优先声明：本文描述当前仓库静态代码。checkpoint、启动参数与数据可能改变实际行为；如与本文不符，以当前代码和实际运行配置为准，并更新本文；如代码与文档不符，以当前代码为准并更新本文。

## 一句话定义

- 【代码事实】BP 是行为示范轨迹，不是普通文本 prompt。一个 BP 由至多 K 个块组成；每块包含三相机当前帧、该帧 state、从该帧起的未来 action chunk，以及 padding/mask 信息：`src/lerobot/datasets/behavior_prompt_dataset.py:278-337`。

## 训练数据流与采样

1. 【代码事实】同一 repo 同时构建 current dataset 与 prompt dataset；prompt dataset 只为 action 配置未来 T 步 delta timestamp，避免为 BP 额外解码历史图：`src/lerobot/datasets/factory.py:977-1007`。
2. 【代码事实】建立 episode→frame 与 task→episodes 索引；候选首先取当前 `task_index` 的 episodes，找不到才退化为全数据 episodes：`src/lerobot/datasets/behavior_prompt_dataset.py:222-246`。
3. 【代码事实】默认 `same_episode_policy="avoid"`：有不同 episode 时排除当前 episode；若没有则允许同 episode。`forbid` 才会在无不同 episode 时抛错。最终使用有种子的 RNG 选择：`src/lerobot/datasets/behavior_prompt_dataset.py:241-256`。
4. 【代码事实】选中 episode 后，按轨迹长度与 action chunk size 决定最多 K 块，并在整条轨迹上 linspace 均匀取关键帧：`src/lerobot/datasets/behavior_prompt_dataset.py:258-276`。
5. 【代码事实】短轨迹后续 pad 到固定 K，长轨迹再均匀采样；无效块写入 mask：`src/lerobot/transforms/core_bp.py:28-65`。

结论：训练策略应表述为“同任务优先、尽量不同 episode”，不能写成“保证不同 episode”，也不能写成任意文本 prompt。

## 推理选择与缓存

- 【代码事实】请求必须给非空 `task_type`，服务端按 YAML 映射到 BP 数据集：`server/bpva_serve.py:492-496`；映射声明见 `.config/bpva_task_bps.yml:1-6`。
- 【代码事实】每个 task 第一次请求时从 episode 0 开始确定性扫描，跳过不可读项，取首个可读 episode；校验后按 task 缓存，后续请求复用同一 BP：`server/bpva_serve.py:272-295`。
- 【代码事实】episode 内块数为 `min(K, ceil(len/T))`，帧位置为 linspace；随后执行相机映射、pad/sample、resize、delta action（如启用）、normalize 与维度 padding：`server/bpva_serve.py:297-350`。
- 【推断】这保证单服务进程内可复现，却造成训练随机 prompt 与推理固定首 episode 的分布差异，并可能让一个有偏 episode 支配该任务全部推理。

## BP 编码器

- 【代码事实】输入 schema 明确是 `images[B,K,3,H,W]`、`state[B,K,D]`、`action[B,K,T,D]`，输出 `[B,K,2048]`；每 token 表示完整 BP 块：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:84-101`。
- 【代码事实】三相机通过 timm ViT（配置默认 CLIP ViT-B/16，可共享且可冻结）各形成一个 768 维 token；state、整个未来 action chunk 各投影成一个 768 维 token：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:103-187`。
- 【代码事实】五个 modality tokens（3 image + state + action）拼平，经 MLP 压成一个 2048 维 chunk token；action 有 step embedding，模态有 type embedding：`src/lerobot/policies/BPVA/bp_transformer_obs_encoder.py:230-282`。
- 【代码事实】配置默认 K=4，但实际 v3 配置把训练 K 设为 10；讨论实验必须引用实际配置/checkpoint：`src/lerobot/policies/BPVA/configuration_bpva.py:148-166`；`configs/bpva_train_v3.jsonc:61-78`。

## 与当前观测及 TBot 主干的连接

- 【代码事实】当前观测数据变换仅生成三相机 Qwen 图像 tokens，不附加 task instruction 文本：`src/lerobot/transforms/core_bp.py:236-270`。
- 【代码事实】当前图像仍经 Qwen3-VL visual encoder；所谓 `lang_tokens` 仍作为 image placeholder 位置结构：`src/lerobot/policies/BPVA/modeling_bpva.py:1062-1094`。
- 【代码事实】K 个 2048 维 BP tokens 直接拼到当前 Qwen 图像 prefix 后，形成 `[current Qwen image tokens ; BP tokens]`；BP token 不是 Qwen image token，仅分配伪 token id 复用 RoPE：`src/lerobot/policies/BPVA/modeling_bpva.py:1097-1147`。
- 【代码事实】训练仍构造 middle（Cosmos/3D）与 suffix（state/noisy action），三专家联合前向并优化 action、future Cosmos、DA3 三类 loss：`src/lerobot/policies/BPVA/modeling_bpva.py:1859-1953`。
- 【代码事实】v3 默认仅冻结 BP 的 ViT；Qwen vision 不冻结且三专家不采用 only 模式：`configs/bpva_train_v3.jsonc:69-89`；冻结逻辑见 `src/lerobot/policies/BPVA/modeling_bpva.py:874-934`。

## 审计补充：选择回退、loss 与缓存语义

- 【代码事实】训练样本若缺失 `task_index`，当前代码以 `0` 静默代替；构建 task 映射时也对缺失字段回退 `0`。若该 task 没有候选 episode，随后再回退到全部 episodes：`src/lerobot/datasets/behavior_prompt_dataset.py:232-246`、`278-286`。这会掩盖数据 schema 错误并可能产生跨任务 BP。
- 【代码事实】当前总 loss 只有 action flow-matching、Cosmos `loss_gen` 与 DA3 `loss_3d`；没有 BP-specific reconstruction、contrastive、alignment 或 usage loss：`src/lerobot/policies/BPVA/modeling_bpva.py:1922-1953`、`2694-2708`。因此不能仅凭训练收敛断言模型使用了 BP。
- 【代码事实】仓库已有 synthetic mask ablation 服务：全零 images/state/action，可把 BP mask 设为 False（无 BP）或 True（零值有效 BP）：`server/bpva_serve_mask_batch.py:1-9`、`117-159`。
- 【待验证】仓库未见与该服务绑定的成功率结果文件；“已有消融入口”不等于“已有消融结论”。
- 【代码事实】生产 `BehaviorPromptCache` 缓存的是 transform 后的 raw BP 张量字典（图像/state/action/mask），不是 `BPObsEncoder`/ViT 输出特征：`server/bpva_serve.py:272-283`、`297-350`。每次请求仍会进入 BP encoder；不能称为 ViT feature cache。

## 不允许的错误表述

- 错：“BP 是给模型的一段文字。” 正：BP 是视觉—状态—未来动作示范轨迹。
- 错：“BPVA 没有语言/VLM。” 正：当前变换不注入 instruction 文本，但 `task_type` 选择 BP，当前图像经 Qwen VLM 视觉编码，understanding expert 仍是 Qwen 模型。
- 错：“BPVA 只训练 BP encoder。” 正：按当前 v3 配置，TBot 的 und/gen/act 与视觉、世界、动作路径仍参与训练；仅外部 Cosmos/DA3 teacher 与 BP ViT 有明确冻结。
- 错：“训练和推理 prompt 选择一致。” 正：训练是同任务候选上的 seeded 随机选择；推理是映射数据集中首个可读 episode 的确定性缓存。
