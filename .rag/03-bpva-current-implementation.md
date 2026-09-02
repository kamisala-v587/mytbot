# BPVA 当前实现

> 截至/最后核验：2026-09-01
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

## BPVAv2 Query Compressor（2026-09-01）

- 【代码事实】`policy.bp_camera_keys` 定义训练后固定的相机槽位与顺序；`dataset.bp_camera_keys` 必须是其非空子集。数据集出现未知槽位会报错，子集会 warning；transform 仍只处理 active dataset keys，不生成空白图：`src/lerobot/datasets/factory.py:980-1000`；`src/lerobot/policies/BPVAv2/configuration_bpva.py:29-86`。
- 【代码事实】每个 active 相机的 `[B,K,patches,patchdim]` 经共享 Qwen visual 编码，按 `grid.prod / spatial_merge_size²` 恢复每图完整 `[N,D_visual]` tokens；同一次 forward 的全部有效图必须有相同 N。编码器不再执行图内 mean 或跨相机 mean，也不配置固定 N：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`。
- 【代码事实】视觉 tokens 逐 token 经 `LayerNorm + Linear(D_visual→512)`；state 每 chunk 由 `32→256→512 + LN` 形成 1 token，action 每 step 由同形 MLP 形成 T tokens并叠加 learned position。可选 visual/state/action type embedding 与固定 camera-slot embedding 默认开启：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`；`src/lerobot/policies/BPVAv2/configuration_bpva.py`。
- 【代码事实】每 chunk memory 为 `[C*N + 1 + T,512]`。缺失 policy 相机槽位由 active 图像推导出的 N 个零 tokens 与 false mask 补齐；state availability、`action_is_pad` 与 image masks 决定有效性，embedding 后无效项再次清零。active chunk 若全部 memory 无效会明确报错，padding chunk 为避免 MHA 全 mask NaN 临时保留一个零 key，最终输出仍清零：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`。
- 【代码事实】5 个 learned queries 经 2 层 pre-norm CrossSelfBlock：cross-attention(query,memory)、query self-attention、`512→2048→512` FFN，均为 residual；输出 `LN+Linear(512→2048)`，每 chunk 得到 `[5,2048]`：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`。
- 【代码事实】K chunks 先得到 `[B,K,5,2048]`，按 chunk 加 learned position，再 flatten 为 `[B,K*5,2048]`；chunk mask 和 indices 各扩展 5 次，无效 chunk tokens 清零。model prefix 直接拼接 current Qwen tokens 与 KQ BP query tokens：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`；`src/lerobot/policies/BPVAv2/modeling_bpva.py:1086-1140`。
- 【代码事实】`bp_freeze_shared_visual=true` 只让 BP 路径的 shared visual 调用处于 `no_grad`，projection、compressor 和 embeddings 仍训练；Qwen visual 仍只在主干注册一次：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py`；`src/lerobot/policies/BPVAv2/modeling_bpva.py`。
- 【checkpoint 边界】新配置写入 `bp_encoder_version="query_compressor_v1"`。非 BPVAv2 来源与缺失/旧版本 BPVAv2 来源均跳过整个 BP encoder，只加载兼容主干；同版本仅在 camera keys、query/layer/head/compressor/hidden、action chunk、state/action max dims 全匹配时加载，否则抛 `ValueError`：`src/lerobot/policies/BPVAv2/modeling_bpva.py:2336-2560`。
- 【配置事实】B200 示例固定 policy 三槽 `image0/1/2`，dataset 只 active `image0`，用于展示 Qwen 编码后两槽 zero-token/false-mask padding：`configs/B200/bpvav2_config_test.jsonc`。

## BPVAv2 初始化 checkpoint 工具（2026-09-01）

- 【代码事实】`--num-policy-camera-slots`（默认 3）决定 checkpoint 固定槽位，`--num-bp-cameras` 只决定随机输入 active 子集且必须不超过 slots；transform 只看到 active 图像字典，config 始终保留完整 policy slots：`tools/generate_bpvav2_init_checkpoint.py`。
- 【代码事实】保存验证要求 `query_compressor_v1`，并检查 query tokens、state/action MLP、action position、compressor layers、output projection，以及启用时的 type/camera embedding 权重；仍拒绝 BP encoder 下重复保存 Qwen visual：`tools/generate_bpvav2_init_checkpoint.py`。
- 【使用方式】`python tools/generate_bpvav2_init_checkpoint.py --tbot-checkpoint /path/to/tbot --output-root /path/to/output --qwen3-vl-dir /path/to/Qwen3-VL --cosmos-dir /path/to/Cosmos --num-policy-camera-slots 3 --num-bp-cameras 1`。
- 【边界/历史记录】初始化 checkpoint 工具本身仍包含由用户显式执行的随机 forward 门禁；下节记录的是随后使用独立 smoke 工具完成的真实数据 schema/runtime 验证，不能回溯解释为该初始化工具的训练或质量验证。

## BPVAv2 真实数据 schema/runtime smoke（2026-09-01）

### 执行条件与结果

- 【运行证据】在真实 repo `/share/RoboTwin-LeRobot-v3.0/beat_block_hammer/aloha-agilex_randomized_500` 上固定读取 current sample 0（episode 0、task_index 0）；数据映射后的真实 state/action 维度为 14/14，BP active 相机仅 canonical `observation.images.image0`。执行摘要为 `python tools/smoke_test_bpvav2.py --repo-id <上述repo> --sample-index 0 --skip-inference`，以及去掉 `--skip-inference` 并指定 `--num-inference-steps 1` 的 forward+inference 命令。smoke 工具的真实数据构建、样本选择和输出字段见 `tools/smoke_test_bpvav2.py:108-162`、`208-265`。
- 【运行证据】实际运行配置为 BP K=4（来自 source config/default，而非 `configs/B200/bpvav2_config_test.jsonc` 中的示例 K=10）、action chunk=50、compressor dim=512、每 chunk 5 queries、2 层 compressor，`lambda_gen=0`、`lambda_3d=0`，inference step=1。对应 BPVAv2 默认结构定义见 `src/lerobot/policies/BPVAv2/configuration_bpva.py:102-116`；smoke 对 source config 的复制及 loss/inference 覆盖见 `tools/smoke_test_bpvav2.py:82-104`。
- 【运行证据】TBot checkpoint `/home/jovyan/workspace/models/tbot-pretrain-v2` 中匹配 `model.bp_obs_encoder.*` 的 tensor 数为 0，因此本次 BP encoder 是随机初始化；其参数量为 10,835,456，参数统计 mean=0.000683585、std=0.0680857。checkpoint key 计数与参数统计实现见 `tools/smoke_test_bpvav2.py:189-194`、`222-249`。
- 【运行证据】真实 forward 成功且 finite：总 loss/action loss 均为 0.548722，gen/3d loss 均为 0；1-step inference 成功且 finite，actions shape 为 `[1,50,14]`。finite 门禁和 forward/inference 调用见 `tools/smoke_test_bpvav2.py:203-205`、`251-265`。

### 实际 shape trace

- 【运行证据】processor BP pixels 为 `[1,4,256,1536]`，grid 为 `[1,4,3]`；Qwen visual concat 为 `[256,2048]`，即每 image `N=64`；projected active image 为 `[1,4,64,512]`。代码在 processor/Qwen/projection 边界记录这些 shape，并依据 grid 与 spatial merge 恢复每图 token 数：`src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py:285-365`。
- 【运行证据】固定 policy camera slots 为 `[1,4,1,64,512]`；state tokens `[1,4,1,512]`，action tokens `[1,4,50,512]`，因此 chunk memory `[1,4,115,512]`。memory 的拼接和 trace 位置见 `src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py:193-242`。
- 【运行证据】flatten 前 learned queries 为 `[4,5,512]`，compressor layer 0/1 输出均为 `[4,5,512]`；output projection 为 `[1,4,5,2048]`，最终 flattened tokens `[1,20,2048]`、mask `[1,20]`。query/layer/output trace 见 `src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py:256-269`，chunk position 与 flatten 见 `src/lerobot/policies/BPVAv2/bp_transformer_obs_encoder.py:460-500`。
- 【代码事实】`tools/smoke_test_bpvav2.py` 是可复用的真实样本 smoke 入口，默认启用 shape trace，可选择跳过 inference，并显式检查 loss 与 actions 的 finite 性：`tools/smoke_test_bpvav2.py:32-52`、`203-265`。

### 本次暴露的问题、warning 与证据边界

- 【运行证据】初次真实执行暴露 `LeRobotDataset.active_camera_keys` 的 Parquet projection bug：MP4-backed video keys 被错误送入 Parquet columns。该问题已修复；投影路径现在明确排除 `meta.video_keys`，video frame 仍在后续 decode 阶段注入：`src/lerobot/datasets/lerobot_dataset.py:876-910`、`src/lerobot/datasets/lerobot_dataset.py:1069-1092`。
- 【运行证据】两次单次环境观测耗时分别约 45.8 秒（forward-only）与 47.8 秒（forward + 1-step inference）。【边界】这些值没有 warmup、重复次数、同步/分段计时或环境控制，只能作为本次 smoke 的运行记录，不能作为性能 benchmark。
- 【运行证据】加载时出现 missing tied `embed_tokens` weight warning，但本次 forward 与 inference 均成功且 finite。【边界】该 warning 尚待解释、当前为非阻断项；不能据此宣称 checkpoint 加载“完全无问题”。torchvision video deprecation/future warning 同样不是本次功能失败，但应在依赖升级时处理。
- 【边界】这是一次真实数据的 schema/runtime smoke，仅证明该固定样本、配置和环境下数据路径、forward 与单步 inference 可运行且数值 finite；它不是训练效果、任务成功率或泛化能力证据。BP encoder 本次随机初始化且未训练，0.548722 loss 没有模型质量意义。
