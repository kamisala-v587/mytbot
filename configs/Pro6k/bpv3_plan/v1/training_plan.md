### BPV3 v1 简单训练计划

目标：从无 q35 VLA 预训练的 BPV3 基座开始，只加载 Qwen3.5 VLM 权重；用不参与测评的 9 个任务做 warmup，再在 41 个适合 BP 的 clean 任务和 myblocks clean 上训练跨轨迹 BP。

### 数据划分

Stage1 warmup 数据：`stage1_excluded9_clean_rand_local.txt`，包含 9 个暂不参与测评任务的 clean + randomized：`move_pillbottle_pad, move_stapler_pad, place_a2b_left, place_a2b_right, place_bread_basket, place_object_basket, place_object_scale, place_object_stand, stamp_seal`。

Stage2 数据：`stage2_robotwin41_clean_plus_myblocks_local.txt`，包含从原 50 任务清单中排除上述 9 个任务后的 41 个 Robotwin clean repo，外加 `/vla/workspace/my_robotwin/data/myblocks_ranking_rgb/my_demo_clean_processed_Lerobot`。

### 配置

- Stage1：`stage1_bpv3_excluded9_warmup_10k.jsonc`，`pretrained_path` 为q35-pt-15w，只加载 `/vla/workspace/models/q35/base_800m`，`bp_sampling_mode=origin`。
- Stage2：`stage2_bpv3_robotwin41_clean_myblocks_20k.jsonc`，需要先把 `policy.pretrained_path` 中的 `<RUN_DIR>` 替换为 Stage1 实际运行目录，`bp_sampling_mode=avoid`。
- 两阶段都启用 `use_language=true`，BP 示教使用三路图像 `image0/image1/image2`，BP action 关闭，BP state 开启。

### 预期与观察重点

这个方案是合理的：9 个不适合跨轨迹 BP 的任务用于 warmup，不进入最终测评；Stage2 避免 randomized 语言描述唯一带来的伪随机问题，并通过 `avoid` 会在同 repo 内严格排除当前轨迹，让 BP 与当前轨迹不一致但任务基本一致。

