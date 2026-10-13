# BPV3 Plan Configs
以配置文件为准，并需要时时修正路径和具体数值
```json
{
  "batch_size": 64,
  "gradient_accumulation_steps": 1,
  "stats": {
    "stage0": {
      "episodes": 2500,
      "frames": 549787,
      "steps": 25772
    },
    "stage1": {
      "episodes": 27500,
      "frames": 6075103,
      "steps": 33224
    },
    "stage15": {
      "episodes": 22550,
      "frames": 5248031,
      "steps": 32801
    },
    "stage2": {
      "episodes": 22550,
      "frames": 5248031,
      "steps": 20501
    }
  },
  "repo_files": {
    "robotwin_50_clean_local.txt": "/vla/workspace/my_tbot/configs/ds_ids/pro6k/bpv3_plan/robotwin_50_clean_local.txt",
    "robotwin_50_clean_rand_local.txt": "/vla/workspace/my_tbot/configs/ds_ids/pro6k/bpv3_plan/robotwin_50_clean_rand_local.txt",
    "task_bp_list_41_clean_local.txt": "/vla/workspace/my_tbot/configs/ds_ids/pro6k/bpv3_plan/task_bp_list_41_clean_local.txt",
    "task_bp_list_41_clean_rand_local.txt": "/vla/workspace/my_tbot/configs/ds_ids/pro6k/bpv3_plan/task_bp_list_41_clean_rand_local.txt"
  },
  "configs": [
    "/vla/workspace/my_tbot/configs/Pro6k/bpv3_plan/stage0_q35_clean_3epoch.jsonc",
    "/vla/workspace/my_tbot/configs/Pro6k/bpv3_plan/stage15_bpv3_avoid_clean_rand_0p4epoch.jsonc",
    "/vla/workspace/my_tbot/configs/Pro6k/bpv3_plan/stage1_bpv3_origin_0p35epoch.jsonc",
    "/vla/workspace/my_tbot/configs/Pro6k/bpv3_plan/stage2_bpv3_align_clean_rand_0p25epoch.jsonc"
  ],
  "note": "stage1 updated to all 50 Robotwin tasks clean+randomized_500 per user request; stage1.5/stage2 remain TASK_BP_LIST 41 clean+rand."
}
```
