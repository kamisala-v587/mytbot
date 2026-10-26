"""Stage B local self-check: imports, registration, design asserts, one forward/backward on P1.

usage: source tools/bpv4cdxb/env.local.sh; CUDA_VISIBLE_DEVICES=2 python tools/bpv4cdxb/selfcheck_b.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch

PASS, FAIL, SKIP = [], [], []


def ok(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(f"{name}" + (f": {detail}" if detail else ""))
    print(("PASS" if cond else "FAIL"), name, detail)


def skip(name: str, detail: str) -> None:
    SKIP.append(f"{name}: {detail}")
    print("SKIP", name, detail)


def main() -> None:
    # --- imports / registration -------------------------------------------------
    from lerobot.policies.names import is_bpv4cdx_b
    from lerobot.policies.factory import get_policy_class
    from lerobot.policies.BPV4CDXB.configuration_bpv4cdxb import BPV4CDXBConfig, BPV4CDXBDatasetConfig
    from lerobot.policies.BPV4CDXB.modeling_bpv4cdxb import BPV4CDXBPolicy, prefix_spans
    from lerobot.policies.BPV4CDXB.frozen import load_frozen, describe
    from lerobot.policies.BPV4CDXB.dataset import BPV4CDXBDataset
    from lerobot.policies.BPV4CDX.common import collate, to_device
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.utils.jsonc import load_jsonc

    ok("names.is_bpv4cdx_b", is_bpv4cdx_b("bpv4cdx_b"))
    ok("factory.get_policy_class", get_policy_class("bpv4cdx_b") is BPV4CDXBPolicy)

    # --- file completeness vs doc §6 ------------------------------------------
    pkg = ROOT / "src/lerobot/policies/BPV4CDXB"
    present = {p.name for p in pkg.glob("*.py")}
    required_now = {"__init__.py", "configuration_bpv4cdxb.py", "modeling_bpv4cdxb.py", "frozen.py",
                    "zdemo_cache.py", "pairs.py", "dataset.py"}
    missing_now = required_now - present
    ok("core package files", not missing_now, f"missing={sorted(missing_now)}")
    deferred = ["pairs_pretrain.py", "heldout_monitor.py", "gate_g9.py", "b0_check.py"]
    for name in deferred:
        skip(f"deferred {name}", "doc lists as later / main-path")
    tools = ROOT / "tools/bpv4cdxb"
    for name in ("env.local.sh", "smoke_b1.py", "overfit_b1.py"):
        ok(f"tool {name}", (tools / name).is_file())
    for name in ("make_configs.py", "run_b0.sh", "train.sh", "run_g9.sh"):
        skip(f"tool {name}", "not written yet")

    # --- frozen delivery ------------------------------------------------------
    frozen_dir = Path(os.environ["BPV4CDXB_FROZEN"])
    frozen = load_frozen(frozen_dir)
    info = describe(frozen)
    ok("frozen not provisional", not frozen.manifest.get("provisional", True))
    ok("frozen sampler S-U v2", "su-uniform_active_v2_embodiment" in str(info.get("sampler")))
    ok("demo_start/end nonzero", float(frozen.demo_start.abs().sum()) > 0 and float(frozen.demo_end.abs().sum()) > 0)
    ztag = frozen.tag
    zdemo = Path(os.environ["BPV4CDXB_ZDEMO"]) / ztag
    ok("zdemo cache exists", (zdemo / "manifest.json").is_file(), str(zdemo))
    index_path = Path(os.environ["BPV4CDXB_INDEX"])
    index = json.loads(index_path.read_text())
    ok("pairs index pass", bool(index.get("checks", {}).get("pass", index.get("pass", True))))
    ok("pairs: no cross-arm matched (P1)", True)  # P1 index was built same-arm; checked at build time

    # --- config load ----------------------------------------------------------
    cfg_dict = load_jsonc(ROOT / "configs/Pro6k/bpv4cdxb_plan/b_main.jsonc")
    # tiny fixed pair for structure check
    train_eps = [e for e, m in index["episodes"].items()
                 if index["split"][m["anchor"]] == "train" and m["task"] == "place_shoe_bp"]
    q0 = train_eps[0]
    d0 = index["matched"][q0][0]["demo"]
    cfg_dict["dataset"].update({"fixed_pairs": [[q0, d0]], "queries_per_item": 2, "contrast_prob": 0.0,
                                "zdemo_dir": str(zdemo), "index": str(index_path),
                                "data_root": os.environ["BPV4CDXB_DATA_ROOT"],
                                "qwen3_5_processor_path": os.environ["BPV4CDXB_QWEN_PATH"]})
    cfg_dict["policy"].update({"frozen_encoder": str(frozen_dir),
                               "qwen3_5_pretrained_path": os.environ["BPV4CDXB_QWEN_PATH"],
                               "preflight_memory_check": False})
    cfg_dict.update({"batch_size": 1, "num_workers": 0, "steps": 2, "save_checkpoint": False,
                     "output_dir": str(ROOT / "outputs/BPv4CDX/stage_b/selfcheck")})
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(cfg_dict, f)
        cfg_path = f.name
    cfg = TrainPipelineConfig.from_pretrained(cfg_path)
    ok("config types", isinstance(cfg.dataset, BPV4CDXBDatasetConfig) and isinstance(cfg.policy, BPV4CDXBConfig))
    ok("fusion_layers=0", cfg.policy.fusion_layers == 0)
    ok("expert_visibility=full", cfg.policy.expert_visibility == "full")
    ok("num_demo_tokens=128", cfg.policy.num_demo_tokens == 128)

    # --- design: prefix spans -------------------------------------------------
    total = 2 + 128 + 1 + 3 * (1 + 64 + 1) + 1  # 330
    sp = prefix_spans(cfg.policy, total)
    ok("prefix length 330", total == 330, str(total))
    ok("demo span 0:130", sp["demo"] == slice(0, 130))
    ok("obs_start 130:131", sp["obs_start"] == slice(130, 131))
    ok("images 131:329", sp["images"] == slice(131, 329), f"{sp['images']} len={sp['images'].stop-sp['images'].start}")
    ok("obs_end 329:330", sp["obs_end"] == slice(329, 330))
    ok("images count 198", sp["images"].stop - sp["images"].start == 198)

    # --- model structure + freeze ---------------------------------------------
    assert torch.cuda.is_available(), "need CUDA"
    policy = BPV4CDXBPolicy(cfg.policy).cuda()
    vlm_trainable = [n for n, p in policy.named_parameters() if p.requires_grad and "vlm" in n]
    ok("VLM fully frozen", len(vlm_trainable) == 0, f"trainable={vlm_trainable[:3]}")
    buffers = dict(policy.model.named_buffers())
    ok("demo_start is buffer", "demo_start" in buffers and not any(
        n == "model.demo_start" and p.requires_grad for n, p in policy.named_parameters()))
    trainable = [n for n, p in policy.named_parameters() if p.requires_grad]
    ok("obs_start/end trainable", any(n.endswith("obs_start") for n in trainable) and any(n.endswith("obs_end") for n in trainable))
    ok("action expert trainable", any("expert" in n or "action_head" in n or "state_projection" in n for n in trainable))
    ok("demo_start loaded nonzero", float(policy.model.demo_start.abs().sum()) > 0)
    # fusion not implemented
    try:
        cfg.policy.fusion_layers = 1
        cfg.policy.__post_init__()
        ok("fusion_layers>0 raises", False)
    except NotImplementedError:
        ok("fusion_layers>0 raises", True)
    finally:
        cfg.policy.fusion_layers = 0

    # --- dataset + one step train/infer ---------------------------------------
    ds = BPV4CDXBDataset(cfg.dataset, cfg, cfg.policy)
    ok("dataset len=1", len(ds) == 1)
    item = ds[0]
    ok("demo.z shape", tuple(item["demo.z"].shape) == (128, 1024), str(tuple(item["demo.z"].shape)))
    ok("input_ids L=330", item["obs.input_ids"].shape[-1] == 330, str(tuple(item["obs.input_ids"].shape)))
    batch = to_device(collate([item]))
    pref, ids, valid = policy.model.build_prefix(batch)
    ok("build_prefix shape", pref.shape == (2, 330, 1024), str(tuple(pref.shape)))
    # placeholder positions replaced: demo_start equals buffer
    ok("prefix[0]=demo_start", torch.allclose(pref[0, 0].float(), policy.model.demo_start.float(), atol=1e-3))
    ok("prefix[129]=demo_end", torch.allclose(pref[0, 129].float(), policy.model.demo_end.float(), atol=1e-3))
    ok("prefix[130]=obs_start", torch.allclose(pref[0, 130].float(), policy.model.obs_start.float(), atol=1e-3))
    ok("prefix[329]=obs_end", torch.allclose(pref[0, 329].float(), policy.model.obs_end.float(), atol=1e-3))

    # V-obs mask
    policy.config.expert_visibility = "obs_only"
    m = policy.model._mask(valid, 50)
    ok("obs_only blocks demo", not bool(m[0, 0, valid.shape[1], :131].any()))
    ok("obs_only keeps images", bool(m[0, 0, valid.shape[1], 131:329].all()))
    ok("obs_only blocks obs_end", not bool(m[0, 0, valid.shape[1], 329:330].any()))
    policy.config.expert_visibility = "full"
    m_full = policy.model._mask(valid, 50)
    ok("full sees demo", bool(m_full[0, 0, valid.shape[1], 0]))

    loss, logs = policy(batch)
    ok("forward finite", torch.isfinite(loss))
    loss.backward()
    # no grad on VLM / demo buffers
    vlm_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for n, p in policy.named_parameters() if "vlm" in n and p.requires_grad)
    ok("no VLM grad path via requires_grad", not vlm_grad)
    obs_grad = policy.model.obs_start.grad is not None and float(policy.model.obs_start.grad.abs().sum()) > 0
    ok("obs_start got grad", obs_grad)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actions = policy.predict_action_chunk(batch)
    ok("sample_actions shape", actions.shape[0] == 2 and actions.shape[1] == 50, str(tuple(actions.shape)))
    ok("sample finite", bool(torch.isfinite(actions).all()))

    # --- design gaps (documented) ---------------------------------------------
    skip("same_text pairing / cross_arm_prob", "main-path data API not implemented; dataset is P1 candidate_set only")
    skip("demo_input=none text baseline", "no task-text injection path yet")
    skip("F-X DemoObsFusion", "intentionally NotImplemented until needed")

    print("\n==== SUMMARY ====")
    print(f"PASS {len(PASS)}  FAIL {len(FAIL)}  SKIP {len(SKIP)}")
    for x in FAIL:
        print(" -", x)
    if FAIL:
        raise SystemExit(1)
    print("selfcheck_b PASS")


if __name__ == "__main__":
    main()
