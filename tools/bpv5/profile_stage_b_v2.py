#!/usr/bin/env python
"""Profile the real four-rank Stage B v2 trainer; no alternate training loop."""
from contextlib import ExitStack
import inspect
import json
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/bpv5/stage_b/v2/test.jsonc"


def main():
    args = sys.argv[1:]
    if args and args not in ([f"--config_path={CONFIG}"], [
        "--config_path=configs/bpv5/stage_b/v2/test.jsonc"
    ]):
        raise ValueError("Only configs/bpv5/stage_b/v2/test.jsonc is accepted; no overrides")
    output = Path(os.environ["BPV5_PROFILE_DIR"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rank = int(os.environ.get("RANK", "0"))
    rank_dir = output / f"rank_{rank:02d}"
    rank_dir.mkdir(exist_ok=False)  # Refuse to mix two runs.
    controller = None
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) != 4:
            raise ValueError("This profile requires WORLD_SIZE=4")
        if not CONFIG.is_file():
            raise FileNotFoundError(CONFIG)
        sys.path.insert(0, str(ROOT / "src"))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import torch
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.scripts import lerobot_train as trainer
        from profile_stage_b_v2_utils import ProfileController
        from profile_stage_b_v2_e2e import aggregate_windows, measure_training_loop

        trainer.register_third_party_plugins()
        sys.argv = [sys.argv[0], f"--config_path={CONFIG}"]
        cfg = TrainPipelineConfig.from_pretrained(str(CONFIG), cli_args=[])
        required = {
            "batch_size": (cfg.batch_size, 2),
            "gradient_accumulation_steps": (cfg.gradient_accumulation_steps, 1),
            "steps": (cfg.steps, 40),
            "policy.type": (cfg.policy.type, "bpv5_stageb_v2"),
            "policy.gradient_checkpointing": (cfg.policy.gradient_checkpointing, False),
        }
        for name, (actual, expected) in required.items():
            if actual != expected:
                raise ValueError(f"test.jsonc: {name} must be {expected!r}, got {actual!r}")
        if cfg.resume:
            raise ValueError("resume must be false (profile step numbering starts at one)")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        # Housekeeping only; preserve the configured optimizer, data and DDP path.
        cfg.save_checkpoint = False
        cfg.eval_freq = 0
        cfg.wandb.enable = False
        cfg.policy.push_to_hub = False
        cfg.output_dir = output / "training"
        if hasattr(cfg.dataset, "heldout_monitor"):
            cfg.dataset.heldout_monitor = False
        controller = ProfileController(rank_dir, cfg)
        original = trainer.update_policy
        signature = inspect.signature(original)

        def measured_update(*a, **kw):
            bound = signature.bind(*a, **kw)
            bound.apply_defaults()
            return controller.update(original, a, kw, bound.arguments)

        with ExitStack() as instrumentation:
            instrumentation.enter_context(patch.object(trainer, "update_policy", measured_update))
            instrumentation.enter_context(measure_training_loop(trainer, controller))
            trainer.train(cfg)
        controller.finish()
        aggregate_windows(output, rank)
    except BaseException as exc:
        diagnostic = {
            "status": "oom" if "out of memory" in str(exc).lower() else "error",
            "rank": rank, "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
            "last_step": controller.step if controller else 0,
        }
        # No collectives, synchronization, empty_cache or CUDA copies on failure.
        try:
            if controller:
                diagnostic["partial_record"] = controller.current
                diagnostic["memory"] = controller.memory()
                diagnostic["whole_step_peak"] = controller.peak()
            (rank_dir / "failure.json").write_text(
                json.dumps(diagnostic, indent=2, allow_nan=False), encoding="utf-8"
            )
        except Exception as report_error:
            (rank_dir / "failure_minimal.txt").write_text(
                f"{diagnostic['traceback']}\nReporting failed: {report_error!r}", encoding="utf-8"
            )
        raise


if __name__ == "__main__":
    main()
