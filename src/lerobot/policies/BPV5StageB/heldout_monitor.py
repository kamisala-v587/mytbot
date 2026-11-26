"""Stage B HOM on the val split.

matched: M1 partner (L0 same leaf / L1 same leaf + exact task text)
blank:   matched demo with Z_demo zeroed
alt:     another M1 partner from the same bucket, when one exists
wrong:   different task, same source group; prefer same robot_type
         (RoboTwin clean/random of the same task name are NOT wrong)

Language:
  Training-time HOM historically used mask_text=False (full task text).
  For models trained with language, that confounds demo-visual identity:
  use pad_language=True to zero demo+current text attention so ratios isolate Z_demo.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.common import paired_bootstrap_ratio, to_device
from lerobot.policies.BPV5QFormer.dataset import _init_timer, _log_init_timing, collate_bpv5
from lerobot.policies.BPV5QFormer.heldout_monitor import (
    log_line,
    masked_error_dims,
    trend_advice,
    write_g9_json,
)
from lerobot.policies.BPV5StageB.pairing import choose_partner, choose_wrong_partner
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION


class StageBHeldoutMonitor:
    def __init__(self, val_ds, *, pad_language: bool = False) -> None:
        self.ds = val_ds
        self.pad_language = bool(pad_language)
        cfg = val_ds.cfg
        started = _init_timer(cfg)
        n = len(val_ds)
        rng = np.random.default_rng(0)
        take = min(int(cfg.heldout_monitor_max_episodes), n)
        self.subset = [int(i) for i in rng.choice(n, size=take, replace=False)] if take else []
        self.batch = int(cfg.heldout_monitor_batch)
        self.starts = {
            i: val_ds.segment_starts(int(val_ds.items[i][2]), mode="fixed") for i in self.subset
        }
        self.matched: dict[int, tuple[str, int]] = {}
        self.alt: dict[int, tuple[str, int]] = {}
        self.wrong: dict[int, tuple[str, int]] = {}
        for i in self.subset:
            rel, ep, _ = val_ds.items[i]
            pool = val_ds.partners[(rel, int(ep))]
            # Preserve the legacy list copy, but never materialize a shared bucket view.
            if isinstance(pool, list):
                pool = list(pool)
            matched_ep = choose_partner(pool, int(ep), i)
            self.matched[i] = (rel, matched_ep)
            if cfg.heldout_monitor_alternate and len(pool) >= 2:
                alt_ep = choose_partner(pool, int(ep), i + 1)
                if alt_ep == matched_ep:
                    alt_ep = choose_partner(pool, int(ep), i + 2)
                if alt_ep != matched_ep:
                    self.alt[i] = (rel, alt_ep)
            if cfg.heldout_monitor_wrong:
                picked = choose_wrong_partner(
                    rel,
                    int(ep),
                    val_ds._pool_eps,
                    index=i,
                    tier_overrides=getattr(cfg, "pairing_tiers", None),
                )
                if picked is not None:
                    self.wrong[i] = picked
        logging.info(
            "Stage B HOM: %d val episodes (M1), alt=%d wrong=%d pad_language=%s "
            "(wrong=same-source prefer same-robot, different task)",
            len(self.subset),
            len(self.alt),
            len(self.wrong),
            self.pad_language,
        )
        _log_init_timing(cfg, "stage_b_heldout_monitor_init", started)

    def _errors(self, policy, idx: list[int], *, partner: dict[int, tuple[str, int]] | None, blank: bool):
        model = policy.model
        dim = int(policy.config.max_action_dim)
        horizon = int(policy.config.chunk_size)
        m = max(len(self.starts[i]) for i in idx)
        items = []
        for i in idx:
            starts = self.starts[i]
            if len(starts) < m:
                starts = starts + [starts[-1]] * (m - len(starts))
            rel, ep = partner[i] if partner is not None else self.matched[i]
            items.append(
                self.ds.assemble(
                    i,
                    ep,
                    starts,
                    demo_rel=rel,
                    mask_text=False,
                    pad_language=self.pad_language,
                )
            )
        batch = to_device(collate_bpv5(items))
        noise = torch.stack(
            [torch.randn((m, horizon, dim), generator=torch.Generator().manual_seed(i)) for i in idx]
        ).reshape(len(idx) * m, horizon, dim).cuda()
        blank_flag = torch.ones(len(idx), dtype=torch.bool, device=noise.device) if blank else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_segments(batch, noise=noise, blank_demo=blank_flag).float()
        target = batch[ACTION].reshape(len(idx) * m, horizon, -1)
        step_ok = batch[ACTION_VALID_MASK].reshape(len(idx) * m, -1).bool()
        dim_mask = batch["action.dim_mask"]
        if dim_mask.ndim == 2:
            dim_mask = dim_mask.repeat_interleave(m, dim=0)
        err = masked_error_dims(pred, target, step_ok, dim_mask).reshape(len(idx), m)
        keys = [(i, k) for i in idx for k in range(len(self.starts[i]))]
        flat = [float(err[b, k]) for b, i in enumerate(idx) for k in range(len(self.starts[i]))]
        return np.asarray(flat, dtype=np.float64), keys

    def __call__(self, policy, accelerator=None, step: int | None = None) -> dict:
        from contextlib import nullcontext

        is_fsdp = False
        if accelerator is not None:
            distributed_type = getattr(accelerator, "distributed_type", None)
            try:
                from accelerate.state import DistributedType

                is_fsdp = distributed_type == DistributedType.FSDP
            except Exception:
                is_fsdp = str(distributed_type).endswith("FSDP")

        # DDP / single-process: main-only eval is fine.
        # FSDP: all ranks must enter so collectives / summon_full_params stay matched.
        if accelerator is not None and not accelerator.is_main_process and not is_fsdp:
            return {}

        import time

        t0 = time.time()
        was = policy.training
        policy.eval()

        summon_ctx = nullcontext()
        if is_fsdp:
            try:
                from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

                root = policy
                # Walk one level if Accelerate wrapped a container; prefer the FSDP module.
                if not isinstance(root, FSDP):
                    for child in policy.modules():
                        if isinstance(child, FSDP):
                            root = child
                            break
                if isinstance(root, FSDP):
                    summon_ctx = FSDP.summon_full_params(root, writeback=False)
            except Exception as exc:
                logging.warning("FSDP HOM summon_full_params unavailable (%s); ranks still barrier", exc)

        matched, blank, alt, wrong = {}, {}, {}, {}
        with summon_ctx:
            for s in range(0, len(self.subset), self.batch):
                idx = self.subset[s : s + self.batch]
                for store, partner, is_blank in (
                    (matched, None, False),
                    (blank, None, True),
                    (alt, self.alt, False),
                    (wrong, self.wrong, False),
                ):
                    group = [i for i in idx if partner is None or i in partner]
                    if not group:
                        continue
                    # Non-main FSDP ranks run the same forwards (same seeds/data) so
                    # all-gather counts match; only main keeps metrics below.
                    values, keys = self._errors(
                        policy,
                        group,
                        partner=None if partner is None else {i: partner[i] for i in group},
                        blank=is_blank,
                    )
                    if accelerator is None or accelerator.is_main_process:
                        for key, value in zip(keys, values):
                            store[key] = float(value)
        if was:
            policy.train()
        if accelerator is not None and is_fsdp:
            accelerator.wait_for_everyone()
        if accelerator is not None and not accelerator.is_main_process:
            return {}

        keys = sorted(matched)
        m = np.array([matched[k] for k in keys], dtype=np.float64)
        b = np.array([blank[k] for k in keys], dtype=np.float64)
        groups = np.array([i for i, _ in keys], dtype=np.int64)

        def side(store):
            sel = [k for k in keys if k in store]
            if len(sel) < 2:
                return None, None
            num = np.array([store[k] for k in sel])
            den = np.array([matched[k] for k in sel])
            g = np.array([i for i, _ in sel])
            return float(num.mean()), paired_bootstrap_ratio(num, den, g)

        alt_mean, alt_ratio = side(alt)
        wrong_mean, wrong_ratio = side(wrong)
        return {
            "step": step,
            "episodes": len(self.subset),
            "segments": int(len(m)),
            "seconds": round(time.time() - t0, 1),
            "pad_language": self.pad_language,
            "wrong_rule": "same_source_prefer_robot_diff_task",
            "mse_matched": float(m.mean()) if len(m) else None,
            "mse_blank": float(b.mean()) if len(b) else None,
            "mse_alternate": alt_mean,
            "mse_wrong": wrong_mean,
            "blank_over_match": paired_bootstrap_ratio(b, m, groups) if len(m) >= 2 else None,
            "alt_over_match": alt_ratio,
            "wrong_over_match": wrong_ratio,
        }


def main() -> None:
    """Offline Stage B HOM. Example:

    python -m lerobot.policies.BPV5StageB.heldout_monitor \\
      --run .../pb3/2026-11-20_10-00 --steps 10000 --pad_language \\
      --out_name heldout_monitor_nolang.jsonl
    """
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV5StageB.dataset import make_bpv5_stageb_dataset
    from lerobot.policies.BPV5StageB.modeling_bpv5stageb import BPV5StageBPolicy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with checkpoints/<step>/pretrained_model")
    ap.add_argument("--max_episodes", type=int, default=16)
    ap.add_argument("--out_name", default="heldout_monitor_nolang.jsonl")
    ap.add_argument("--steps", default="", help="comma-separated steps (default: every saved step)")
    ap.add_argument("--pad_language", action="store_true", help="zero demo+current text attention")
    ap.add_argument("--keep_language", action="store_true", help="full task text (legacy HOM)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.pad_language and args.keep_language:
        raise SystemExit("pass only one of --pad_language / --keep_language")
    pad = bool(args.pad_language) or not bool(args.keep_language)
    # default: pad_language=True for offline identity probe unless --keep_language
    if not args.pad_language and not args.keep_language:
        pad = True

    run = Path(args.run)
    ckpts = sorted(p for p in (run / "checkpoints").glob("*/pretrained_model") if p.parent.name.isdigit())
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(p.parent.name) in wanted]
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run}")

    cfg = TrainPipelineConfig.from_pretrained(str(ckpts[-1] / "train_config.json"))
    cfg.dataset.heldout_monitor = True
    cfg.dataset.heldout_monitor_max_episodes = args.max_episodes
    cfg.dataset.heldout_monitor_alternate = True
    cfg.dataset.heldout_monitor_wrong = True
    train_ds, _ = make_bpv5_stageb_dataset(cfg)
    monitor = StageBHeldoutMonitor(train_ds.derive(split="val", segment_mode="fixed"), pad_language=pad)

    out = run / args.out_name
    if args.overwrite and out.is_file():
        out.unlink()
    history = [json.loads(line) for line in out.read_text().splitlines()] if out.is_file() else []
    with out.open("a", encoding="utf-8") as f:
        for ck in ckpts:
            step = int(ck.parent.name)
            policy = BPV5StageBPolicy.from_pretrained(str(ck), config=cfg.policy).cuda().eval()
            m = monitor(policy, None, step)
            decay = int(getattr(cfg.policy, "scheduler_decay_steps", 0) or 0)
            m["advice"] = trend_advice(history, m, lr_at_floor=bool(decay and step >= decay))
            m["source"] = "offline"
            m["pad_language"] = pad
            history.append(m)
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
            write_g9_json(run, m, step, ckpt=str(ck.resolve()))
            print(step, log_line(m), f"pad_language={pad}", flush=True)
            del policy
            torch.cuda.empty_cache()
    print(out)


__all__ = ["StageBHeldoutMonitor", "log_line", "trend_advice", "write_g9_json", "main"]


if __name__ == "__main__":
    main()
