"""BPv5 Stage A held-out monitor (HOM): generation MSE on held-out trajectories at every checkpoint save.

Conditions (demo-swap protocol; alt optional / deferred):
  matched   own demo, canonical task text
  blank     matched batch with Z_demo zeroed
  wrong     a held-out demo of a different repo (prefer same source)
  alt       (optional) L0 same-repo demo i≠j (RoboTwin etc.); drawn from train_ds when
            val holdout has only one ep/repo so in-pool alt would be empty

Training loss is flow-matching velocity error; HOM reports the masked MSE of the actions the model actually
samples (fixed noise per episode). Results: <output_dir>/heldout_monitor.jsonl and optional
<output_dir>/g9/a_<step>.json (notebook-friendly mean_error / ratios).

In training: dataset.make_heldout_monitor() once; lerobot_train calls it after every checkpoint save (main process).
Offline: python -m lerobot.policies.BPV5QFormer.heldout_monitor --run <output_dir>/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4CDX.common import paired_bootstrap_ratio, to_device
from lerobot.policies.BPV4CDX.heldout_monitor import _subset, trend_advice
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from . import data80d as D
from .dataset import collate_bpv5
from .modeling_bpv5qformer import ACTION_DIM_MASK

CONDITIONS = ("matched", "blank", "alt", "wrong")


def _traj(actions: np.ndarray, points: int = 64) -> np.ndarray:
    t = np.linspace(0, max(len(actions) - 1, 1), points)
    return np.stack(
        [np.interp(t, np.arange(len(actions)), actions[:, d]) for d in range(actions.shape[1])], 1
    ).reshape(-1)


def masked_error_dims(
    pred: torch.Tensor, target: torch.Tensor, step_valid: torch.Tensor, dim_mask: torch.Tensor
) -> torch.Tensor:
    """Per-segment MSE over valid steps and real action dims: [N,H,A] -> [N]."""
    if dim_mask.ndim == 1:
        dim_mask = dim_mask[None, :].expand(pred.shape[0], -1)
    a = target.shape[-1]
    w = step_valid[..., None].float() * dim_mask[:, None, :a].float()
    return ((pred[..., :a].float() - target.float()) ** 2 * w).sum((1, 2)) / w.sum((1, 2)).clamp_min(1)


def log_line(m: dict) -> str:
    parts = [f"E_matched={m['mse_matched']:.4f}" if m.get("mse_matched") is not None else "E_matched=-"]
    mse_key = {"blank": "mse_blank", "alt": "mse_alternate", "wrong": "mse_wrong"}
    for c in ("blank", "alt", "wrong"):
        v = m.get(f"{c}_over_match")
        if isinstance(v, dict) and v.get("ratio") is not None:
            lo, hi = (v.get("ci95") or [None, None])[:2]
            if lo is not None and hi is not None:
                parts.append(f"{c}/matched={v['ratio']:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                parts.append(f"{c}/matched={v['ratio']:.3f}")
        elif m.get(mse_key[c]) is not None:
            parts.append(f"{mse_key[c]}={m[mse_key[c]]:.4f}")
    parts.append(f"episodes={m.get('episodes')} segments={m.get('segments')}")
    if m.get("advice"):
        a = m["advice"]
        parts.append(f"-> {a.get('recommendation')} ({a.get('reason')})")
    return " ".join(parts)


def g9_payload(m: dict, *, ckpt: str | None = None) -> dict:
    """Notebook-compatible dump (stage_b_compare / Stage A monitor)."""
    mean_error = {
        "matched": m.get("mse_matched"),
        "blank": m.get("mse_blank"),
        "alt": m.get("mse_alternate"),
        "wrong": m.get("mse_wrong"),
    }
    ratios = {
        "blank": m.get("blank_over_match"),
        "alt": m.get("alt_over_match"),
        "wrong": m.get("wrong_over_match"),
    }
    return {
        "step": m.get("step"),
        "queries": m.get("episodes"),
        "segments": m.get("segments"),
        "ckpt": ckpt,
        "mean_error": mean_error,
        "ratios": ratios,
        "mse_matched": m.get("mse_matched"),
        "mse_blank": m.get("mse_blank"),
        "mse_alternate": m.get("mse_alternate"),
        "mse_wrong": m.get("mse_wrong"),
        "blank_over_match": m.get("blank_over_match"),
        "alt_over_match": m.get("alt_over_match"),
        "wrong_over_match": m.get("wrong_over_match"),
        "by_source": m.get("by_source"),
        "seconds": m.get("seconds"),
    }


class HeldoutMonitor:
    """Main-process evaluation on a fixed, task-stratified held-out subset."""

    def __init__(self, train_ds) -> None:
        cfg = train_ds.cfg
        self.cfg = cfg
        self.train_ds = train_ds
        # prefer explicit val split; if empty (no holdout), fall back to a fixed subset of train
        # reuse train_ds repo index so HOM does not scan the data root again
        if int(cfg.val_holdout_last_per_repo) > 0:
            self.ds = train_ds.derive(split="val", segment_mode="fixed", max_trajectories=0)
        else:
            self.ds = train_ds.derive(split="train", segment_mode="fixed", max_trajectories=0)
        n = len(self.ds)
        # Stage A bucket = repo id (same task / embodiment); source = top-level dataset family
        repos = [rel for rel, _, _ in self.ds.items]
        sources = [rel.split("/")[0] for rel in repos]
        rng = np.random.default_rng(0)
        self.subset = _subset(repos, int(cfg.heldout_monitor_max_episodes), rng)
        self.batch = int(cfg.heldout_monitor_batch)
        seg = int(cfg.segment_len)
        self.starts = {
            i: list(range(0, max(1, self.ds.items[i][2]), seg)) or [0] for i in self.subset
        }
        self.task = {i: repos[i] for i in range(n)}  # repo-level task bucket (L0)
        self.source = {i: sources[i] for i in range(n)}
        self.partner_alt: dict[int, int] = {}
        self.partner_wrong: dict[int, int] = {}
        # Query indices whose alt demo index refers to train_ds (L0 fallback).
        self.partner_alt_from_train: set[int] = set()
        if cfg.heldout_monitor_alternate or cfg.heldout_monitor_wrong:
            by_task: dict[str, list[int]] = defaultdict(list)
            by_source: dict[str, list[int]] = defaultdict(list)
            for i in range(n):
                by_task[self.task[i]].append(i)
                by_source[self.source[i]].append(i)
            traj: dict[tuple[str, int], np.ndarray] = {}

            def _load_traj_rel_ep(rel: str, ep: int) -> np.ndarray:
                key = (rel, int(ep))
                if key not in traj:
                    ep_meta = D.episodes(rel)[ep]
                    # frames loader is shared via either dataset
                    frames = self.train_ds._frames
                    _, action = frames.episode(rel, ep_meta)
                    traj[key] = _traj(action.astype(np.float32))
                return traj[key]

            # L0 alt pool: all train episodes of the same repo (exclude query ep)
            train_by_repo: dict[str, list[int]] = defaultdict(list)
            for j, (rel, ep, _) in enumerate(self.train_ds.items):
                train_by_repo[rel].append(j)

            for a in self.subset:
                rel_a, ep_a, _ = self.ds.items[a]
                if cfg.heldout_monitor_alternate:
                    # Prefer in-val same-repo others; else L0 from train_ds.
                    in_val = [b for b in by_task[self.task[a]] if b != a]
                    if in_val:
                        ta = _load_traj_rel_ep(rel_a, ep_a)
                        d = {}
                        for b in in_val:
                            rel_b, ep_b, _ = self.ds.items[b]
                            tb = _load_traj_rel_ep(rel_b, ep_b)
                            if tb.shape == ta.shape:
                                d[b] = float(np.linalg.norm(ta - tb))
                        if d:
                            med = float(np.median(list(d.values())))
                            far = [b for b, v in d.items() if v > med] or list(d)
                            self.partner_alt[a] = int(rng.choice(far))
                    else:
                        cands = [
                            j
                            for j in train_by_repo.get(rel_a, [])
                            if int(self.train_ds.items[j][1]) != int(ep_a)
                        ]
                        if cands:
                            ta = _load_traj_rel_ep(rel_a, ep_a)
                            d = {}
                            for j in cands:
                                _, ep_b, _ = self.train_ds.items[j]
                                tb = _load_traj_rel_ep(rel_a, ep_b)
                                if tb.shape == ta.shape:
                                    d[j] = float(np.linalg.norm(ta - tb))
                            pool = d or {j: 0.0 for j in cands}
                            med = float(np.median(list(pool.values()))) if pool else 0.0
                            far = [j for j, v in pool.items() if v > med] or list(pool)
                            self.partner_alt[a] = int(rng.choice(far))
                            self.partner_alt_from_train.add(a)
                if cfg.heldout_monitor_wrong:
                    others = [b for b in by_source[self.source[a]] if self.task[b] != self.task[a]]
                    if not others:
                        others = [b for b in range(n) if self.task[b] != self.task[a]]
                    if others:
                        self.partner_wrong[a] = int(rng.choice(others))
        logging.info(
            "BPv5 HOM: %d held-out episodes (of %d), %d alt partners (%d via train L0), %d wrong partners, batch=%d",
            len(self.subset),
            n,
            len(self.partner_alt),
            len(self.partner_alt_from_train),
            len(self.partner_wrong),
            self.batch,
        )

    def _item(
        self,
        query_i: int,
        starts: list[int],
        demo_i: int | None = None,
        *,
        demo_ds=None,
    ) -> dict:
        rel, ep, length = self.ds.items[query_i]
        ep_meta = D.episodes(rel)[ep]
        state, action = self.ds._frames.episode(rel, ep_meta)
        m = len(starts)
        src = demo_ds if demo_ds is not None else self.ds
        demo = src.demo(demo_i if demo_i is not None else query_i)
        target = self.ds.target(
            query_i,
            starts,
            state=state,
            action=action,
            length=length,
            generic=[False] * m,
            dropped=[False] * m,
        )
        return {**demo, **target, **self.ds.meta(query_i, length)}

    @torch.no_grad()
    def _errors(
        self,
        policy,
        idx: list[int],
        *,
        demo_of: dict[int, int] | None = None,
        demo_ds=None,
        blank: bool = False,
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        model = policy.model if hasattr(policy, "model") else policy
        dim = int(policy.config.max_action_dim)
        horizon = int(policy.config.chunk_size)
        m = max(len(self.starts[i]) for i in idx)
        pad = lambda xs: xs + [xs[-1]] * (m - len(xs))  # noqa: E731
        items = []
        for i in idx:
            if demo_of is not None and i in demo_of:
                items.append(self._item(i, pad(self.starts[i]), demo_i=demo_of[i], demo_ds=demo_ds))
            else:
                items.append(self._item(i, pad(self.starts[i])))
        batch = to_device(collate_bpv5(items))
        noise = torch.stack(
            [torch.randn((m, horizon, dim), generator=torch.Generator().manual_seed(i)) for i in idx]
        )
        noise = noise.reshape(len(idx) * m, horizon, dim).cuda()
        blank_flag = torch.ones(len(idx), dtype=torch.bool, device=noise.device) if blank else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_segments(batch, noise=noise, blank_demo=blank_flag).float()
        target = batch[ACTION].reshape(len(idx) * m, horizon, -1)
        step_ok = batch[ACTION_VALID_MASK].reshape(len(idx) * m, -1).bool()
        dim_mask = batch[ACTION_DIM_MASK]
        if dim_mask.ndim == 2:
            dim_mask = dim_mask.repeat_interleave(m, dim=0)
        err = masked_error_dims(pred, target, step_ok, dim_mask).reshape(len(idx), m)
        keep = [(b, k) for b, i in enumerate(idx) for k in range(len(self.starts[i]))]
        return np.array([float(err[b, k]) for b, k in keep]), [(idx[b], k) for b, k in keep]

    def __call__(self, policy, accelerator=None, step: int | None = None) -> dict:
        if accelerator is not None and not accelerator.is_main_process:
            return {}
        import time

        t0 = time.time()
        was_training = policy.training
        policy.eval()
        matched_by_key: dict[tuple[int, int], float] = {}
        blank_by_key: dict[tuple[int, int], float] = {}
        alt_by_key: dict[tuple[int, int], float] = {}
        wrong_by_key: dict[tuple[int, int], float] = {}
        for s in range(0, len(self.subset), self.batch):
            idx = self.subset[s : s + self.batch]
            e_m, keys = self._errors(policy, idx)
            for k, v in zip(keys, e_m):
                matched_by_key[k] = float(v)
            e_b, keys_b = self._errors(policy, idx, blank=True)
            for k, v in zip(keys_b, e_b):
                blank_by_key[k] = float(v)
            with_alt = [i for i in idx if i in self.partner_alt]
            if with_alt:
                # Split val-pool vs train-L0 alt so demo indices stay on the right dataset.
                alt_val = [i for i in with_alt if i not in self.partner_alt_from_train]
                alt_tr = [i for i in with_alt if i in self.partner_alt_from_train]
                for group, ds in ((alt_val, self.ds), (alt_tr, self.train_ds)):
                    if not group:
                        continue
                    e_a, ka = self._errors(
                        policy,
                        group,
                        demo_of={i: self.partner_alt[i] for i in group},
                        demo_ds=ds,
                    )
                    for k, v in zip(ka, e_a):
                        alt_by_key[k] = float(v)
            with_wrong = [i for i in idx if i in self.partner_wrong]
            if with_wrong:
                e_w, kw = self._errors(policy, with_wrong, demo_of=self.partner_wrong)
                for k, v in zip(kw, e_w):
                    wrong_by_key[k] = float(v)
        if was_training:
            policy.train()

        keys = sorted(matched_by_key)
        matched = np.array([matched_by_key[k] for k in keys], dtype=np.float64)
        blank = np.array([blank_by_key[k] for k in keys], dtype=np.float64)
        groups = np.array([i for i, _ in keys], dtype=np.int64)
        alt_keys = [k for k in keys if k in alt_by_key]
        wrong_keys = [k for k in keys if k in wrong_by_key]
        alt = np.array([alt_by_key[k] for k in alt_keys], dtype=np.float64)
        wrong = np.array([wrong_by_key[k] for k in wrong_keys], dtype=np.float64)
        alt_den = np.array([matched_by_key[k] for k in alt_keys], dtype=np.float64)
        wrong_den = np.array([matched_by_key[k] for k in wrong_keys], dtype=np.float64)
        alt_groups = np.array([i for i, _ in alt_keys], dtype=np.int64)
        wrong_groups = np.array([i for i, _ in wrong_keys], dtype=np.int64)

        def ratio(num: np.ndarray, den: np.ndarray, g: np.ndarray) -> dict | None:
            if len(num) < 2 or len(den) < 2:
                return None
            return paired_bootstrap_ratio(num, den, g)

        out = {
            "step": step,
            "episodes": len(self.subset),
            "segments": int(len(matched)),
            "seconds": round(time.time() - t0, 1),
            "mse_matched": float(matched.mean()) if len(matched) else None,
            "mse_blank": float(blank.mean()) if len(blank) else None,
            "mse_alternate": float(alt.mean()) if len(alt) else None,
            "mse_wrong": float(wrong.mean()) if len(wrong) else None,
            "blank_over_match": ratio(blank, matched, groups),
            "alt_over_match": ratio(alt, alt_den, alt_groups) if len(alt) else None,
            "wrong_over_match": ratio(wrong, wrong_den, wrong_groups) if len(wrong) else None,
        }
        by_source: dict[str, dict] = {}
        for src in sorted({self.source[i] for i in self.subset}):
            sel = np.array([self.source[i] == src for i, _ in keys])
            if not sel.any():
                continue
            by_source[src] = {
                "segments": int(sel.sum()),
                "mse_matched": float(matched[sel].mean()),
                "blank_over_match": ratio(blank[sel], matched[sel], groups[sel]),
            }
        out["by_source"] = by_source
        return out


def write_g9_json(output_dir: Path, record: dict, step: int, ckpt: str | None = None) -> Path | None:
    if not record or record.get("mse_matched") is None:
        return None
    g9_dir = Path(output_dir) / "g9"
    g9_dir.mkdir(parents=True, exist_ok=True)
    path = g9_dir / f"a_{int(step):06d}.json"
    path.write_text(json.dumps(g9_payload(record, ckpt=ckpt), indent=2, ensure_ascii=False))
    return path


def main() -> None:
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.BPV5QFormer.modeling_bpv5qformer import BPV5QFormerPolicy

    from .dataset import make_bpv5_qformer_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with checkpoints/<step>/pretrained_model")
    ap.add_argument("--max_episodes", type=int, default=24)
    ap.add_argument("--out_name", default="heldout_monitor.jsonl")
    ap.add_argument("--steps", default="", help="comma-separated steps (default: every saved step)")
    ap.add_argument(
        "--alternate",
        action="store_true",
        help="enable L0 same-repo alt-demo (i≠j; uses train_ds when val has 1 ep/repo)",
    )
    ap.add_argument("--no_wrong", action="store_true", help="disable wrong-demo partners")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="replace out_name instead of append",
    )
    args = ap.parse_args()
    run = Path(args.run)
    ckpts = sorted(p for p in (run / "checkpoints").glob("*/pretrained_model") if p.parent.name.isdigit())
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(p.parent.name) in wanted]
    if not ckpts:
        raise SystemExit(f"no checkpoints under {run}")
    cfg = TrainPipelineConfig.from_pretrained(str(ckpts[-1] / "train_config.json"))
    cfg.dataset.heldout_monitor_max_episodes = args.max_episodes
    if args.alternate:
        cfg.dataset.heldout_monitor_alternate = True
    if args.no_wrong:
        cfg.dataset.heldout_monitor_wrong = False
    train_ds, _ = make_bpv5_qformer_dataset(cfg)
    monitor = HeldoutMonitor(train_ds)
    out = run / args.out_name
    if args.overwrite and out.is_file():
        out.unlink()
    history = [json.loads(line) for line in out.read_text().splitlines()] if out.is_file() else []
    with out.open("a", encoding="utf-8") as f:
        for ck in ckpts:
            step = int(ck.parent.name)
            policy = BPV5QFormerPolicy.from_pretrained(str(ck), config=cfg.policy).cuda().eval()
            m = monitor(policy, None, step)
            decay = int(getattr(cfg.policy, "scheduler_decay_steps", 0) or 0)
            m["advice"] = trend_advice(history, m, lr_at_floor=bool(decay and step >= decay))
            m["alt_pairing"] = "L0_same_repo"
            history.append(m)
            f.write(json.dumps({**m, "source": "offline"}) + "\n")
            write_g9_json(run, m, step, ckpt=str(ck.resolve()))
            print(step, log_line(m), flush=True)
            del policy
            torch.cuda.empty_cache()
    print(out)


if __name__ == "__main__":
    main()
