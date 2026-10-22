"""G4 (text shortcut), G5 (fine-grained behaviour), G6 (state shortcut, AE-S0 only), G7 (held-out, seeds, round-trip)
for Stage A-AE (design doc 7.0, 7.5 ~ 7.8).

Every target segment (episode q, fixed starts 0, 50, 100, ...) is predicted under several demo / text conditions with
the SAME flow noise (K noise draws per segment, --seeds), so errors are paired per segment:
  matched            demo = q, canonical text
  matched_generic    demo = q, generic text
  alternate          demo = r, same task, action distance d(q, r) above the within-task --alt_quantile of all pairs
                     (default: median); --n_alt partners per q (fixed seed), errors averaged over the partners
  alternate_generic  first alternate partner, generic text
  blank              demo features all zero
  shuffle            demo slots in a random order (slot times kept)
  wrong              most scene-similar demo of another task (first-slot head-camera ViT cosine)
  position           same task + same grasp arm, joint configuration farther apart than the group median
                     (--protocol v1: configuration at the first gripper close, group = (task, arm);
                      v2: configuration at the place = first reopen after it, group = (task, arm, clean / randomized))
  arm                same task, other grasp arm
  hard               v1: same task, scene within the top quarter of within-task scene similarity, action distance
                     above the task median (in practice mostly the other arm); v2: the same inside the
                     (task, arm, source) group, so the hard negative uses the same arm in the same kind of scene
  textonly           separate text-only model (--textonly_ckpt), canonical text
d(q, r): absolute actions resampled to 64 points on normalized time (the same normalized time windows), L2.
Metric: masked MSE of sampled actions (normalized delta space). Ratios E_cond / E_match with a trajectory-level
paired bootstrap 95% CI (2000 resamples), overall and per task.
G5 extras: Spearman(Z_demo distance, action distance) per task with a bootstrap CI over tasks; follow check
(with B's video the output is closer to B's segment at the same normalized time than to A's).
G6 (only if policy.use_segment_start_state): swap video (= alternate, same state), swap state (alternate partner's
state at the same normalized time), masked state (zeros), state only (blank video + state).
G7 round-trip: matched predictions of all segments -> absolute actions (segment-start state + inverse normalization),
stitched in time and compared with the true trajectory: per-dimension cross-correlation lag (time drift) and the
last-segment error vs the middle segments (tail misalignment). G7 seeds: --ckpt_seed2 evaluates a second training
seed on the same partners / noise; directions must agree.
Thresholds the design doc leaves open (marked "provisional" in the result): alternate distance quantile 0.5,
Z correlation CI low > 0, |median lag| and median |lag| <= 2 frames, tail ratio <= 1.2.

Usage: python -m lerobot.policies.BPV4CDX.gate_g4_g7 --ckpt <pretrained_model> [--textonly_ckpt <...>] \
         [--ckpt_seed2 <...>] --out <dir> --name <run> [--seeds 1] [--n_alt 1] [--split val] [--dataset.x=...]
Local paths: export BPV4CDX_DATA_ROOT / BPV4CDX_QWEN_PATH / BPV4CDX_CACHE_ROOT (+ BPV4CDX_CACHE_PREFIX) / BPV4CDX_REPO.
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict

import numpy as np
import torch

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

from .common import (ActionInverter, bootstrap_mean, collate, load_cfg, make_dataset, make_policy, masked_error,
                     paired_bootstrap_ratio, reproducibility, to_device, write_result)
from . import embodiment as E
from .ae3_data import source_of
from .heldout_monitor import _subset

THRESH = 1.10
PROVISIONAL = {"alt_quantile": 0.5, "z_corr_ci_low": 0.0, "max_abs_lag": 2.0, "tail_ratio": 1.2, "max_lag_search": 25}
PAIRED = ("alternate", "position", "arm", "hard", "wrong")


def traj_vector(x: np.ndarray, points: int = 64) -> np.ndarray:
    t = np.linspace(0, len(x) - 1, points)
    return np.stack([np.interp(t, np.arange(len(x)), x[:, d]) for d in range(x.shape[1])], axis=1).reshape(-1)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def best_lag(pred: np.ndarray, true: np.ndarray, max_lag: int) -> float | None:
    """Median over moving dims of the lag maximizing the correlation of pred(t + lag) with true(t)."""
    lags = []
    for d in range(true.shape[1]):
        y = true[:, d] - true[:, d].mean()
        if y.std() < 1e-3:
            continue
        x = pred[:, d] - pred[:, d].mean()
        best, best_c = 0, -np.inf
        for lag in range(-max_lag, max_lag + 1):
            a, b = (x[lag:], y[: len(y) - lag]) if lag >= 0 else (x[:lag], y[-lag:])
            if len(a) < 10:
                continue
            c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            if c > best_c:
                best, best_c = lag, c
        lags.append(best)
    return float(np.median(lags)) if lags else None


class Evaluator:
    def __init__(self, cfg, ckpt, textonly_ckpt, seeds: int, batch: int, split: str = "val", n_alt: int = 1,
                 alt_quantile: float = PROVISIONAL["alt_quantile"], n_pos: int = 1, partner_seed: int = 0,
                 protocol: str = "v1", max_episodes: int = 0, sources: list[str] | None = None):
        self.cfg = cfg
        overrides = {"manifest_sources": list(sources)} if sources else {}
        self.ds = make_dataset(cfg, split=split, segment_mode="fixed", **overrides)
        if max_episodes and max_episodes < len(self.ds):
            # task-stratified fixed subset (same rule as HOM), for large multi-source held-out sets
            picks = _subset([self.ds.repos[ri].task for ri, _ in self.ds.items], max_episodes, np.random.default_rng(0))
            self.ds.items = [self.ds.items[i] for i in picks]
        self.policy = make_policy(cfg, ckpt)
        self.text_policy = make_policy(load_cfg(textonly_ckpt), textonly_ckpt) if textonly_ckpt else None
        self.seeds, self.batch, self.n_alt, self.alt_quantile = seeds, batch, n_alt, alt_quantile
        self.n_pos, self.partner_seed, self.protocol = n_pos, partner_seed, protocol
        ds = self.ds
        self.n = len(ds)
        self.task = np.array([ds.repos[ri].task for ri, _ in ds.items])
        self.is_rand = np.array(["randomized" in ds.repos[ri].repo for ri, _ in ds.items])
        self.repo_of = np.array([ri for ri, _ in ds.items])
        self.source = np.array([source_of(ds.repos[ri].rel) for ri, _ in ds.items])
        self.states = [ds.repos[ri].states[ep] for ri, ep in ds.items]
        self.actions = [ds.repos[ri].actions[ep] for ri, ep in ds.items]
        self.lengths = np.array([len(s) for s in self.states])
        # action trajectories on normalized time (own action dims; only compared inside a task, i.e. one robot)
        self.traj = [traj_vector(a) for a in self.actions]
        self.arm, self.config_at_close, self.config_at_place = [], [], []
        for (ri, _), s in zip(ds.items, self.states):
            emb = ds.repos[ri].emb
            # grasp arm / position proxies are defined for the dual-arm 6+1 layout only (RoboTwin, Lift-2, Split Aloha)
            fc = E.first_close(s, emb) if emb.dual_arm_61 else None
            self.arm.append(fc[0] if fc else "none")
            joints = emb.arms[fc[0]][0] if fc else None
            self.config_at_close.append(s[fc[1], joints] if fc else None)
            reopen = E.first_reopen(s, emb, fc[0], fc[1]) if fc else None
            # never reopened before the end (object still held): the last frame is the place configuration
            place = reopen if reopen is not None else len(s) - 1
            self.config_at_place.append(s[place, joints] if fc else None)
        self.arm = np.array(self.arm)
        # v1: position = arm configuration at the first gripper close (closer to the grasp position);
        # v2: at the first reopen after it (the place), partners from the same source (clean / randomized)
        self.position_config = self.config_at_place if protocol == "v2" else self.config_at_close
        self.starts = [list(range(0, int(L), ds.cfg.segment_len)) for L in self.lengths]
        self.scene = None  # [n, D] first-slot head-camera ViT feature, filled by encode_all
        self._inverters: dict[int, ActionInverter] = {}

    # --- partners -----------------------------------------------------------------------------------------------
    def partners(self, rng) -> dict[str, list]:
        out = {k: [None] * self.n for k in ("position", "arm", "hard", "wrong")}
        out["alternate"] = [[] for _ in range(self.n)]
        out["position_all"] = [[] for _ in range(self.n)]
        rngs = {k: np.random.default_rng([self.partner_seed, n]) for n, k in enumerate(("alternate", "position", "arm"))}
        scene = self.scene / np.linalg.norm(self.scene, axis=1, keepdims=True)
        by_task = defaultdict(list)
        for i, t in enumerate(self.task):
            by_task[t].append(i)
        for t, idx in by_task.items():
            d = {(i, j): float(np.linalg.norm(self.traj[i] - self.traj[j])) for i in idx for j in idx if i != j}
            vals = list(d.values())
            alt_floor = np.quantile(vals, self.alt_quantile) if vals else 0.0
            med = np.median(vals) if vals else 0.0
            sim = {(i, j): float(scene[i] @ scene[j]) for i in idx for j in idx if i != j}
            sim_floor = np.quantile(list(sim.values()), 0.75) if sim else 1.0
            v2 = self.protocol == "v2"
            # v1 keeps its single shared stream (reproducible partners); v2 gives every category its own stream,
            # so changing the position / hard rules does not change the alternate / arm partners
            r_alt, r_pos, r_arm = (rngs["alternate"], rngs["position"], rngs["arm"]) if v2 else (rng, rng, rng)
            # v2 source = the repo (RoboTwin: clean_50 vs randomized; other datasets: one repo per task folder)
            group = (lambda i: (self.arm[i], int(self.repo_of[i]))) if v2 else (lambda i: self.arm[i])  # noqa: E731
            close = {}
            for g in {group(i) for i in idx if self.arm[i] != "none"}:
                members = [i for i in idx if group(i) == g and self.position_config[i] is not None]
                cd = {(i, j): float(np.linalg.norm(self.position_config[i] - self.position_config[j]))
                      for i in members for j in members if i != j}
                close[g] = (cd, np.median(list(cd.values())) if cd else 0.0)
            for i in idx:
                elig = [j for j in idx if j != i and d[(i, j)] > alt_floor]
                if elig:
                    out["alternate"][i] = [int(j) for j in r_alt.choice(elig, size=min(self.n_alt, len(elig)), replace=False)]
                if self.arm[i] != "none" and group(i) in close:
                    cd, cmed = close[group(i)]
                    far = [j for (a, j), v in cd.items() if a == i and v > cmed]
                    if far:
                        # n_pos == 1 keeps the single draw (same partners as earlier runs with the same seed)
                        picks = [r_pos.choice(far)] if self.n_pos == 1 else r_pos.choice(far, size=min(self.n_pos, len(far)), replace=False)
                        out["position"][i] = int(picks[0])
                        out["position_all"][i] = [int(j) for j in picks]
                other = [j for j in idx if self.arm[j] not in (self.arm[i], "none") and self.arm[i] != "none"]
                if other:
                    out["arm"][i] = int(r_arm.choice(other))
                if v2:
                    # same grasp arm and source: scene in the top quarter and actions farther than the median,
                    # both measured inside that (task, arm, source) group
                    mates = [j for j in idx if j != i and self.arm[i] != "none" and group(j) == group(i)]
                    gp = [(a, b) for a in mates + [i] for b in mates + [i] if a < b]
                    if len(mates) >= 2:
                        g_sim = np.quantile([sim[p] for p in gp], 0.75)
                        g_med = np.median([d[p] for p in gp])
                        hard = [j for j in mates if sim[(i, j)] >= g_sim and d[(i, j)] > g_med]
                    else:
                        hard = []
                else:
                    hard = [j for j in idx if j != i and sim[(i, j)] >= sim_floor and d[(i, j)] > med]
                if hard:
                    out["hard"][i] = max(hard, key=lambda j: sim[(i, j)])
            others = np.flatnonzero(self.task != t)
            for i in idx:
                out["wrong"][i] = int(others[np.argmax(scene[others] @ scene[i])])
        return out

    # --- encoding -----------------------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_all(self, policy, transform=None) -> torch.Tensor | None:
        if policy.model.demo_encoder is None:
            return None
        zs, scene = [], []
        for s in range(0, self.n, 32):
            batch = to_device(collate([self.ds.demo(i) for i in range(s, min(self.n, s + 32))]))
            if self.scene is None:
                scene.append(batch["demo.feats"][:, 0, 0].float().mean(1).cpu())
            if transform is not None:
                batch = transform(batch, s)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                zs.append(policy.model.encode_demo(batch))
        if self.scene is None:
            self.scene = torch.cat(scene).numpy()
        return torch.cat(zs)

    def shuffle(self, batch, offset):
        k = batch["demo.feats"].shape[1]
        out = dict(batch)
        feats = batch["demo.feats"].clone()
        for b in range(feats.shape[0]):
            perm = torch.randperm(k, generator=torch.Generator().manual_seed(1000 + offset + b)).to(feats.device)
            feats[b] = batch["demo.feats"][b, perm]
        out["demo.feats"] = feats
        return out

    def mapped_starts(self, i: int, j: int) -> list[int]:
        """Segment starts of episode j at the same normalized times as the segment starts of episode i."""
        return [int(round(s / max(1, self.lengths[i] - 1) * (self.lengths[j] - 1))) for s in self.starts[i]]

    # --- prediction ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def errors(self, policy, z_rows, generic: bool, use_demo: bool = True, follow=None, state_from=None,
               zero_state: bool = False, keep_pred: bool = False, dropped: bool = False) -> dict:
        """z_rows[i] = Z_demo for target episode i (or None) -> {"err": {(episode, segment, seed): error}, ...}.
        follow[i] = partner j: also the error w.r.t. j's segment at the same normalized time ("err_partner").
        state_from[i] = partner j: the state slot gets j's state at the same normalized time (G6)."""
        out, out_b, preds = {}, {}, {}
        dim = int(policy.config.max_action_dim)
        horizon = int(policy.config.chunk_size)
        order = sorted(range(self.n), key=lambda i: len(self.starts[i]))
        for s in range(0, self.n, self.batch):
            idx = order[s : s + self.batch]
            # pad every episode in the micro-batch to the same number of segments by repeating its last segment
            m = max(len(self.starts[i]) for i in idx)
            pad = lambda xs: xs + [xs[-1]] * (m - len(xs))  # noqa: E731
            batch = to_device(collate([self.ds.target(i, pad(self.starts[i]), generic=[generic] * m, dropped=[dropped] * m)
                                       for i in idx]))
            if state_from is not None:
                alt = [self.ds.target(state_from[i], pad(self.mapped_starts(i, state_from[i])), generic=[generic] * m)
                       if state_from[i] is not None else None for i in idx]
                st = batch[OBS_STATE].clone()
                for bi, a in enumerate(alt):
                    if a is not None:
                        st[bi] = a[OBS_STATE].to(st.device)
                batch[OBS_STATE] = st
            if zero_state:
                batch[OBS_STATE] = torch.zeros_like(batch[OBS_STATE])
            partner = None
            if follow is not None:
                rows = [collate([self.ds.target(follow[i] if follow[i] is not None else i,
                                                pad(self.mapped_starts(i, follow[i]) if follow[i] is not None else self.starts[i]),
                                                generic=[generic] * m)]) for i in idx]
                partner = to_device({k: torch.cat([r[k] for r in rows]) for k in (ACTION, ACTION_VALID_MASK)})
            z = torch.stack([z_rows[i] for i in idx]) if use_demo else None
            target = batch[ACTION].reshape(len(idx) * m, horizon, -1)
            valid = batch[ACTION_VALID_MASK].reshape(len(idx) * m, -1).bool()
            for seed in range(self.seeds):
                noise = torch.stack([torch.randn((m, horizon, dim), generator=torch.Generator().manual_seed(seed * 100003 + i))
                                     for i in idx]).reshape(len(idx) * m, horizon, dim).cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pred = policy.model.sample_segments(batch, z_demo=z, noise=noise, use_demo=use_demo).float()
                err = masked_error(pred, target, valid).reshape(len(idx), m).cpu().numpy()
                if partner is not None:
                    pv = partner[ACTION_VALID_MASK].reshape(len(idx) * m, -1).bool() & valid
                    err_b = masked_error(pred, partner[ACTION].reshape(len(idx) * m, horizon, -1), pv)
                    err_b = err_b.reshape(len(idx), m).cpu().numpy()
                p = pred.reshape(len(idx), m, horizon, -1)[..., : target.shape[-1]].cpu().numpy() if keep_pred and seed == 0 else None
                for bi, i in enumerate(idx):
                    for k in range(len(self.starts[i])):
                        out[(i, k, seed)] = float(err[bi, k])
                        if partner is not None and follow[i] is not None:
                            out_b[(i, k, seed)] = float(err_b[bi, k])
                        if p is not None:
                            preds[(i, k)] = p[bi, k]
            logging.info("segments %d / %d", min(self.n, s + self.batch), self.n)
        return {"err": out, "err_partner": out_b, "pred": preds}

    # --- round-trip ---------------------------------------------------------------------------------------------
    def inverter(self, i: int) -> ActionInverter:
        ri = self.ds.items[i][0]
        if ri not in self._inverters:
            self._inverters[ri] = ActionInverter(self.ds.repos[ri])
        return self._inverters[ri]

    def round_trip(self, preds: dict) -> dict:
        horizon = int(self.ds.cfg.segment_len)
        # the inverse transform must reproduce the raw actions from the normalized labels
        tgt = self.ds.target(0, self.starts[0][:1], generic=[False])
        v = int(tgt[ACTION_VALID_MASK][0].sum())
        rec = self.inverter(0)(tgt[ACTION][0].numpy(), self.states[0][0])[:v]
        inverse_err = float(np.abs(rec - self.actions[0][:v, : rec.shape[1]]).max())
        lags, tail, middle, thirds = [], [], [], defaultdict(list)
        for i in range(self.n):
            L, segs = int(self.lengths[i]), []
            for k, s in enumerate(self.starts[i]):
                steps = min(horizon, L - s)
                seg = self.inverter(i)(preds[(i, k)], self.states[i][s])[:steps]
                segs.append(seg)
                e = float(np.mean((seg - self.actions[i][s : s + steps, : seg.shape[1]]) ** 2))
                if k == len(self.starts[i]) - 1:
                    tail.append(e)
                elif k > 0:
                    middle.append(e)
                thirds[min(2, int(3 * s / max(1, L - 1)))].append(e)
            pred = np.concatenate(segs)
            lag = best_lag(pred, self.actions[i][:L, : pred.shape[1]], PROVISIONAL["max_lag_search"])
            if lag is not None:
                lags.append(lag)
        lags = np.array(lags)
        res = {
            "inverse_transform_max_abs_error": inverse_err,
            "lag_frames": {"median": float(np.median(lags)), "mean": float(lags.mean()),
                           "median_abs": float(np.median(np.abs(lags))),
                           "frac_abs_le_2": float(np.mean(np.abs(lags) <= 2)), "episodes": int(len(lags))},
            "abs_error_thirds": [float(np.mean(thirds[j])) for j in range(3)],
            "tail_ratio": float(np.mean(tail) / max(1e-12, np.mean(middle))) if middle else None,
        }
        # systematic drift = a consistent lag across episodes, so the median (not the outlier-prone mean) is judged
        res["pass_provisional"] = bool(inverse_err < 1e-3 and abs(res["lag_frames"]["median"]) <= PROVISIONAL["max_abs_lag"]
                                       and res["lag_frames"]["median_abs"] <= PROVISIONAL["max_abs_lag"]
                                       and (res["tail_ratio"] is None or res["tail_ratio"] <= PROVISIONAL["tail_ratio"]))
        return res

    # --- run ----------------------------------------------------------------------------------------------------
    def run(self, policy=None, part=None, with_textonly: bool = True) -> dict:
        policy = policy or self.policy
        z = self.encode_all(policy)
        if part is None:
            part = self.partners(np.random.default_rng(self.partner_seed))
        z_blank = self.encode_all(policy, lambda b, o: {**b, "demo.feats": torch.zeros_like(b["demo.feats"])})
        z_shuffle = self.encode_all(policy, self.shuffle)
        rows = lambda name: [z[j] if j is not None else z[i] for i, j in enumerate(part[name])]  # noqa: E731
        first_alt = [a[0] if a else None for a in part["alternate"]]
        cond, follow = {}, {}
        logging.info("matched")
        r = self.errors(policy, z, generic=False, keep_pred=True)
        cond["matched"], preds = r["err"], r["pred"]
        cond["matched_generic"] = self.errors(policy, z, generic=True)["err"]
        # text masked out (BOS visible): G4 without a text-only model (AE-3 plan 7.1)
        cond["matched_textdrop"] = self.errors(policy, z, generic=False, dropped=True)["err"]
        cond["blank"] = self.errors(policy, z_blank, generic=False)["err"]
        cond["shuffle"] = self.errors(policy, z_shuffle, generic=False)["err"]
        alt_err = defaultdict(list)
        for a in range(self.n_alt):
            js = [al[a] if len(al) > a else None for al in part["alternate"]]
            r = self.errors(policy, [z[j] if j is not None else z[i] for i, j in enumerate(js)], generic=False,
                            follow=js if a == 0 else None)
            for key, e in r["err"].items():
                if js[key[0]] is not None:
                    alt_err[key].append(e)
            if a == 0:
                follow["alternate"] = r["err_partner"]
        cond["alternate"] = {k: float(np.mean(v)) for k, v in alt_err.items()}
        cond["alternate_generic"] = self.errors(policy, [z[j] if j is not None else z[i] for i, j in enumerate(first_alt)],
                                                generic=True)["err"]
        for name in ("position", "arm", "hard", "wrong"):
            r = self.errors(policy, rows(name), generic=False, follow=part[name] if name in ("position", "arm") else None)
            cond[name] = r["err"]
            if name in ("position", "arm"):
                follow[name] = r["err_partner"]
        # further position partners (--n_pos > 1): errors averaged over the partners of each target episode
        pos_err = defaultdict(list, {k: [e] for k, e in cond["position"].items() if part["position"][k[0]] is not None})
        for a in range(1, self.n_pos):
            js = [p[a] if len(p) > a else None for p in part["position_all"]]
            if not any(j is not None for j in js):
                break
            for key, e in self.errors(policy, [z[j] if j is not None else z[i] for i, j in enumerate(js)], generic=False)["err"].items():
                if js[key[0]] is not None:
                    pos_err[key].append(e)
        if self.n_pos > 1:
            cond["position"] = {**cond["position"], **{k: float(np.mean(v)) for k, v in pos_err.items()}}
        if with_textonly and self.text_policy is not None:
            cond["textonly"] = self.errors(self.text_policy, [None] * self.n, generic=False, use_demo=False)["err"]
        g6 = None
        if policy.config.use_segment_start_state:
            cond["swap_state"] = self.errors(policy, z, generic=False, state_from=first_alt)["err"]
            cond["state_masked"] = self.errors(policy, z, generic=False, zero_state=True)["err"]

        has_partner = {"alternate": lambda i: bool(part["alternate"][i]), "alternate_generic": lambda i: first_alt[i] is not None,
                       "swap_state": lambda i: first_alt[i] is not None}
        for name in ("position", "arm", "hard", "wrong"):
            has_partner[name] = (lambda nm: lambda i: part[nm][i] is not None)(name)

        def ratio(name, base="matched", episodes=None):
            keys = [k for k in cond[base] if (episodes is None or k[0] in episodes)]
            if name in has_partner:
                keys = [k for k in keys if has_partner[name](k[0])]
            if not keys:
                return None
            num = np.array([cond[name][k] for k in keys])
            den = np.array([cond[base][k] for k in keys])
            return paired_bootstrap_ratio(num, den, np.array([k[0] for k in keys]))

        main_conds = [c for c in ("alternate", "blank", "shuffle", "wrong", "position", "arm", "hard", "textonly") if c in cond]
        res = {"n_episodes": self.n, "n_segments": len(cond["matched"]) // self.seeds, "noise_samples_K": self.seeds,
               "n_alt": self.n_alt, "alt_quantile": self.alt_quantile,
               "n_pos": self.n_pos, "partner_seed": self.partner_seed, "protocol": self.protocol,
               "partners_found": {k: int(sum(bool(x) if isinstance(x, list) else x is not None for x in v)) for k, v in part.items()},
               "mean_error": {k: float(np.mean(list(v.values()))) for k, v in cond.items()}}
        res["ratios"] = {name: ratio(name) for name in main_conds}
        res["ratios"]["alternate_generic_vs_matched_generic"] = ratio("alternate_generic", "matched_generic")
        res["ratios"]["matched_generic_vs_matched"] = ratio("matched_generic")
        res["ratios"]["matched_textdrop_vs_matched"] = ratio("matched_textdrop")
        res["by_split"] = {sp: {n: ratio(n, episodes=set(np.flatnonzero(self.is_rand == (sp == "rand"))))
                                for n in ("alternate", "position", "arm")} for sp in ("clean", "rand")}
        res["by_task"] = {t: {n: ratio(n, episodes=set(np.flatnonzero(self.task == t))) for n in main_conds}
                          for t in np.unique(self.task)}
        res["by_source"] = {s: {n: ratio(n, episodes=set(np.flatnonzero(self.source == s)))
                                for n in main_conds + ["matched_textdrop"] if n in cond}
                            for s in np.unique(self.source)}

        # follow check: with B's video the output is closer to B's segment than to A's
        res["follow"] = {}
        for name, eb in follow.items():
            keys = [k for k in eb if k in cond[name]]
            res["follow"][name] = {"frac_closer_to_partner": float(np.mean([eb[k] < cond[name][k] for k in keys])) if keys else None,
                                   "n": len(keys)}

        # Z_demo distance vs action-trajectory distance (within task, Spearman; bootstrap CI over tasks)
        if z is not None:
            zm = torch.nn.functional.normalize(z.float().mean(1), dim=-1).cpu().numpy()
            rhos = {}
            for t in np.unique(self.task):
                idx = np.flatnonzero(self.task == t)
                if len(idx) < 4:
                    continue
                a, b = [], []
                for x in range(len(idx)):
                    for y in range(x + 1, len(idx)):
                        a.append(1 - float(zm[idx[x]] @ zm[idx[y]]))
                        b.append(float(np.linalg.norm(self.traj[idx[x]] - self.traj[idx[y]])))
                rhos[t] = spearman(np.array(a), np.array(b))
            vals = np.array([v for v in rhos.values() if np.isfinite(v)])
            res["z_vs_action_distance"] = {**bootstrap_mean(vals), "tasks_positive": int((vals > 0).sum()), "per_task": rhos}

        res["round_trip"] = self.round_trip(preds)
        thirds = defaultdict(list)
        for (i, k, _), e in cond["matched"].items():
            thirds[min(2, int(3 * self.starts[i][k] / max(1, self.lengths[i] - 1)))].append(e)
        res["round_trip"]["normalized_error_thirds"] = [float(np.mean(thirds[j])) for j in range(3)]

        ok = lambda r: bool(r is not None and r["ratio"] >= THRESH and r["ci95"][0] > 1.0)  # noqa: E731
        rr = res["ratios"]
        res["G4"] = {
            "alternate_gap": ok(rr["alternate"]),
            "textonly_gap": ok(rr.get("textonly")) if "textonly" in rr else None,
            "video_still_used_with_generic_text": ok(rr["alternate_generic_vs_matched_generic"]),
            "blank_aux": ok(rr["blank"]),
            # design doc 7.5: suggested, adoption pending -> reported, not gating
            "generic_video_beats_textonly_canonical": (bool(res["mean_error"]["matched_generic"] < res["mean_error"]["textonly"])
                                                       if "textonly" in res["mean_error"] else None),
        }
        # without a text-only model (AE-3): the language must not be needed, i.e. the generic sentence and the masked
        # text leave the matched error within 10% (deviation from the design doc, see AE-3 plan 7.1)
        lang = [rr.get("matched_generic_vs_matched"), rr.get("matched_textdrop_vs_matched")]
        res["G4"]["language_not_needed"] = bool(all(r is not None and r["ratio"] < THRESH for r in lang))
        res["G4"]["pass"] = bool(res["G4"]["alternate_gap"] and res["G4"]["video_still_used_with_generic_text"]
                                 and (res["G4"]["textonly_gap"] if res["G4"]["textonly_gap"] is not None
                                      else res["G4"]["language_not_needed"]))
        zc = res.get("z_vs_action_distance", {})
        res["G5"] = {"position_gap": ok(rr["position"]), "arm_gap": ok(rr["arm"]),
                     "hard_negative_gap": ok(rr["hard"]) if rr.get("hard") else None,
                     "z_corr_ci_low": zc.get("ci95", [None])[0],
                     "z_distance_tracks_action_distance": bool(zc.get("ci95", [None])[0] is not None
                                                               and zc["ci95"][0] > PROVISIONAL["z_corr_ci_low"])}
        res["G5"]["pass"] = bool(res["G5"]["position_gap"] and res["G5"]["arm_gap"]
                                 and res["G5"]["z_distance_tracks_action_distance"])
        if policy.config.use_segment_start_state:
            g6 = {"swap_video_gap": ok(rr["alternate"]), "swap_state": ratio("swap_state"), "state_masked": ratio("state_masked"),
                  "video_plus_state_vs_state_only": res["mean_error"]["matched"] / res["mean_error"]["blank"]}
            g6["pass"] = bool(g6["swap_video_gap"] and g6["video_plus_state_vs_state_only"] <= 0.95)
        res["G6"] = g6 if g6 is not None else {"status": "N/A (use_segment_start_state is off)"}
        res["G7_round_trip"] = {"pass": res["round_trip"]["pass_provisional"], "provisional": True}
        res["G7_heldout_part"] = {"G4_G5_heldout": bool(res["G4"]["pass"] and res["G5"]["pass"]),
                                  "round_trip_ok_provisional": res["G7_round_trip"]["pass"]}
        res["next_steps"] = next_steps(res)
        return res


def next_steps(res: dict) -> list[str]:
    """Side branches of design doc 4.0 for the failed checks."""
    out, rr = [], res["ratios"]
    if not res["G4"]["pass"]:
        if res["G4"]["textonly_gap"] is False:
            out.append("G4: text-only close to the main model -> 3.5.1 raise generic_text_prob (0.8, then 1.0); rerun from AE-2")
        if not res["G4"]["alternate_gap"] and not res["G4"]["blank_aux"]:
            out.append("G4: blank and alternate close to matched -> check the S-U control, then 3.5.3 expert_context=full_prefix; AE-2")
        if not res["G4"]["video_still_used_with_generic_text"]:
            out.append("G4: video unused with the generic text -> check that the video enters the Q-Former; do not move on")
    if not res["G5"]["position_gap"]:
        out.append("G5: position class failed -> 3.2 P64; rerun from AE-0 (G2) -> AE-1 -> AE-2")
    if not res["G5"]["arm_gap"]:
        out.append("G5: arm class failed -> side branch pending in the design doc")
    if rr.get("blank") and not res["G4"]["blank_aux"] and res["G4"]["alternate_gap"]:
        out.append("note: alternate gap without a blank gap -> check blank as out-of-distribution input (not a veto)")
    if not res["G7_round_trip"]["pass"]:
        out.append("G7: round-trip check failed (provisional thresholds) -> check data distribution, text leakage, video / action alignment")
    return out


def seeds_consistent(a: dict, b: dict) -> dict:
    same = {k: bool(a["ratios"].get(k) and b["ratios"].get(k) and (a["ratios"][k]["ratio"] > 1) == (b["ratios"][k]["ratio"] > 1))
            for k in ("alternate", "position", "arm")}
    return {"same_direction": same, "both_G4_G5": bool(a["G4"]["pass"] and a["G5"]["pass"] and b["G4"]["pass"] and b["G5"]["pass"]),
            "pass": bool(all(same.values()) and a["G4"]["pass"] and a["G5"]["pass"] and b["G4"]["pass"] and b["G5"]["pass"])}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--textonly_ckpt", default=None)
    ap.add_argument("--ckpt_seed2", default=None, help="same variant trained with the second seed (G7)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--seeds", type=int, default=1, help="flow noise draws K per segment (design doc 7.0, K pending)")
    ap.add_argument("--n_alt", type=int, default=1, help="alternate partners per target episode")
    ap.add_argument("--n_pos", type=int, default=1, help="position-class partners per target episode (errors averaged)")
    ap.add_argument("--partner_seed", type=int, default=0, help="rng seed of the partner choice")
    ap.add_argument("--protocol", choices=["v1", "v2"], default="v1",
                    help="G5 partners. v1: position = arm configuration at the first gripper close, hard negatives "
                         "over the whole task. v2: position = configuration at the place (first reopen); position "
                         "and hard-negative partners from the same grasp arm and the same source (clean / randomized)")
    ap.add_argument("--alt_quantile", type=float, default=PROVISIONAL["alt_quantile"])
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--split", choices=["val", "all"], default="val",
                    help="val = held-out of the training repos; all = every episode of --dataset.repo_id_file (test data)")
    ap.add_argument("--max_episodes", type=int, default=0,
                    help="task-stratified subset of the held-out episodes (AE-3 has ~10k; 0 = all)")
    ap.add_argument("--sources", default="", help="','-separated top-level sources of the split manifest (AE-3)")
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    ev = Evaluator(cfg, args.ckpt, args.textonly_ckpt, args.seeds, args.batch, split=args.split, n_alt=args.n_alt,
                   alt_quantile=args.alt_quantile, n_pos=args.n_pos, partner_seed=args.partner_seed,
                   protocol=args.protocol, max_episodes=args.max_episodes,
                   sources=args.sources.split(",") if args.sources else None)
    res = ev.run()
    res["provisional_thresholds"] = PROVISIONAL
    if args.ckpt_seed2:
        part = ev.partners(np.random.default_rng(args.partner_seed))  # same scene features and rng: same partners
        policy2 = make_policy(load_cfg(args.ckpt_seed2, overrides), args.ckpt_seed2)
        res2 = ev.run(policy2, part=part, with_textonly="textonly" in res["ratios"])
        res["seed2"] = {"ckpt": args.ckpt_seed2, "ratios": res2["ratios"], "G4": res2["G4"], "G5": res2["G5"],
                        "G7_round_trip": res2["G7_round_trip"]}
        res["G7_seeds"] = seeds_consistent(res, res2)
    else:
        res["G7_seeds"] = {"pass": None, "note": "needs --ckpt_seed2 (or freeze_ae --seed2_gate)"}
    res["G7"] = {"pass": bool(res["G7_heldout_part"]["G4_G5_heldout"] and res["G7_round_trip"]["pass"]
                              and res["G7_seeds"]["pass"]),
                 "sampler_cache_consistent": None}
    res.update({"ckpt": args.ckpt, "textonly_ckpt": args.textonly_ckpt, "split": args.split,
                "repo_id_file": cfg.dataset.repo_id_file, "reproducibility": reproducibility(cfg, args.ckpt, ev.policy, ev.ds)})
    res["G7"]["sampler_cache_consistent"] = res["reproducibility"].get("sampler_matches_config")
    path = write_result(args.out, args.name, res)
    print(path)
    print({k: res[k] for k in ("mean_error", "G4", "G5", "G7_heldout_part", "G7_seeds")})


if __name__ == "__main__":
    main()
