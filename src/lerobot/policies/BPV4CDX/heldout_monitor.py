"""HOM (held-out monitor): the actual generation error on the held-out episodes, at every checkpoint save.

The training loss is a flow-matching velocity error at random noise levels; it is not the error of the actions the
model really produces. HOM samples the actions (fixed noise per episode, the policy's Euler steps) for every fixed
target segment (starts 0, 50, ...) of unseen held-out episodes and reports the masked MSE to the true actions:
  matched           own demo, canonical text
  matched_generic   own demo, generic text
  matched_textdrop  own demo, text masked out (attention mask all zero, BOS still visible) -> textdrop / matched;
                    out of distribution unless the model was trained with dataset.text_drop_prob > 0
  alternate         another held-out demo of the same task whose action trajectory is farther than the task
                    median (fixed partners), canonical text -> E_alt / E_match (optional)
Episodes: a fixed, task-stratified subset (dataset.heldout_monitor_max_episodes, round robin over tasks; 0 = all);
every heldout_monitor_full_every-th call all held-out episodes (the subset metrics are then reported as well, from
the same predictions, so the subset series never breaks).
Errors: per-segment MSE over valid steps and real action dims, averaged over segments; in the normalized delta space
(`mse`, the same metric and noise as gate_g4_g7 with K = 1) and in action units (`mse_raw`); per source (`by_source`).
End-effector error (matched, RoboTwin aloha-agilex episodes only): predicted and true actions back to absolute joints
(from the true segment-start state), link6 positions by forward kinematics (ee_fk.py), mean distance in cm over the
valid steps of a segment: `ee_cm_matched` (both arms) and `ee_cm_matched_moving` (the arm with the longer true path).
Trend (`trend_advice`): changes over the recent saves and whether stopping is recommended (AE-3 plan 6.3); advisory.

In training: lerobot_train calls dataset.make_heldout_monitor() once and the monitor after every checkpoint save;
results go to the log, wandb and <output_dir>/heldout_monitor.jsonl.
Offline (back-fill a finished run): python -m lerobot.policies.BPV4CDX.heldout_monitor --run <output_dir>/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from .ae3_data import source_of
from .common import ActionInverter, collate, masked_error, to_device
from .ee_fk import ArmFK
from .paths import is_robotwin_like

REPO_URDF = Path(__file__).resolve().parents[4] / "assets/urdf/aloha-agilex/arx5_description_isaac.urdf"


def _traj(actions: np.ndarray, points: int = 64) -> np.ndarray:
    t = np.linspace(0, len(actions) - 1, points)
    return np.stack([np.interp(t, np.arange(len(actions)), actions[:, d]) for d in range(actions.shape[1])], 1).reshape(-1)


def _subset(tasks: list[str], k: int, rng) -> list[int]:
    """Task-stratified fixed choice of k episodes: round robin over tasks, random order inside each task."""
    n = len(tasks)
    if not k or k >= n:
        return list(range(n))
    pools = {t: list(rng.permutation([i for i in range(n) if tasks[i] == t])) for t in sorted(set(tasks))}
    picks = []
    while len(picks) < k:
        for t in pools:
            if pools[t] and len(picks) < k:
                picks.append(int(pools[t].pop(0)))
    return sorted(picks)


class HeldoutMonitor:
    def __init__(self, train_ds) -> None:
        cfg = train_ds.cfg
        self.cfg = cfg
        self.ds = train_ds.derive(split="val", segment_mode="fixed", max_trajectories=0, episode_whitelist=[])
        n = len(self.ds)
        repos = self.ds.repos
        tasks = [repos[ri].task for ri, _ in self.ds.items]
        rng = np.random.default_rng(0)
        self.subset = _subset(tasks, int(cfg.heldout_monitor_max_episodes), rng)
        self.full_every = int(getattr(cfg, "heldout_monitor_full_every", 0))
        self.all = list(range(n)) if self.full_every else self.subset
        self.calls = 0
        self.batch = int(cfg.heldout_monitor_batch)
        seg = int(cfg.segment_len)
        self.starts = {i: list(range(0, len(repos[self.ds.items[i][0]].states[self.ds.items[i][1]]), seg)) for i in self.all}
        self.is_rand = {i: "randomized" in repos[self.ds.items[i][0]].repo for i in self.all}
        self.source = {i: source_of(repos[self.ds.items[i][0]].rel) for i in self.all}
        self.sources = sorted(set(self.source.values()))
        self.std, self.inv = {}, {}
        for i in self.all:
            ri = self.ds.items[i][0]
            if ri not in self.std:
                inv = ActionInverter(repos[ri])
                self.std[ri] = torch.tensor(inv.std, dtype=torch.float32)
                self.inv[ri] = tuple(torch.tensor(x, dtype=torch.float64) for x in (inv.mean, inv.std, inv.mask * (not inv.skip)))
        urdf = getattr(cfg, "heldout_monitor_urdf", "auto")
        urdf = str(REPO_URDF) if urdf == "auto" else urdf
        self.fk = ArmFK(urdf) if urdf and Path(urdf).is_file() else None
        # end-effector error only where the URDF applies: RoboTwin aloha-agilex (dual-arm 6+1)
        self.ee_ok = {i: bool(self.fk is not None and is_robotwin_like(repos[self.ds.items[i][0]].repo)
                              and repos[self.ds.items[i][0]].emb.dual_arm_61) for i in self.all}
        if self.fk is None:
            logging.info("HOM: no URDF (%r), end-effector error skipped", urdf)
        # alternate demos: any held-out demo of the same task (only the targets are sub-sampled)
        self.partner = {}
        if cfg.heldout_monitor_alternate:
            by_task = defaultdict(list)
            for b in range(n):
                by_task[tasks[b]].append(b)
            traj = {}
            for a in self.all:
                pool = by_task[tasks[a]]
                for b in pool:
                    if b not in traj:
                        traj[b] = _traj(repos[self.ds.items[b][0]].actions[self.ds.items[b][1]])
                d = {b: float(np.linalg.norm(traj[a] - traj[b])) for b in pool if b != a and traj[b].shape == traj[a].shape}
                if not d:
                    continue
                med = float(np.median(list(d.values())))
                far = [b for b, v in d.items() if v > med]
                if far:
                    self.partner[a] = int(rng.choice(far))
        logging.info("HOM: %d held-out episodes in the subset (%d in total, full every %s), %d sources, %d alternate "
                     "partners, %d end-effector episodes", len(self.subset), n, self.full_every or "never",
                     len(self.sources), len(self.partner), sum(self.ee_ok.values()))

    @torch.no_grad()
    def _errors(self, policy, idx: list[int], generic: bool, demo_of: dict | None = None, dropped: bool = False):
        """-> per-segment normalized MSE, raw MSE, (episode, segment) keys, end-effector cm [S, 2] or None."""
        model, dim = policy.model, int(policy.config.max_action_dim)
        horizon = int(policy.config.chunk_size)
        m = max(len(self.starts[i]) for i in idx)
        pad = lambda xs: xs + [xs[-1]] * (m - len(xs))  # noqa: E731
        use_demo = model.demo_encoder is not None
        items = []
        for i in idx:
            demo = self.ds.demo(demo_of.get(i, i) if demo_of else i) if use_demo else {}
            items.append({**demo, **self.ds.target(i, pad(self.starts[i]), generic=[generic] * m, dropped=[dropped] * m)})
        batch = to_device(collate(items))
        noise = torch.stack([torch.randn((m, horizon, dim), generator=torch.Generator().manual_seed(i)) for i in idx])
        noise = noise.reshape(len(idx) * m, horizon, dim).cuda()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model.sample_segments(batch, noise=noise, use_demo=use_demo).float()
        target = batch[ACTION].reshape(len(idx) * m, horizon, -1)
        valid = batch[ACTION_VALID_MASK].reshape(len(idx) * m, -1).bool()
        err = masked_error(pred, target, valid).reshape(len(idx), m)
        # raw units: per repo action dims (different robots in one batch)
        raw = torch.zeros(len(idx), m, device=pred.device)
        for b, i in enumerate(idx):
            std = self.std[self.ds.items[i][0]].to(pred.device)
            a = std.numel()
            sl = slice(b * m, (b + 1) * m)
            diff = (pred[sl, :, :a] - target[sl, :, :a]) * std
            w = valid[sl, :, None].float()
            raw[b] = ((diff ** 2) * w).sum((1, 2)) / (w.sum((1, 2)) * a).clamp_min(1)
        keep = [(b, k) for b, i in enumerate(idx) for k in range(len(self.starts[i]))]
        ee = None
        ee_idx = [b for b, i in enumerate(idx) if self.ee_ok[i]]
        if ee_idx:
            ee_all = np.full((len(idx), m, 2), np.nan)
            sel = [idx[b] for b in ee_idx]
            rows = torch.cat([torch.arange(b * m, (b + 1) * m) for b in ee_idx]).to(pred.device)
            ee_all[ee_idx] = self._ee_cm(sel, pad, pred[rows], target[rows], valid[rows], m, horizon)
            ee = np.array([ee_all[b, k] for b, k in keep])
        return (np.array([float(err[b, k]) for b, k in keep]), np.array([float(raw[b, k]) for b, k in keep]),
                [(idx[b], k) for b, k in keep], ee)

    def _ee_cm(self, idx, pad, pred, target, valid, m, horizon) -> np.ndarray:
        """Per segment [B, m, 2]: mean link6 distance (cm) over valid steps, both arms averaged / the moving arm.
        Actions back to absolute joints: normalized delta * (std + eps) + mean + segment-start state on delta dims."""
        dev = pred.device
        a = 14
        mean = torch.stack([self.inv[self.ds.items[i][0]][0][:a] for i in idx]).to(dev)
        std = torch.stack([self.inv[self.ds.items[i][0]][1][:a] for i in idx]).to(dev)
        mask = torch.stack([self.inv[self.ds.items[i][0]][2][:a] for i in idx]).to(dev)
        start = torch.stack([torch.as_tensor(np.stack([self.ds.repos[self.ds.items[i][0]].states[self.ds.items[i][1]][s, :a]
                                                       for s in pad(self.starts[i])]), dtype=torch.float64) for i in idx]).to(dev)
        base = (start * mask[:, None]).reshape(len(idx), m, 1, a)

        def absolute(x):
            x = x[..., :a].double().reshape(len(idx), m, horizon, a)
            return x * (std + 1e-6)[:, None, None] + mean[:, None, None] + base

        p, t = self.fk.positions(absolute(pred)), self.fk.positions(absolute(target))  # [B, m, H, 2, 3]
        dist = (p - t).norm(dim=-1) * 100  # cm, [B, m, H, 2]
        w = valid.reshape(len(idx), m, horizon, 1).double()
        per_arm = (dist * w).sum(2) / w.sum(2).clamp_min(1)  # [B, m, 2]
        motion = (t[:, :, 1:] - t[:, :, :-1]).norm(dim=-1).mul(w[:, :, 1:]).sum(2)  # true path length per arm
        moving = per_arm.gather(-1, motion.argmax(-1, keepdim=True)).squeeze(-1)
        return torch.stack([per_arm.mean(-1), moving], -1).cpu().numpy()

    # accumulator layout (summed across ranks): global block + per-source block, for the run set and the subset
    G = 22  # mse, raw, n | clean mse, n | rand mse, n | generic mse, raw, n | alt mse, matched-on-alt, n |
            # ee both, ee moving, n_ee | textdrop mse, raw, n | (2 spare)

    def _accumulate(self, acc, e, r, keys, ee, eg, rg, ed, rd, alt):
        rand = np.array([self.is_rand[i] for i, _ in keys])
        acc[0:3] += [e.sum(), r.sum(), len(e)]
        acc[3:5] += [e[~rand].sum(), (~rand).sum()]
        acc[5:7] += [e[rand].sum(), rand.sum()]
        acc[7:10] += [eg.sum(), rg.sum(), len(eg)]
        if ee is not None:
            ok = ~np.isnan(ee[:, 0])
            acc[13:16] += [ee[ok, 0].sum(), ee[ok, 1].sum(), ok.sum()]
        acc[16:19] += [ed.sum(), rd.sum(), len(ed)]
        if alt is not None:
            ea, ka = alt
            em = {key: v for key, v in zip(keys, e)}
            acc[10:13] += [ea.sum(), sum(em[k] for k in ka), len(ea)]
        for s_i, s in enumerate(self.sources):
            sel = np.array([self.source[i] == s for i, _ in keys])
            if sel.any():
                o = self.G + 4 * s_i
                acc[o : o + 4] += [e[sel].sum(), sel.sum(), eg[sel].sum(), ed[sel].sum()]

    def _metrics(self, t, n_episodes) -> dict:
        ratio = lambda a, b: float(a / b) if b > 0 else None  # noqa: E731
        out = {"episodes": n_episodes, "segments": int(t[2]),
               "mse_matched": ratio(t[0], t[2]), "mse_raw_matched": ratio(t[1], t[2]),
               "mse_matched_clean": ratio(t[3], t[4]), "mse_matched_rand": ratio(t[5], t[6]),
               "mse_matched_generic": ratio(t[7], t[9]), "mse_raw_matched_generic": ratio(t[8], t[9]),
               "mse_matched_textdrop": ratio(t[16], t[18]), "textdrop_over_match": ratio(t[16], t[0])}
        if t[15] > 0:
            out.update({"ee_cm_matched": ratio(t[13], t[15]), "ee_cm_matched_moving": ratio(t[14], t[15]),
                        "ee_segments": int(t[15])})
        if t[12] > 0:
            out.update({"mse_alternate": ratio(t[10], t[12]), "alt_over_match": ratio(t[10], t[11]),
                        "alt_segments": int(t[12])})
        out["by_source"] = {}
        for s_i, s in enumerate(self.sources):
            o = self.G + 4 * s_i
            if t[o + 1] > 0:
                out["by_source"][s] = {"segments": int(t[o + 1]), "mse_matched": ratio(t[o], t[o + 1]),
                                       "generic_over_match": ratio(t[o + 2], t[o]), "textdrop_over_match": ratio(t[o + 3], t[o])}
        return out

    def __call__(self, policy, accelerator=None, step: int | None = None) -> dict:
        rank = accelerator.process_index if accelerator is not None else 0
        world = accelerator.num_processes if accelerator is not None else 1
        self.calls += 1
        full = bool(self.full_every and self.calls % self.full_every == 0)
        run = self.all if full else self.subset
        in_subset = set(self.subset)
        mine = run[rank::world]
        was_training = policy.training
        policy.eval()
        size = self.G + 4 * len(self.sources)
        acc, acc_sub = np.zeros(size), np.zeros(size)
        for s in range(0, len(mine), self.batch):
            idx = mine[s : s + self.batch]
            e, r, keys, ee = self._errors(policy, idx, generic=False)
            eg, rg, _, _ = self._errors(policy, idx, generic=True)
            ed, rd, _, _ = self._errors(policy, idx, generic=False, dropped=True)
            with_p = [i for i in idx if i in self.partner]
            alt = None
            if with_p:
                ea, _, ka, _ = self._errors(policy, with_p, generic=False, demo_of=self.partner)
                alt = (ea, ka)
            self._accumulate(acc, e, r, keys, ee, eg, rg, ed, rd, alt)
            if full:
                sel = np.array([i in in_subset for i, _ in keys])
                if sel.any():
                    alt_sub = None
                    if alt is not None:
                        asel = np.array([i in in_subset for i, _ in alt[1]])
                        alt_sub = (alt[0][asel], [k for k, a in zip(alt[1], asel) if a])
                    self._accumulate(acc_sub, e[sel], r[sel], [k for k, x in zip(keys, sel) if x],
                                     None if ee is None else ee[sel], eg[sel], rg[sel], ed[sel], rd[sel], alt_sub)
        if was_training:
            policy.train()
        total = torch.tensor(np.concatenate([acc, acc_sub]), device="cuda", dtype=torch.float64)
        if accelerator is not None and world > 1:
            total = accelerator.reduce(total, reduction="sum")
        t = total.cpu().numpy()
        out = {"step": step, "full": full, **self._metrics(t[:size], len(run))}
        if full:
            out["subset"] = self._metrics(t[size:], len(self.subset))
        return out


def subset_view(record: dict) -> dict:
    """The subset metrics of a HOM record (top level, or under `subset` for a full evaluation)."""
    return record.get("subset", record) if record.get("full") else record


def trend_advice(history: list[dict], current: dict, lr_at_floor: bool, window: int = 5) -> dict:
    """AE-3 plan 6.3 (advisory only): changes of the subset metrics over the last `window` saves, and 'can stop' when
    the learning rate is at its floor, each of the last 3 saves improved the matched error by < 1% and
    E_alt / E_match moved by < 2%."""
    series = [subset_view(r) for r in history + [current] if r.get("mse_matched") is not None or r.get("subset")]
    series = [s for s in series if s.get("mse_matched") is not None]
    if len(series) < 2:
        return {"recommendation": "continue", "reason": "not enough saves yet"}

    def rel(key, k):
        a, b = series[-1 - k].get(key), series[-1].get(key)
        return None if a in (None, 0) or b is None else (b - a) / abs(a)

    k = min(window, len(series) - 1)
    out = {"window_saves": k, "matched_change": rel("mse_matched", k), "alt_ratio_change": rel("alt_over_match", k),
           "ee_change": rel("ee_cm_matched_moving", k), "lr_at_floor": lr_at_floor}
    steps = [(series[-1 - j].get("mse_matched"), series[-j].get("mse_matched")) for j in range(1, min(3, len(series) - 1) + 1)]
    small = len(steps) >= 3 and all(a and b is not None and (a - b) / a < 0.01 for a, b in steps)
    alt3 = rel("alt_over_match", min(3, len(series) - 1))
    stable_alt = alt3 is None or abs(alt3) < 0.02
    out["matched_last3_each_below_1pct"] = bool(small)
    out["alt_ratio_last3_below_2pct"] = bool(stable_alt)
    reasons = []
    if not lr_at_floor:
        reasons.append("learning rate not at its floor yet")
    if not small:
        reasons.append("matched error still improving by >= 1% per save (or < 3 saves)")
    if not stable_alt:
        reasons.append("E_alt / E_match still changing by >= 2%")
    out["recommendation"] = "can stop" if not reasons else "continue"
    out["reason"] = "; ".join(reasons) if reasons else "all stop conditions met (advisory)"
    return out


def log_line(m: dict) -> str:
    keys = ("mse_matched", "mse_raw_matched", "mse_matched_generic", "mse_matched_clean", "mse_matched_rand",
            "mse_matched_textdrop", "textdrop_over_match", "mse_alternate", "alt_over_match", "ee_cm_matched",
            "ee_cm_matched_moving")
    line = ", ".join(f"{k}={m[k]:.4f}" for k in keys if m.get(k) is not None) + f", segments={m['segments']}"
    if m.get("full"):
        line += " [full held-out set]"
    if m.get("advice"):
        a = m["advice"]
        chg = lambda v: "-" if v is None else f"{100 * v:+.1f}%"  # noqa: E731
        line += (f" | last {a.get('window_saves', 0)} saves: matched {chg(a.get('matched_change'))}, E_alt/E_match "
                 f"{chg(a.get('alt_ratio_change'))}, ee {chg(a.get('ee_change'))} -> {a['recommendation']} ({a['reason']})")
    return line


def main() -> None:
    """Back-fill HOM for every saved checkpoint of a finished run (single GPU)."""
    from .common import load_cfg, make_dataset, make_policy

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training output dir with checkpoints/<step>/pretrained_model")
    ap.add_argument("--max_episodes", type=int, default=50, help="held-out episodes (task-stratified); 0 = all")
    ap.add_argument("--out_name", default="heldout_monitor.jsonl")
    ap.add_argument("--steps", default="", help="comma-separated steps to evaluate (default: every saved step)")
    args, overrides = ap.parse_known_args()
    ckpts = sorted(p for p in Path(args.run, "checkpoints").glob("*/pretrained_model") if p.parent.name.isdigit())
    if args.steps:
        wanted = {int(s) for s in args.steps.split(",")}
        ckpts = [p for p in ckpts if int(p.parent.name) in wanted]
    cfg = load_cfg(ckpts[-1], overrides)
    cfg.dataset.heldout_monitor_max_episodes = args.max_episodes
    cfg.dataset.heldout_monitor_full_every = 0
    monitor = HeldoutMonitor(make_dataset(cfg, split="train"))
    out = Path(args.run) / args.out_name
    history = [json.loads(line) for line in out.read_text().splitlines()] if out.is_file() else []
    with out.open("a") as f:
        for ck in ckpts:
            m = monitor(make_policy(load_cfg(ck, overrides), ck), None, int(ck.parent.name))
            m["advice"] = trend_advice(history, m, int(ck.parent.name) >= int(cfg.policy.scheduler_decay_steps))
            history.append(m)
            f.write(json.dumps({**m, "source": "offline"}) + "\n")
            print(ck.parent.name, log_line(m), flush=True)
    print(out)


if __name__ == "__main__":
    main()
