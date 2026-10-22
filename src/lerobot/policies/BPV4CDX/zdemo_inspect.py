"""Inside the demo encoder: how does the Temporal Q-Former read the demo, and what does Z_demo encode? (diagnostic)

The encoder never sees the target progress (segment_query is added later in the Qwen prefix), so Z_demo should be a
time-ordered encoding / summary of the WHOLE demo rather than a selection of some frames.

Q-Former attention (attention weights of the 2 cross-attention and 2 self-attention layers, head mean):
  cross  query slot (32 slots x 4 queries) -> key slot / camera / spatial token (480 = 32 x 3 x 5);
         own-slot mass (x uniform = 32 x mass), mass within +-2 slots, normalized entropy of the key-slot
         distribution per query; coverage = attention received by each key slot summed over all queries
         (x average slot), reported as a profile over the trajectory and aligned to grasp / place
  self   query slot -> query slot mixing (own-slot mass, entropy)
Z_demo content (episode-level 5-fold linear probes on the held-out set; targets demeaned per task):
  local   Z tokens of slot i (mean of its 4 tokens) -> robot state at that slot's frame       (time-local encoding)
  global  Z tokens of slot i -> state at the first gripper close and at the last frame        (summary of the demo)
  per slot i, R^2 for the trained encoder, a randomly initialised one, and the pooled raw ViT features of slot i;
  token geometry: mean cosine between the 128 tokens, cosine between neighbouring slots, effective rank.

Usage: python -m lerobot.policies.BPV4CDX.zdemo_inspect --ckpt <pretrained_model> [--episodes 0] --out <dir> --name <run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path

from .common import collate, load_cfg, make_dataset, make_policy, reproducibility, to_device, write_result
from .gate_g0 import first_close
from .paths import task_name
from .profiles import ANCHORS, plot_profiles, profile
from .saliency_zdemo import first_reopen


class QFormerRecorder:
    """Records head-averaged attention weights of every cross / self attention of the demo encoder."""

    def __init__(self, encoder) -> None:
        self.mods = []
        for li, layer in enumerate(encoder.layers):
            self.mods.append((f"cross{li}", layer.cross))
            if layer.self_attn is not None:
                self.mods.append((f"self{li}", layer.self_attn))
        self.weights: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for name, mod in self.mods:
            orig = mod.forward

            def fwd(*a, _orig=orig, _name=name, **kw):
                kw["need_weights"], kw["average_attn_weights"] = True, True
                out, w = _orig(*a, **kw)
                self.weights[_name] = w.float().cpu()
                return out, w

            mod.forward = fwd
        return self

    def __exit__(self, *exc):
        for _, mod in self.mods:
            del mod.forward
        return False


def entropy_norm(p: np.ndarray) -> np.ndarray:
    p = p / np.maximum(p.sum(-1, keepdims=True), 1e-12)
    return -(p * np.log(np.maximum(p, 1e-12))).sum(-1) / np.log(p.shape[-1])


@torch.no_grad()
def encode(policy, ds, idx: list[int]) -> tuple[np.ndarray, dict]:
    """-> Z [N, 128, D] and per-episode Q-Former attention summaries."""
    enc = policy.model.demo_encoder
    zs, att = [], {"cross_slot": [], "cross_cam": [], "cross_space": [], "self_slot": []}
    for s in range(0, len(idx), 16):
        batch = to_device(collate([ds.demo(i) for i in idx[s : s + 16]]))
        with QFormerRecorder(enc) as rec:
            z = enc(batch["demo.feats"].float(), batch["demo.time"], batch["demo.camera_valid"])
        zs.append(z.float().cpu())
        b, k, c, p = batch["demo.feats"].shape[:4]
        q = enc.per_slot
        for name, w in rec.weights.items():
            if name.startswith("cross"):
                w = w.reshape(b, k, q, k, c, p)  # query slot, query, key slot, camera, spatial
                att["cross_slot"].append((name, w.sum((4, 5)).mean(2).numpy()))   # [b, 32, 32]
                att["cross_cam"].append((name, w.sum((3, 5)).mean((1, 2)).numpy()))  # [b, 3]
                att["cross_space"].append((name, w.sum((3, 4)).mean((1, 2)).numpy()))  # [b, P]
            else:
                w = w.reshape(b, k, q, k, q)
                att["self_slot"].append((name, w.sum(4).mean(2).numpy()))  # [b, 32, 32]
    out = {}
    for key, items in att.items():
        names = sorted({n for n, _ in items})
        out[key] = {n: np.concatenate([a for m, a in items if m == n]) for n in names}
    return torch.cat(zs).numpy(), out


def slot_summary(m: np.ndarray) -> dict:
    """m [N, 32 query slots, 32 key slots] (rows sum to 1)."""
    k = m.shape[-1]
    eye = np.eye(k, dtype=bool)
    near = np.abs(np.arange(k)[:, None] - np.arange(k)[None]) <= 2
    received = m.sum(1) / k  # [N, 32], mean over query slots; uniform = 1 / 32
    return {"own_slot_x_uniform": float(m[:, eye].mean() * k), "within_2_slots": float(m[:, near].sum() / (m.shape[0] * k)),
            "within_2_slots_uniform": float(near.sum() / k / k), "entropy_norm": float(entropy_norm(m).mean()),
            "coverage_min_x_uniform": float((received.min(1) * k).mean()),
            "coverage_frac_slots_ge_half": float(((received * k) >= 0.5).mean()),
            "matrix_mean": (m.mean(0)).round(4).tolist()}


def probe_r2(x: np.ndarray, y: np.ndarray) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13)))
    pred = cross_val_predict(model, x, y, cv=KFold(5, shuffle=True, random_state=0))
    ss = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((pred - y) ** 2).sum() / max(ss, 1e-12))


def demean(y: np.ndarray, group: np.ndarray) -> np.ndarray:
    y = y.astype(np.float64).copy()
    for g in np.unique(group):
        y[group == g] -= y[group == g].mean(0)
    return y


def geometry(z: np.ndarray) -> dict:
    zn = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-12)
    cos = np.einsum("nid,njd->nij", zn, zn)
    k = z.shape[1]
    off = ~np.eye(k, dtype=bool)
    slot = zn.reshape(z.shape[0], 32, -1, z.shape[-1]).mean(2)
    slot = slot / np.maximum(np.linalg.norm(slot, axis=-1, keepdims=True), 1e-12)
    neigh = np.einsum("nid,nid->ni", slot[:, :-1], slot[:, 1:]).mean()
    far = np.einsum("nd,nd->n", slot[:, 0], slot[:, -1]).mean()
    sv = np.linalg.svd(z - z.mean(1, keepdims=True), compute_uv=False)
    pr = (sv ** 2).sum(-1) ** 2 / np.maximum((sv ** 4).sum(-1), 1e-12)
    return {"mean_token_cos": float(cos[:, off].mean()), "neighbour_slot_cos": float(neigh),
            "first_last_slot_cos": float(far), "effective_rank": float(pr.mean()), "tokens": int(k)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--episodes", type=int, default=0, help="held-out episodes (0 = all 750)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    ds = make_dataset(cfg, split="val", segment_mode="fixed")
    rng = np.random.default_rng(0)
    idx = sorted(rng.choice(len(ds), size=args.episodes, replace=False).tolist()) if args.episodes else list(range(len(ds)))
    trained = make_policy(cfg, args.ckpt)
    torch.manual_seed(0)
    rand = make_policy(cfg, None)

    tasks = np.array([task_name(ds.repos[ds.items[i][0]].repo) for i in idx])
    frames, local, grasp_cfg, final, times, events = [], [], [], [], [], []
    for i in idx:
        ri, ep = ds.items[i]
        st = ds.repos[ri].states[ep]
        rec = torch.load(cache_path(ds.cfg.cache_dir, ds.repos[ri].repo, ep), map_location="cpu", weights_only=False)
        f = rec["frames"].numpy()
        frames.append(f)
        times.append(rec["time_ratio"].numpy())
        local.append(st[f, :14])
        fc = first_close(st)
        grasp_cfg.append(st[fc[1], :14] if fc else st[len(st) // 2, :14])
        final.append(st[-1, :14])
        pl = first_reopen(st, fc[0], fc[1]) if fc else None
        events.append((fc[1] / max(1, len(st) - 1) if fc else None, pl / max(1, len(st) - 1) if pl is not None else None))
    local, grasp_cfg, final = np.stack(local), np.stack(grasp_cfg), np.stack(final)

    res = {"episodes": len(idx)}
    z_tr, att_tr = encode(trained, ds, idx)
    z_rd, att_rd = encode(rand, ds, idx)
    for label, att in (("trained", att_tr), ("random_init", att_rd)):
        r = {}
        for name, m in att["cross_slot"].items():
            r[name] = slot_summary(m)
            r[name]["camera"] = att["cross_cam"][name].mean(0).round(4).tolist()
            r[name]["spatial"] = att["cross_space"][name].mean(0).round(4).tolist()
        for name, m in att["self_slot"].items():
            r[name] = slot_summary(m)
        res[f"qformer_{label}"] = r
    res["geometry"] = {"trained": geometry(z_tr), "random_init": geometry(z_rd)}

    # coverage of the trajectory: attention received per key slot, aligned like the other profiles
    bins = np.linspace(-1, 1, 21)
    cov_profiles = {}
    for label, att in (("trained", att_tr), ("random init", att_rd)):
        m = att["cross_slot"][sorted(att["cross_slot"])[-1]]  # last cross-attention layer
        received = m.sum(1) / m.shape[-1]  # [N, 32]
        per = {a: [] for a in ANCHORS}
        for n in range(len(idx)):
            t = times[n]
            anchors = {"segment": 0.5, "grasp": events[n][0], "place": events[n][1]}
            for a in ANCHORS:
                if anchors[a] is None:
                    continue
                d = t - anchors[a]
                ib = np.digitize(d, bins) - 1
                ok = (ib >= 0) & (ib < len(bins) - 1)
                mass, count = np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
                np.add.at(mass, ib[ok], received[n][ok])
                np.add.at(count, ib[ok], 1)
                per[a].append((mass, count))
        cov_profiles[label] = {a: profile(per[a], bins, min_pairs=30) for a in ANCHORS}
    res["coverage_profiles"] = cov_profiles

    # probes per slot
    pooled = np.stack([ds.demo(i)["demo.feats"].float().mean((1, 2)).numpy() for i in idx])  # [N, 32, D] raw ViT
    feats = {"z_trained": z_tr.reshape(len(idx), 32, -1, z_tr.shape[-1]).mean(2),
             "z_random": z_rd.reshape(len(idx), 32, -1, z_rd.shape[-1]).mean(2), "vit_slot": pooled}
    probes = {}
    for fname, x in feats.items():
        rows = {"local_state": [], "grasp_config": [], "final_state": []}
        for s in range(32):
            rows["local_state"].append(probe_r2(x[:, s], demean(local[:, s], tasks)))
            rows["grasp_config"].append(probe_r2(x[:, s], demean(grasp_cfg, tasks)))
            rows["final_state"].append(probe_r2(x[:, s], demean(final, tasks)))
        probes[fname] = {k: np.round(v, 4).tolist() for k, v in rows.items()}
        print(fname, {k: round(float(np.mean(v)), 3) for k, v in rows.items()}, flush=True)
    res["slot_probes"] = probes

    out = Path(args.out)
    res["figures"] = figures(out / "figures", args.name, res, att_tr, att_rd, cov_profiles)
    res["reproducibility"] = reproducibility(cfg, args.ckpt, trained, ds)
    print(write_result(out, args.name, res))
    for label in ("trained", "random_init"):
        print(label, {n: {k: v for k, v in s.items() if k not in ("matrix_mean",)} for n, s in res[f"qformer_{label}"].items()})
    print(res["geometry"])


def figures(out: Path, name: str, res: dict, att_tr: dict, att_rd: dict, cov: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = []
    # 1) query slot x key slot attention (cross layers, self layers), trained vs random
    names = sorted(att_tr["cross_slot"]) + sorted(att_tr["self_slot"])
    fig, axes = plt.subplots(2, len(names), figsize=(3.4 * len(names), 6.6))
    for col, n in enumerate(names):
        for row, (label, att) in enumerate((("trained", att_tr), ("random init", att_rd))):
            m = (att["cross_slot"] if n.startswith("cross") else att["self_slot"])[n].mean(0)
            ax = axes[row, col]
            ax.imshow(m * m.shape[-1], cmap="magma", vmin=0, vmax=max(3.0, float((m * m.shape[-1]).max())))
            ax.set_title(f"{label}: {n}", fontsize=8)
            ax.set_xlabel("key slot", fontsize=7)
            ax.set_ylabel("query slot", fontsize=7)
    fig.suptitle(f"{name}: Q-Former attention between slots (x uniform; diagonal = own slot)", fontsize=10)
    fig.tight_layout()
    f = out / f"{name}_qformer_slots.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    # 2) coverage of the trajectory
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    titles = {"segment": "slot time - 0.5 (trajectory middle)", "grasp": "slot time - grasp", "place": "slot time - place"}
    for ax, a in zip(axes, ANCHORS):
        plot_profiles(ax, {label: p[a] for label, p in cov.items()}, f"attention received per key slot ({titles[a]})")
        ax.set_xlabel("normalized time")
    axes[0].set_ylabel("received attention (x average slot)")
    fig.tight_layout()
    f = out / f"{name}_coverage.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    # 3) slot probes
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, target in zip(axes, ("local_state", "grasp_config", "final_state")):
        for fname, p in res["slot_probes"].items():
            ax.plot(range(32), p[target], marker="o", ms=2, label=fname)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"slot token -> {target} (R^2, task-demeaned)", fontsize=9)
        ax.set_xlabel("slot")
        ax.legend(fontsize=7)
    axes[0].set_ylabel("R^2 (5-fold, episodes)")
    fig.tight_layout()
    f = out / f"{name}_slot_probes.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    return files


if __name__ == "__main__":
    main()
