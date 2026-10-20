"""Linear probes and compressor attention on the FIRST keyframe (arm still at its home pose, so only the scene varies).

Question: does the BP encoder encode where the object is and where it must go, rather than the arm trajectory?
Targets (forward kinematics, robot base frame): grasp point (first gripper close) and place point (next opening).
Targets are residuals after removing the per (task, grasp arm) mean, i.e. the within-task variation of object /
target position; R^2 is measured on held-out episodes. Ridge regression, alpha by 5-fold CV on the train episodes.

Attention: per task, the head-camera patches whose ViT features vary most across episodes at keyframe 0 (top 10%)
mark where objects are (background and home-pose arm do not vary). We report the share of compressor
cross-attention on those patches (uniform = 10%).

Usage: python -m lerobot.policies.BPV4BPEn.probe_eval --labels labels.json --out report.json \
         --ckpt main10k=<pretrained_model> --ckpt nomask10k=<...>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.BPV4BPEn.build_feature_cache import cache_path
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import BPChunkCompressor, BPTrajectoryResampler, sinusoidal_encoding

HOLDOUT = {"clean_50": 5, "randomized_100": 10}
DIM = 1024


def episodes(repo_file: str):
    out = []
    for repo in [l.strip() for l in Path(repo_file).read_text().splitlines() if l.strip()]:
        n = int(json.loads(Path(repo, "meta", "info.json").read_text())["total_episodes"])
        hold = next((v for k, v in HOLDOUT.items() if k in repo), 0)
        out += [(repo, ep, ep >= n - hold) for ep in range(n)]
    return out


class FirstKeyframe(torch.utils.data.Dataset):
    def __init__(self, items, cache_dir):
        self.items, self.cache_dir = items, cache_dir

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        repo, ep, _ = self.items[i]
        rec = torch.load(cache_path(self.cache_dir, repo, ep), map_location="cpu", weights_only=False)
        return {"feats": rec["feats"][:1].float(), "time": rec["time_ratio"][:1], "index": i}


def load_modules(ckpt: str | None, cfg):
    torch.manual_seed(0)
    enc, res = BPChunkCompressor(cfg, DIM), BPTrajectoryResampler(cfg, DIM)
    if ckpt:
        with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f:
            for module, prefix in ((enc, "model.bp_obs_encoder."), (res, "model.bp_resampler.")):
                state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
                module.load_state_dict(state, strict=True)
    return enc.cuda().eval().float(), res.cuda().eval().float()


@torch.no_grad()
def compressor_with_attention(enc: BPChunkCompressor, feats: torch.Tensor):
    """feats [B, 1, C, T, Dv] -> tokens [B, Q, D], attention of the last layer averaged over heads/queries [B, C*T]."""
    b, k, c, t, _ = feats.shape
    x = enc.input_proj(feats) + enc.camera_embedding[None, None, :, None]
    x = x.reshape(b * k, c * t, -1)
    q = enc.queries[None].expand(b * k, -1, -1)
    attn = None
    for layer in enc.layers:
        kvn = layer.cross_norm_kv(x)
        out, w = layer.cross(layer.cross_norm_q(q), kvn, kvn, need_weights=True, average_attn_weights=True)
        attn = w.mean(1)
        q = q + out
        qn = layer.self_norm(q)
        q = q + layer.self_attn(qn, qn, qn, need_weights=False)[0]
        q = q + layer.ff(layer.ff_norm(q))
    return enc.output_proj(q), attn


@torch.no_grad()
def features(loader, models: dict):
    """'.mean' = token mean (no spatial layout), '.tok' = all tokens concatenated (keeps which token / patch)."""
    feats = {"vit.mean": [], "vit_head.tok": [], "vit_all.tok": []}
    for name in models:
        for s in ("enc.mean", "enc.tok", "res.mean", "res.tok", "attn"):
            feats[f"{name}.{s}"] = []
    for batch in loader:
        x = batch["feats"].cuda()  # [B, 1, C, T, Dv]
        b = x.shape[0]
        feats["vit.mean"].append(x[:, 0].mean(2).reshape(b, -1).cpu())
        feats["vit_head.tok"].append(x[:, 0, 0].reshape(b, -1).half().cpu())
        feats["vit_all.tok"].append(x[:, 0].reshape(b, -1).half().cpu())
        for name, (enc, res) in models.items():
            tok, attn = compressor_with_attention(enc, x)
            tok = tok + sinusoidal_encoding(batch["time"].cuda(), tok.shape[-1])[:, :, None].reshape(-1, 1, tok.shape[-1])
            r = res(tok, torch.ones(tok.shape[:2], dtype=torch.bool, device=tok.device))
            feats[f"{name}.enc.mean"].append(tok.mean(1).cpu())
            feats[f"{name}.enc.tok"].append(tok.reshape(b, -1).half().cpu())
            feats[f"{name}.res.mean"].append(r.mean(1).cpu())
            feats[f"{name}.res.tok"].append(r.reshape(b, -1).half().cpu())
            feats[f"{name}.attn"].append(attn.cpu())
    return {k: torch.cat(v).numpy() for k, v in feats.items()}


@torch.no_grad()
def gram(x: np.ndarray, train: np.ndarray, chunk: int = 8192) -> torch.Tensor:
    """Linear kernel of features standardized with train statistics, accumulated over feature chunks: [N, N]."""
    n = len(x)
    g = torch.zeros(n, n, dtype=torch.float64, device="cuda")
    for s in range(0, x.shape[1], chunk):
        c = torch.from_numpy(np.ascontiguousarray(x[:, s:s + chunk])).cuda().float()
        mu, sd = c[train].mean(0), c[train].std(0) + 1e-6
        c = (c - mu) / sd
        g += (c @ c.T).double()
    return g


def ridge_r2(g: torch.Tensor, tr: np.ndarray, va: np.ndarray, y: np.ndarray):
    """Kernel (dual) ridge on a precomputed linear gram; alpha relative to mean diagonal, chosen by 5-fold CV."""
    tri, vai = np.flatnonzero(tr), np.flatnonzero(va)
    yt = torch.tensor(y[tri], dtype=torch.float64, device="cuda")
    ym = yt.mean(0)
    scale = float(g.diagonal()[torch.from_numpy(tri).cuda()].mean())

    def fit_pred(fit_idx, pred_idx, alpha, y_fit):
        fi, pi = torch.from_numpy(fit_idx).cuda(), torch.from_numpy(pred_idx).cuda()
        k = g[fi][:, fi] + alpha * scale * torch.eye(len(fi), dtype=g.dtype, device=g.device)
        coef = torch.linalg.solve(k, y_fit - ym)
        return g[pi][:, fi] @ coef + ym

    folds = np.array_split(np.random.default_rng(0).permutation(len(tri)), 5)
    best, best_err = None, None
    for alpha in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 1e2):
        err = 0.0
        for f in folds:
            m = np.ones(len(tri), bool)
            m[f] = False
            err += float(((fit_pred(tri[m], tri[f], alpha, yt[m]) - yt[f]) ** 2).sum())
        if best_err is None or err < best_err:
            best, best_err = alpha, err
    pred = fit_pred(tri, vai, best, yt).cpu().numpy()
    y_va = y[vai]
    r2 = 1 - ((pred - y_va) ** 2).sum(0) / ((y_va - y_va.mean(0)) ** 2).sum(0)
    return {"r2_mean": float(r2.mean()), "r2_xyz": [float(v) for v in r2], "alpha": best,
            "val_index": vai.tolist(), "pred": pred.round(5).tolist(), "true": y_va.round(5).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo_file", default="/vla/workspace/my_tbot/configs/ds_ids/pro6k/robo_bpv4_bpen_clean50_rand100.txt")
    ap.add_argument("--cache_dir", default="/vla/workspace/cache/bpv4_bpen_vit_base800m_k32_v2")
    ap.add_argument("--ckpt", action="append", default=[], help="name=path to pretrained_model")
    args = ap.parse_args()

    labels = json.loads(Path(args.labels).read_text())
    items = episodes(args.repo_file)
    ckpts = dict(c.split("=", 1) for c in args.ckpt)
    cfg = PreTrainedConfig.from_pretrained(next(iter(ckpts.values())))
    models = {"random": load_modules(None, cfg), **{n: load_modules(p, cfg) for n, p in ckpts.items()}}
    loader = torch.utils.data.DataLoader(FirstKeyframe(items, args.cache_dir), batch_size=64, num_workers=16)
    feats = features(loader, models)

    task = np.array([Path(r).parent.name for r, _, _ in items])
    is_val = np.array([v for _, _, v in items])
    result = {"num_train": int((~is_val).sum()), "num_val": int(is_val.sum()), "probe": {}, "attention": {}}
    targets = {}
    for target in ("grasp", "place"):
        y, arm, has = np.zeros((len(items), 3)), np.array([""] * len(items), dtype=object), np.zeros(len(items), bool)
        for i, (r, e, _) in enumerate(items):
            rec = labels.get(f"{r}|{e}")
            if rec and rec.get(target):
                y[i], arm[i], has[i] = rec[target], rec["arm"], True
        group = np.array([f"{t}|{a}" for t, a in zip(task, arm)])
        means = {g: y[has & ~is_val & (group == g)].mean(0) for g in np.unique(group[has & ~is_val])}
        ok = has & np.array([g in means for g in group])
        resid = y - np.stack([means.get(g, np.zeros(3)) for g in group])
        tr, va = ok & ~is_val, ok & is_val
        targets[target] = (tr, va, resid)
        result["probe"][target] = {"n_train": int(tr.sum()), "n_val": int(va.sum()),
                                   "within_group_std_m": [float(v) for v in resid[va].std(0)]}
    for name, x in feats.items():
        if name.endswith(".attn"):
            continue
        g = gram(x, ~is_val)
        for target, (tr, va, resid) in targets.items():
            result["probe"][target][name] = ridge_r2(g, tr, va, resid)
            print(target, name, result["probe"][target][name]["r2_xyz"], flush=True)
        del g

    # attention on the head-camera patches that vary most across episodes of the same task (train episodes)
    head = slice(0, 64)
    var_mask = np.zeros((len(items), 64), bool)
    raw = feats["vit_head.tok"].reshape(len(items), 64, DIM).astype(np.float32)  # [N, 64, D] head camera
    for t in np.unique(task):
        sel = (task == t) & ~is_val
        score = raw[sel].std(0).mean(-1)  # [64]
        top = np.argsort(score)[-int(round(0.1 * 64)):]
        var_mask[np.ix_(np.flatnonzero(task == t), top)] = True
    for name in models:
        a = feats[f"{name}.attn"]  # [N, C*T]
        head_mass = a[:, head].sum(1)
        on_var = (a[:, head] * var_mask).sum(1) / np.clip(head_mass, 1e-9, None)
        result["attention"][name] = {
            "head_camera_share": float(head_mass[is_val].mean()),
            "varying_patch_share_within_head": float(on_var[is_val].mean()),
            "uniform_reference": float(var_mask[0].mean()),
        }
    result["items"] = [f"{r}|{e}" for r, e, _ in items]
    Path(args.out).write_text(json.dumps(result))
    print(json.dumps(result["attention"], indent=1))


if __name__ == "__main__":
    main()
