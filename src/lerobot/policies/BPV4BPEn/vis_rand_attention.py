"""Where does the policy look inside the BP, as a function of the query progress p? (unseen randomized episodes)

For each episode and a grid of query progress values p, two visibility conditions (none / training-style event mask):
  saliency     |grad x input| of the flow loss w.r.t. the cached ViT tokens of every BP keyframe -> [K, C, 64].
               This covers the whole path (compressor, resampler, VLM, action expert), so it is the p-dependent
               "where the action comes from" map.
  resampler    last-layer cross-attention of the 128 BP latents over the keyframe tokens -> [K] (p-independent:
               the resampler never sees p; only the visible set changes with the mask)
  compressor   last-layer cross-attention of the per-keyframe queries over the 3 x 64 image tokens (p-independent)

Usage: python -m lerobot.policies.BPV4BPEn.vis_rand_attention --ckpt main10k=<pretrained_model> [--ckpt nomask10k=...] \
         --repo <lerobot repo> --episodes 100:120 --out <fig dir> --show 100,101,102
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import CAMERAS, cache_path
from lerobot.policies.BPV4BPEn.configuration_bpv4_bpen import BPV4BPEnDatasetConfig
from lerobot.policies.BPV4BPEn.dataset import BPV4BPEnDataset
from lerobot.policies.BPV4BPEn.keyframes import event_segment
from lerobot.policies.BPV4BPEn.modeling_bpv4_bpen import BPV4BPEnPolicy, sinusoidal_encoding
from lerobot.policies.BPV4BPEn.probe_eval import compressor_with_attention
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION, OBS_STATE

P_GRID = (np.arange(10) + 0.5) / 10
TIMES = (0.3, 0.7)
CAM_SHORT = ("head", "left wrist", "right wrist")


def make_dataset(ckpt: Path, repo: str) -> BPV4BPEnDataset:
    train_cfg = json.loads((ckpt / "train_config.json").read_text())
    d = dict(train_cfg["dataset"])
    known = set(BPV4BPEnDatasetConfig.__dataclass_fields__)
    kwargs = {k: v for k, v in d.items() if k in known and k not in ("type", "image_transforms")}
    kwargs["split"] = "all"
    if isinstance(kwargs.get("mask_window_ratio"), list):
        kwargs["mask_window_ratio"] = tuple(kwargs["mask_window_ratio"])
    cfg = BPV4BPEnDatasetConfig(**kwargs)
    return BPV4BPEnDataset(cfg, SimpleNamespace(dataset=cfg), [repo], chunk_size=int(train_cfg["policy"]["chunk_size"]))


def build_queries(ds: BPV4BPEnDataset, ep: int) -> dict:
    """One trajectory, one query per p in P_GRID, laid out as a batch of B = len(P_GRID) trajectories x 1 query
    (so that every query gets its own copy of the BP features and its own gradient)."""
    cfg, repo = ds.cfg, ds.repos[0]
    rec = torch.load(cache_path(cfg.bp_feature_cache_dir, repo.repo, ep), map_location="cpu", weights_only=False)
    k_max, k_eff = int(cfg.bp_num_keyframes), int(rec["frames"].shape[0])
    feats = torch.zeros((k_max, *rec["feats"].shape[1:]))
    feats[:k_eff] = rec["feats"].float()
    time = torch.zeros(k_max)
    time[:k_eff] = rec["time_ratio"]
    valid = np.arange(k_max) < k_eff
    frames = np.full(k_max, -10**9, dtype=np.int64)
    frames[:k_eff] = rec["frames"].numpy()
    flips, active, length = rec["flips"].numpy(), tuple(rec["active"]), int(rec["length"])
    ids, attn = ds._tokens(0, ep)
    out = {k: [] for k in (OBS_STATE, ACTION, ACTION_VALID_MASK, "vis_event")}
    q_frames = [int(round(p * (length - 1))) for p in P_GRID]
    for f in q_frames:
        q = repo.query(ep, f, ds.chunk_size)
        for key in (OBS_STATE, ACTION, ACTION_VALID_MASK):
            out[key].append(q[key][None])
        s, e = event_segment(flips, active, f)
        hide = (frames > s) & (frames <= e) & valid
        if valid.sum() - hide.sum() < int(cfg.mask_min_visible_keyframes):
            hide[:] = False  # the training-time fallback would be a random window; keep it unmasked instead
        out["vis_event"].append(torch.from_numpy(valid & ~hide))
    b = len(q_frames)
    batch = {key: torch.stack(v) for key, v in out.items() if key != "vis_event"}
    batch.update({
        "observation.input_ids": ids[None, None].expand(b, 1, -1).clone(),
        "observation.attention_mask": attn[None, None].expand(b, 1, -1).clone(),
        "bp.feats": feats[None].expand(b, *feats.shape).clone(),
        "bp.time": time[None].expand(b, -1).clone(),
        "bp.kf_valid": torch.from_numpy(valid)[None].expand(b, -1).clone(),
        "query.progress": torch.tensor([f / max(1, length - 1) for f in q_frames], dtype=torch.float32)[:, None],
        "query.progress_drop": torch.zeros(b, 1, dtype=torch.bool),
    })
    vis = {"none": torch.from_numpy(valid)[None].expand(b, -1).clone(), "event": torch.stack(out["vis_event"])}
    meta = {"frames": frames[:k_eff], "flips": flips, "length": length, "q_frames": q_frames, "k_eff": k_eff,
            "time": rec["time_ratio"].numpy()}
    return batch, vis, meta


def saliency(policy: BPV4BPEnPolicy, batch: dict, visible: torch.Tensor) -> np.ndarray:
    """[B, K, C, T] |grad x input| of the flow loss, normalised per query over all tokens."""
    model = policy.model
    feats = batch["bp.feats"].clone().requires_grad_(True)
    b = feats.shape[0]
    noise = torch.randn((b, batch[ACTION].shape[2], policy.config.max_action_dim),
                        generator=torch.Generator().manual_seed(0)).to(feats.device)
    valid = batch[ACTION_VALID_MASK].reshape(b, -1)
    raw = batch[ACTION].shape[-1]
    for t in TIMES:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            flow = model.forward_queries({**batch, "bp.feats": feats}, visible[:, None], noise=noise,
                                         time=torch.full((b,), t, device=feats.device))["flow"]
        flow = flow[..., :raw].float()
        w = valid[..., None].float()
        loss = ((flow * w).sum((1, 2)) / (w.sum((1, 2)) * raw).clamp_min(1)).sum()
        loss.backward()
    s = (feats.grad * feats.detach()).sum(-1).abs()  # [B, K, C, T]
    s = s * visible[:, :, None, None]
    return (s / s.sum((1, 2, 3), keepdim=True).clamp_min(1e-12)).cpu().numpy()


@torch.no_grad()
def resampler_attention(model, feats: torch.Tensor, time: torch.Tensor, visible: torch.Tensor) -> np.ndarray:
    """[B, K]: last-layer cross-attention of the BP latents, averaged over heads and latents, summed per keyframe."""
    enc, res = model.bp_obs_encoder, model.bp_resampler
    content = enc(feats.to(enc.input_proj.weight.dtype))
    content = content + sinusoidal_encoding(time, content.shape[-1]).to(content.dtype)[:, :, None]
    b, k, q, _ = content.shape
    tokens = content.reshape(b, k * q, -1)
    key_vis = visible[:, :, None].expand(-1, -1, q).reshape(b, k * q)
    lat = res.queries[None].expand(b, -1, -1)
    attn = None
    for layer in res.layers:
        kvn = layer.cross_norm_kv(tokens)
        out, w = layer.cross(layer.cross_norm_q(lat), kvn, kvn, key_padding_mask=~key_vis, need_weights=True,
                             average_attn_weights=True)
        attn = w.mean(1)  # [B, K*Q]
        lat = lat + out
        if layer.self_attn is not None:
            ln = layer.self_norm(lat)
            lat = lat + layer.self_attn(ln, ln, ln, need_weights=False)[0]
        lat = lat + layer.ff(layer.ff_norm(lat))
    return attn.reshape(b, k, q).sum(-1).float().cpu().numpy()


def images(repo: str, ep: int, frames) -> np.ndarray:
    """[K, C, 224, 224, 3] keyframe images (resized with pad, as fed to the ViT)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.transforms.utils import resize_with_pad

    ds = LeRobotDataset(repo, video_backend="torchcodec")
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    out = []
    for f in frames:
        sample = ds[start + int(f)]
        out.append([resize_with_pad(sample[c][None], 224, 224)[0].permute(1, 2, 0).clamp(0, 1).numpy() for c in CAMERAS])
    return np.array(out)


def nearest_kf(frames: np.ndarray, f: int) -> int:
    return int(np.argmin(np.abs(frames - f)))


def fig_temporal(name, ep, meta, imgs, sal, res_attn, out: Path) -> None:
    k = meta["k_eff"]
    frames, flips = meta["frames"], set(meta["flips"].tolist())
    fig = plt.figure(figsize=(k * 0.62, 12))
    gs = fig.add_gridspec(5, k, height_ratios=[0.9, 4, 4, 0.7, 0.7], hspace=0.35, wspace=0.04)
    for j in range(k):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(imgs[j, 0])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(str(frames[j]), fontsize=6)
        if frames[j] in flips:
            for s in ax.spines.values():
                s.set_edgecolor("red"), s.set_linewidth(2)
    rows = [("none", "no mask"), ("event", "event mask (as in training)")]
    for r, (cond, title) in enumerate(rows):
        ax = fig.add_subplot(gs[1 + r, :])
        m = sal[cond].sum((2, 3))[:, :k]  # [P, K]
        shown = np.ma.masked_where(m <= 0, m)
        im = ax.imshow(shown, aspect="auto", cmap="magma", vmin=0, vmax=max(0.15, float(m.max())))
        ax.set_facecolor("#9fb6cd")
        for i, f in enumerate(meta["q_frames"]):
            ax.plot(nearest_kf(frames, f), i, marker="o", ms=6, mfc="none", mec="cyan", mew=1.5)
        ax.set_yticks(range(len(P_GRID)), [f"p={p:.2f}" for p in P_GRID], fontsize=8)
        ax.set_xticks(range(k), [str(f) for f in frames], fontsize=6, rotation=90)
        ax.set_title(f"{title}: saliency share per keyframe for each query p (cyan circle = keyframe nearest to the "
                     f"query frame, blue-grey = hidden)", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.015, pad=0.005)
    for r, (cond, title) in enumerate(rows):
        ax = fig.add_subplot(gs[3 + r, :])
        a = res_attn[cond][:1, :k] if cond == "none" else res_attn[cond][:, :k].mean(0, keepdims=True)
        ax.imshow(a, aspect="auto", cmap="viridis")
        ax.set_yticks([0], ["resampler"], fontsize=8), ax.set_xticks([])
        ax.set_title(f"resampler attention over keyframes ({title}{', averaged over p' if cond == 'event' else ''}; "
                     f"independent of p)", fontsize=8)
    fig.suptitle(f"{name} | place_empty_cup randomized ep{ep}: which BP keyframes the action depends on "
                 f"(top: head camera, red = gripper open/close keyframe)", fontsize=11)
    fig.savefig(out, dpi=90, bbox_inches="tight")
    plt.close(fig)


def overlay(ax, img, m, title, vmax=None):
    ax.imshow(img)
    if m is not None:
        v = np.clip(m.reshape(8, 8) / ((vmax if vmax is not None else float(m.max())) + 1e-12), 0, 1)
        heat = np.zeros((8, 8, 4))
        heat[..., 0], heat[..., 1], heat[..., 3] = 1.0, 0.1, 0.75 * v
        ax.imshow(heat, extent=(0, 224, 224, 0), interpolation="nearest")
        r, c = divmod(int(np.argmax(m)), 8)
        ax.add_patch(plt.Rectangle((c * 28, r * 28), 28, 28, fill=False, ec="yellow", lw=1.5))
    ax.set_title(title, fontsize=7)
    ax.set_xticks([]), ax.set_yticks([])


def fig_spatial(name, ep, meta, imgs, sal, comp_attn, out: Path, cond="event", ps=(0, 4, 8), top=3) -> None:
    """Per selected p: the most salient keyframes, all three cameras, with the per-token saliency overlaid."""
    k = meta["k_eff"]
    frames = meta["frames"]
    fig, axes = plt.subplots(len(ps) + 1, top * 3, figsize=(top * 3 * 2.7, (len(ps) + 1) * 3.4))
    for r, pi in enumerate(ps):
        s = sal[cond][pi, :k]  # [K, C, T]
        best = np.argsort(s.sum((1, 2)))[::-1][:top]
        vmax = float(s[best].max())
        for j, kf in enumerate(best):
            for c in range(3):
                overlay(axes[r, j * 3 + c], imgs[kf, c], s[kf, c],
                        f"p={P_GRID[pi]:.2f} kf {frames[kf]} {CAM_SHORT[c]}\nshare {s[kf, c].sum():.2f}", vmax)
                print(f"{name} ep{ep} p={P_GRID[pi]:.2f} kf {frames[kf]} {CAM_SHORT[c]} share {s[kf, c].sum():.3f} "
                      f"max token {divmod(int(np.argmax(s[kf, c])), 8)}", flush=True)
    # compressor attention (p-independent) on the gripper-event keyframes
    ev = [j for j, f in enumerate(frames) if f in set(meta["flips"].tolist())][:top] or list(range(top))
    for j, kf in enumerate(ev[:top]):
        a = comp_attn[kf]
        for c in range(3):
            overlay(axes[-1, j * 3 + c], imgs[kf, c], a[c * 64:(c + 1) * 64],
                    f"compressor attn kf {frames[kf]} {CAM_SHORT[c]}\nshare {a[c * 64:(c + 1) * 64].sum():.2f}",
                    float(a.max()))
    for j in range(len(ev[:top]) * 3, top * 3):
        axes[-1, j].axis("off")
    fig.suptitle(f"{name} | ep{ep}, {('event mask' if cond == 'event' else 'no mask')}: top-{top} salient keyframes "
                 f"per query p (rows 1-{len(ps)}; red opacity = saliency per 8x8 token, shared scale per row, yellow box = "
                 f"max token); last row: compressor attention on the gripper-event keyframes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.0)
    fig.savefig(out, dpi=95)
    plt.close(fig)


def anchor_keyframes(meta) -> list[tuple[str, int]]:
    """Six fixed, meaningful keyframes: first, grasp, right after the grasp, middle of the carry, release, last."""
    frames, flips, k = meta["frames"], [int(f) for f in meta["flips"]], meta["k_eff"]
    near = lambda f: int(np.argmin(np.abs(frames - f)))  # noqa: E731
    grasp = near(flips[0]) if flips else k // 3
    release = near(flips[1]) if len(flips) > 1 else min(k - 2, grasp + (k - grasp) // 2)
    return [("first frame", 0), ("grasp", grasp), ("just after grasp", min(grasp + 1, k - 1)),
            ("carrying", (grasp + release) // 2), ("release", release), ("last frame", k - 1)]


def smooth_overlay(ax, img, m, vmax):
    from PIL import Image

    ax.imshow(img)
    if m is not None:
        v = np.clip(m.reshape(8, 8) / (vmax + 1e-12), 0, 1).astype(np.float32)
        up = np.asarray(Image.fromarray(v).resize((224, 224), Image.BILINEAR))
        rgba = plt.get_cmap("inferno")(up)
        rgba[..., 3] = 0.85 * up
        ax.imshow(rgba)
        for rank, (idx, col) in enumerate(zip(np.argsort(m.reshape(-1))[::-1][:3], ("cyan", "lime", "white")), 1):
            r, c = divmod(int(idx), 8)
            ax.add_patch(plt.Rectangle((c * 28, r * 28), 28, 28, fill=False, ec=col, lw=1.5))
            ax.text(c * 28 + 2, r * 28 + 10, str(rank), color=col, fontsize=7, fontweight="bold")
    ax.set_xticks([]), ax.set_yticks([])


def fig_spatial_clean(name, ep, meta, imgs, sal, comp_attn, out: Path, cond: str) -> None:
    """Rows: queries p = 0.05 / 0.45 / 0.85 and the (p-independent) compressor attention; columns: six fixed
    keyframes; head camera only. Heat = saliency per head-camera token, one colour scale per row."""
    anchors = anchor_keyframes(meta)
    frames = meta["frames"]
    rows = [(0, "p = 0.05\n(going to grasp)"), (4, "p = 0.45\n(carrying)"), (8, "p = 0.85\n(about to release)")]
    fig, axes = plt.subplots(len(rows) + 1, len(anchors), figsize=(len(anchors) * 2.6, (len(rows) + 1) * 2.75))
    for r, (pi, label) in enumerate(rows):
        s = sal[cond][pi]  # [K, C, T], sums to 1 over all keyframes / cameras / tokens
        vmax = max(float(s[kf, 0].max()) for _, kf in anchors)
        q_frame = meta["q_frames"][pi]
        for j, (title, kf) in enumerate(anchors):
            ax = axes[r, j]
            hidden = s[kf].sum() == 0
            smooth_overlay(ax, imgs[kf, 0] * (0.35 if hidden else 1.0), None if hidden else s[kf, 0], vmax)
            text = "hidden" if hidden else f"all cams {100 * s[kf].sum():.1f}%  head {100 * s[kf, 0].sum():.1f}%"
            ax.set_title(text, fontsize=8, color="gray" if hidden else "black")
            if r == 0:
                ax.text(0.5, 1.28, f"{title} (frame {frames[kf]})", transform=ax.transAxes, ha="center", fontsize=9,
                        fontweight="bold")
        axes[r, 0].set_ylabel(f"{label}\nquery frame {q_frame}", fontsize=9)
    for j, (_, kf) in enumerate(anchors):
        a = comp_attn[kf][:64]
        smooth_overlay(axes[-1, j], imgs[kf, 0], a, float(max(comp_attn[k][:64].max() for _, k in anchors)))
        axes[-1, j].set_title(f"head share {100 * a.sum():.0f}% of this frame", fontsize=8)
    axes[-1, 0].set_ylabel("BP encoder itself\n(compressor attention,\nindependent of p)", fontsize=9)
    fig.suptitle(f"{name} | place_empty_cup randomized ep{ep} | {'no mask' if cond == 'none' else 'event mask (as in training)'}"
                 f" | head camera only. Rows 1-3: what the predicted action depends on (gradient saliency); "
                 f"brighter = more; boxes 1 / 2 / 3 = strongest tokens (cyan / green / white)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=1.2)
    fig.savefig(out, dpi=100)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="name=path")
    ap.add_argument("--repo", default="/vla/workspace/data/robotwin2.0/place_empty_cup/aloha-agilex_randomized_500")
    ap.add_argument("--episodes", default="100:120")
    ap.add_argument("--show", default="100,101,102")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(v) for v in args.episodes.split(":"))
    show = {int(v) for v in args.show.split(",") if v}
    device = torch.device("cuda")

    summary = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        policy = BPV4BPEnPolicy.from_pretrained(path).to(device).eval()
        for p in policy.parameters():
            p.requires_grad_(False)
        ds = make_dataset(Path(path), args.repo)
        rel_bins = np.linspace(-1, 1, 21)
        agg = {c: {"rel": np.zeros(len(rel_bins) - 1), "cam": np.zeros((len(P_GRID), 3)), "n": 0} for c in ("none", "event")}
        for ep in range(lo, hi):
            batch, vis, meta = build_queries(ds, ep)
            batch = {k: v.to(device) for k, v in batch.items()}
            vis = {c: v.to(device) for c, v in vis.items()}
            sal = {c: saliency(policy, batch, v) for c, v in vis.items()}
            res_attn = {c: resampler_attention(policy.model, batch["bp.feats"], batch["bp.time"], v) for c, v in vis.items()}
            for c in sal:
                s = sal[c][:, : meta["k_eff"]].sum((2, 3))  # [P, K]
                rel = meta["time"][None] - np.array([f / max(1, meta["length"] - 1) for f in meta["q_frames"]])[:, None]
                agg[c]["rel"] += np.histogram(rel.reshape(-1), bins=rel_bins, weights=s.reshape(-1))[0]
                agg[c]["cam"] += sal[c].sum((1, 3))
                agg[c]["n"] += 1
            if ep in show:
                kf_feats = batch["bp.feats"][:1, : meta["k_eff"]].reshape(meta["k_eff"], 1, 3, 64, -1)
                enc = policy.model.bp_obs_encoder
                comp = compressor_with_attention(enc, kf_feats.to(enc.input_proj.weight.dtype))[1].float().cpu().numpy()
                imgs = images(args.repo, ep, meta["frames"])
                fig_temporal(name, ep, meta, imgs, sal, res_attn, out / f"rand_cup_{name}_ep{ep}_temporal.png")
                fig_spatial(name, ep, meta, imgs, sal, comp, out / f"rand_cup_{name}_ep{ep}_spatial.png")
                for c in ("none", "event"):
                    fig_spatial_clean(name, ep, meta, imgs, sal, comp, out / f"rand_cup_{name}_ep{ep}_head_{c}.png", c)
            print(name, "ep", ep, flush=True)
        summary[name] = {c: {"rel_centers": ((rel_bins[1:] + rel_bins[:-1]) / 2).tolist(),
                             "rel": (agg[c]["rel"] / agg[c]["n"] / len(P_GRID)).tolist(),
                             "cam": (agg[c]["cam"] / agg[c]["n"]).tolist()} for c in agg}
        del policy
        torch.cuda.empty_cache()

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for name, d in summary.items():
        for c, st in (("none", "-"), ("event", "--")):
            axes[0].plot(d[c]["rel_centers"], d[c]["rel"], st, marker="o", ms=3, label=f"{name}, {c}")
    axes[0].axvline(0, color="gray", lw=0.8)
    axes[0].set_xlabel("keyframe time − query progress p"), axes[0].set_ylabel("saliency share per bin")
    axes[0].set_title("Where in time the action depends on (avg over episodes and p)", fontsize=10), axes[0].legend(fontsize=7)
    for ax, c in zip(axes[1:], ("none", "event")):
        for name, d in summary.items():
            cam = np.array(d[c]["cam"])
            for ci, col in enumerate(("C0", "C1", "C2")):
                ax.plot(P_GRID, cam[:, ci], "-" if name == list(summary)[0] else "--", color=col, marker="o", ms=3,
                        label=f"{name}: {CAM_SHORT[ci]}")
        ax.set_xlabel("query progress p"), ax.set_ylabel("saliency share"), ax.set_ylim(0, 1)
        ax.set_title(f"saliency share per camera ({c} mask)", fontsize=10), ax.legend(fontsize=7)
    fig.suptitle(f"place_empty_cup randomized, episodes {lo}-{hi - 1} (unseen)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "rand_cup_summary.png", dpi=110)
    (out / "rand_cup_summary.json").write_text(json.dumps(summary))
    print("done")


if __name__ == "__main__":
    main()
