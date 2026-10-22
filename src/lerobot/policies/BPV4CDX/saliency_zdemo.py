"""G7-S (not a gate): which demo frame / camera / image position does the reconstructed action depend on?

Saliency = |sum_d grad * input| of the flow loss of one target segment w.r.t. the RAW cached ViT tokens
[32 slots, 3 cameras, 8 x 8 tokens] (the P5 pooling is linear, so the 8 x 8 map is a valid attribution; P64 is
attributed directly). Flow times 0.3 / 0.7 with fixed noise are summed (as in bpv4 vis_rand_attention); every segment
gets its own copy of the demo so gradients do not mix; the map is normalized to sum 1 per segment.

Per segment (fixed starts 0, 50, ...):
  time      slot mass [32]; own-window lift = mass on the slots inside the segment's frame range / their share of
            slots (> 1: reads its own part of the demo); saliency-weighted time offset to the segment centre;
            lift (x 32) of the first slot, last slot, grasp slot (first gripper close) and place slot (first reopen)
  camera    mass per camera (head / left wrist / right wrist)
  space     head-camera 8 x 8 map; with --data probe: lift of the 3 x 3 tokens around the grasped object's initial
            position on the slots up to the grasp, and around its final (placed) position on the slots between grasp
            and place (world-frame poses from scene_info.json projected with the head-camera calibration)
Phases: segment centre before the grasp, between grasp and place (carrying), after the place.
Baseline: the same model with a randomly initialised encoder / expert (saliency sanity check), same episodes.
Time profiles (profiles.py): slot saliency vs slot time minus the segment centre / the grasp / the place, with a 95% CI
over episodes; bins with too few (segment, slot) pairs are left empty.
Figures (--figures N): the three time profiles, per-episode segment x slot heatmaps, head-camera overlays.

Usage: python -m lerobot.policies.BPV4CDX.saliency_zdemo --ckpt <pretrained_model> --data heldout|probe \
         [--probe_repo_file <txt> --labels <labels_probe.json>] [--episodes 100] --out <dir> --name <run> [--figures 3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from lerobot.policies.BPV4BPEn.build_feature_cache import CAMERAS, cache_path
from lerobot.policies.q35.transform_q35 import ACTION_VALID_MASK
from lerobot.utils.constants import ACTION

from .common import load_cfg, make_dataset, make_policy, reproducibility, to_device, write_result
from .dataset import pool_spatial
from .gate_g0 import GRIPPER, first_close
from .paths import task_name
from .profiles import ANCHORS, anchor_times, episode_bins, plot_profiles, profile

TIMES = (0.3, 0.7)
CAM_SHORT = ("head", "left wrist", "right wrist")


def first_reopen(state: np.ndarray, arm: str, after: int) -> int | None:
    opened = state[after:, GRIPPER[arm]] >= 0.5
    idx = np.nonzero(opened)[0]
    return int(after + idx[0]) if len(idx) else None


def project_token(repo: str, ep: int, point) -> int | None:
    """3D world point -> head-camera token index (8 x 8, row-major) of the 224 x 224 padded image (bpv4 convention)."""
    import h5py

    from lerobot.policies.BPV4BPEn.probe_identity import ROBOTWIN_DATA, SOURCE_CONFIG

    src = Path(repo, "meta", "source_episodes.json")
    if src.exists():
        ep = json.loads(src.read_text())["source_episodes"][ep]
    with h5py.File(ROBOTWIN_DATA / task_name(repo) / SOURCE_CONFIG / "data" / f"episode{ep}.hdf5", "r") as f:
        intr, extr = f["observation/head_camera/intrinsic_cv"][0], f["observation/head_camera/extrinsic_cv"][0]
    cam = extr @ np.r_[np.asarray(point[:3], dtype=np.float64), 1.0]
    if cam[2] <= 0:
        return None
    u, v = (intr @ cam)[:2] / cam[2]
    col, row = int(u * 0.7 // 28), int((v * 0.7 + 28) // 28)
    return row * 8 + col if 0 <= col < 8 and 0 <= row < 8 else None


def neighbourhood(token: int) -> np.ndarray:
    r, c = divmod(token, 8)
    m = np.zeros(64, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if 0 <= r + dr < 8 and 0 <= c + dc < 8:
                m[(r + dr) * 8 + c + dc] = True
    return m


def episode_saliency(policy, ds, i: int) -> dict:
    """-> sal [S, 32, 3, 64] (sum 1 per segment), slot times, frames, segment starts / valid steps."""
    model, cfg = policy.model, ds.cfg
    ri, ep = ds.items[i]
    rec = torch.load(cache_path(cfg.cache_dir, ds.repos[ri].repo, ep), map_location="cpu", weights_only=False)
    length = len(ds.repos[ri].states[ep])
    starts = list(range(0, length, cfg.segment_len))
    s = len(starts)
    tgt = ds.target(i, starts, generic=[False] * s)
    batch = to_device({k: v[:, None] for k, v in tgt.items()})  # S trajectories x 1 segment
    raw = rec["feats"].float().cuda()[None].expand(s, -1, -1, -1, -1).clone().requires_grad_(True)  # [S, 32, 3, 64, D]
    valid_cam = rec.get("camera_valid")
    batch["demo.time"] = rec["time_ratio"].float().cuda()[None].expand(s, -1)
    batch["demo.camera_valid"] = (valid_cam.bool() if valid_cam is not None else torch.ones(32, 3, dtype=torch.bool)).cuda()[None].expand(s, -1, -1)
    a = batch[ACTION].shape[-1]
    noise = torch.randn((s, cfg.segment_len, int(policy.config.max_action_dim)), generator=torch.Generator().manual_seed(i)).cuda()
    w = batch[ACTION_VALID_MASK].reshape(s, -1, 1).float()
    with torch.enable_grad():
        for t in TIMES:
            pooled = torch.stack([pool_spatial(raw[j], int(cfg.spatial_tokens)) for j in range(s)])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                flow = model.forward_segments({**batch, "demo.feats": pooled}, noise=noise,
                                              time=torch.full((s,), t, device="cuda"))["flow"]
            loss = ((flow[..., :a].float() * w).sum((1, 2)) / (w.sum((1, 2)) * a).clamp_min(1)).sum()
            loss.backward()
    sal = (raw.grad * raw.detach()).sum(-1).abs()
    sal = (sal / sal.sum((1, 2, 3), keepdim=True).clamp_min(1e-12)).cpu().numpy()
    steps = [min(cfg.segment_len, length - st) for st in starts]
    return {"sal": sal, "time": rec["time_ratio"].numpy(), "frames": rec["frames"].numpy(), "starts": starts,
            "steps": steps, "length": length}


def segment_metrics(e: dict, state: np.ndarray, grasp_tok: int | None, place_tok: int | None,
                    grasp_frame: int | None = None, place_frame: int | None = None) -> list[dict]:
    frames, t, L = e["frames"], e["time"], e["length"]
    if grasp_frame is None:
        fc = first_close(state)
        grasp_frame = fc[1] if fc else None
        place_frame = first_reopen(state, fc[0], fc[1]) if fc else None
    nearest = lambda f: int(np.argmin(np.abs(frames - f)))  # noqa: E731
    g_slot = nearest(grasp_frame) if grasp_frame is not None else None
    p_slot = nearest(place_frame) if place_frame is not None else None
    out = []
    for k, (st, n) in enumerate(zip(e["starts"], e["steps"])):
        sal = e["sal"][k]
        slot = sal.sum((1, 2))
        inside = (frames >= st) & (frames <= st + n - 1)
        if not inside.any():
            inside = np.zeros(32, bool)
            inside[nearest(st + n / 2)] = True
        centre = (st + (n - 1) / 2) / max(1, L - 1)
        mid = st + (n - 1) / 2
        phase = ("pre_grasp" if grasp_frame is None or mid < grasp_frame else
                 "carry" if place_frame is None or mid < place_frame else "post_place")
        m = {"phase": phase, "own_window_lift": float(slot[inside].sum() / inside.mean()),
             "time_offset": float((slot * (t - centre)).sum()), "first_lift": float(slot[0] * 32),
             "last_lift": float(slot[-1] * 32), "camera": sal.sum((0, 2)).tolist(),
             "grasp_lift": float(slot[g_slot] * 32) if g_slot is not None else None,
             "place_lift": float(slot[p_slot] * 32) if p_slot is not None else None,
             "head_map": sal[:, 0].sum(0).tolist()}
        head = sal[:, 0]  # [32, 64]
        if grasp_tok is not None and grasp_frame is not None:
            pre = frames <= grasp_frame
            nb = neighbourhood(grasp_tok)
            if pre.any() and head[pre].sum() > 0:
                m["grasp_point_lift"] = float(head[pre][:, nb].sum() / head[pre].sum() / nb.mean())
        if place_tok is not None and grasp_frame is not None and place_frame is not None:
            mid_slots = (frames >= grasp_frame) & (frames <= place_frame)
            nb = neighbourhood(place_tok)
            if mid_slots.any() and head[mid_slots].sum() > 0:
                m["place_point_lift"] = float(head[mid_slots][:, nb].sum() / head[mid_slots].sum() / nb.mean())
        out.append(m)
    return out


def summarize(rows: list[dict]) -> dict:
    def agg(sel):
        res = {"segments": len(sel)}
        if not sel:
            return res
        for key in ("own_window_lift", "time_offset", "first_lift", "last_lift", "grasp_lift", "place_lift",
                    "grasp_point_lift", "place_point_lift"):
            v = np.array([r[key] for r in sel if r.get(key) is not None], dtype=np.float64)
            if len(v):
                res[key] = {"mean": float(v.mean()), "median": float(np.median(v)),
                            **({"frac_gt_1": float((v > 1).mean())} if key.endswith("lift") else {}), "n": int(len(v))}
        res["camera"] = dict(zip(CAM_SHORT, np.mean([r["camera"] for r in sel], 0).round(4).tolist()))
        return res

    return {"all": agg(rows), **{ph: agg([r for r in rows if r["phase"] == ph]) for ph in ("pre_grasp", "carry", "post_place")}}


def aligned_profiles(eps: dict, metas: dict, bins: np.ndarray) -> dict:
    """Slot-saliency profiles aligned to the segment centre, the first gripper close and the place (with CIs)."""
    per = {a: [] for a in ANCHORS}
    for i, e in eps.items():
        anchors = anchor_times(e["starts"], e["steps"], e["length"], metas[i]["grasp_frame"], metas[i]["place_frame"])
        slot = e["sal"].sum((2, 3))  # [S, 32]
        for a in ANCHORS:
            if anchors[a] is not None:
                per[a].append(episode_bins(e["time"], slot, anchors[a], bins))
    return {a: profile(per[a], bins) for a in ANCHORS}


def figures(out: Path, name: str, ds, eps: dict, metas: dict, profiles: dict, bins: np.ndarray, n: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lerobot.policies.BPV4BPEn.vis_rand_attention import images

    out.mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
    for ax, a, xl in zip(axes, ANCHORS, ("slot time - segment centre", "slot time - first gripper close (grasp)",
                                         "slot time - first reopen (place)")):
        plot_profiles(ax, {label: prof[a] for label, prof in profiles.items()}, f"aligned to {a}")
        ax.set_xlabel(f"{xl} (normalized)")
    axes[0].set_ylabel("saliency per slot (x average slot)")
    fig.suptitle(f"{name}: slot saliency vs time (95% CI over episodes; bins with < 30 pairs omitted)", fontsize=10)
    fig.tight_layout()
    f = out / f"{name}_time_profile.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    for i in list(eps)[:n]:
        e, meta = eps[i], metas[i]
        ri, ep = ds.items[i]
        slot = e["sal"].sum((2, 3))  # [S, 32]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(e["starts"]) + 1.5))
        ax.imshow(slot, aspect="auto", cmap="magma")
        ax.set_yticks(range(len(e["starts"])), [f"{s}" for s in e["starts"]], fontsize=6)
        ax.set_xticks(range(32), [str(int(x)) for x in e["frames"]], fontsize=5, rotation=90)
        for key, color in (("grasp_slot", "cyan"), ("place_slot", "lime")):
            if meta.get(key) is not None:
                ax.axvline(meta[key], color=color, lw=1)
        for k, (st, n_) in enumerate(zip(e["starts"], e["steps"])):
            inside = np.nonzero((e["frames"] >= st) & (e["frames"] <= st + n_ - 1))[0]
            if len(inside):
                ax.add_patch(plt.Rectangle((inside[0] - 0.5, k - 0.5), len(inside), 1, fill=False, ec="white", lw=0.6))
        ax.set_xlabel("slot (source frame); cyan = grasp, green = place, white box = segment's own frames")
        ax.set_ylabel("segment start")
        ax.set_title(f"{task_name(ds.repos[ri].repo)} ep{ep}: slot saliency per target segment", fontsize=8)
        fig.tight_layout()
        f = out / f"{name}_ep{i}_slots.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
        # head-camera overlays at the grasp / place slots for the segment that contains each event
        shows = [(lab, meta[key], meta[fk]) for lab, key, fk in (("grasp", "grasp_slot", "grasp_frame"), ("place", "place_slot", "place_frame"))
                 if meta.get(key) is not None]
        if not shows:
            continue
        imgs = images(ds.repos[ri].repo, ep, [e["frames"][s] for _, s, _ in shows])
        fig, axes = plt.subplots(1, len(shows), figsize=(3.2 * len(shows), 3.4))
        for ax, (lab, s, fr), img in zip(np.atleast_1d(axes), shows, imgs):
            k = int(np.clip(fr // ds.cfg.segment_len, 0, len(e["starts"]) - 1))
            m = e["sal"][k, :, 0].sum(0).reshape(8, 8)  # head map of that segment over all slots
            ax.imshow(img[0])
            ax.imshow(np.kron(m / max(1e-12, m.max()), np.ones((28, 28))), cmap="jet", alpha=0.45)
            for tok_key, mk in (("grasp_tok", "x"), ("place_tok", "+")):
                if meta.get(tok_key) is not None:
                    r, c = divmod(meta[tok_key], 8)
                    ax.plot(c * 28 + 14, r * 28 + 14, mk, color="white", ms=10, mew=2)
            ax.set_title(f"{lab} slot (frame {e['frames'][s]}), segment {e['starts'][k]}", fontsize=7)
            ax.axis("off")
        fig.tight_layout()
        f = out / f"{name}_ep{i}_head.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
    return files


def run(policy, ds, picks, labels, data) -> tuple[list[dict], dict, dict]:
    rows, eps, metas = [], {}, {}
    for n, i in enumerate(picks):
        ri, ep = ds.items[i]
        repo = ds.repos[ri].repo
        state = ds.repos[ri].states[ep]
        e = episode_saliency(policy, ds, i)
        g_tok = p_tok = gf = pf = None
        if data == "probe":
            from lerobot.policies.BPV4BPEn.probe_identity import episode_meta

            lab = labels.get(f"{repo}|{ep}") or {}
            gf, pf = lab.get("grasp_frame"), lab.get("place_frame")
            # world-frame object poses (the label points are FK positions in the robot frame)
            om = episode_meta(repo, ep, json.loads(Path(repo, "meta", "scene_info.json").read_text()))
            g_tok = om["head_token"]
            p_tok = project_token(repo, ep, om["grasp_final"]) if om["grasp_final"] is not None else None
        seg = segment_metrics(e, state, g_tok, p_tok, gf, pf)
        for r in seg:
            r.update({"task": task_name(repo), "episode": int(i)})
        rows += seg
        if gf is None:
            fc = first_close(state)
            gf = fc[1] if fc else None
            pf = first_reopen(state, fc[0], fc[1]) if fc else None
        near = lambda f: int(np.argmin(np.abs(e["frames"] - f))) if f is not None else None  # noqa: E731
        eps[i] = e
        metas[i] = {"grasp_frame": gf, "place_frame": pf, "grasp_slot": near(gf), "place_slot": near(pf),
                    "grasp_tok": g_tok, "place_tok": p_tok}
        if (n + 1) % 20 == 0:
            print(f"{n + 1} / {len(picks)}", flush=True)
    return rows, eps, metas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["heldout", "probe"], default="heldout")
    ap.add_argument("--probe_repo_file", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--episodes", type=int, default=100, help="episodes (fixed-seed random choice; 0 = all)")
    ap.add_argument("--random_baseline", type=int, default=1, help="also run a randomly initialised model")
    ap.add_argument("--random_episodes", type=int, default=0, help="episodes for the random baseline (0 = same as trained)")
    ap.add_argument("--figures", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    labels = {}
    if args.data == "probe":
        if not args.probe_repo_file or not args.labels:
            raise SystemExit("--data probe needs --probe_repo_file and --labels")
        ds = make_dataset(cfg, repo_id_file=args.probe_repo_file, data_root=None, split="all", segment_mode="fixed")
        labels = json.loads(Path(args.labels).read_text())
    else:
        ds = make_dataset(cfg, split="val", segment_mode="fixed")
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(ds), size=min(args.episodes, len(ds)), replace=False).tolist()) if args.episodes else list(range(len(ds)))
    trained = make_policy(cfg, args.ckpt)
    rows, eps, metas = run(trained, ds, picks, labels, args.data)
    res = {"data": args.data, "episodes": len(picks), "segments": len(rows), "trained": summarize(rows),
           "by_task": {t: summarize([r for r in rows if r["task"] == t])["all"] for t in sorted({r["task"] for r in rows})}}
    bins = np.linspace(-1, 1, 21)
    profiles = {"trained": aligned_profiles(eps, metas, bins)}
    head_map = np.mean([r["head_map"] for r in rows], 0)
    res["head_map_mean"] = (head_map / head_map.sum()).reshape(8, 8).round(4).tolist()
    if args.random_baseline:
        torch.manual_seed(0)
        rand = make_policy(cfg, None)
        r_picks = picks[: args.random_episodes] if args.random_episodes else picks
        r_rows, r_eps, r_metas = run(rand, ds, r_picks, labels, args.data)
        res["random_init"] = summarize(r_rows)
        profiles["random init"] = aligned_profiles(r_eps, r_metas, bins)
        del rand
    res["time_profile"] = {"bins": bins.tolist(), **profiles}
    if args.figures:
        res["figures"] = figures(Path(args.out) / "figures", args.name, ds, eps, metas, profiles, bins, args.figures)
    res["reproducibility"] = reproducibility(cfg, args.ckpt, trained, ds)
    print(write_result(args.out, args.name, res))
    print(json.dumps(res["trained"]["all"], indent=1))


if __name__ == "__main__":
    main()
