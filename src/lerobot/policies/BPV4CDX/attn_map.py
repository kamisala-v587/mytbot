"""Attention maps of the Stage A-AE model: which inputs do the tokens attend to? (diagnostic, not a gate)

The 6 full-attention layers of Qwen3.5 (3, 7, 11, 15, 19, 23) are the only layers with an explicit attention matrix
and the only places where prefix and suffix meet (the 18 linear-attention layers mix tokens inside the prefix or the
suffix only). The attention probabilities are recomputed from the same query / key / mask the model uses (softmax of
scaled q k^T + mask, GQA heads expanded, mean over the 8 query heads) without changing the forward pass.

Token groups: BOS | text (valid tokens only) | demo_start | Z_demo (128 = 32 slots x 4) | demo_end | segment_query |
state | action (50). Per layer, for each query group, the attention mass on each key group (rows sum to 1), averaged
over the query tokens of the group and over segments. With expert_context=segment_query, state / action rows can only
see segment_query, state and action: their split shows how much the action expert reads the demo channel
(segment_query) vs the state slot vs the other action tokens. The segment_query row shows what the demo channel
collects from the prefix (image tokens Z_demo vs language vs special tokens). Mass per token (mass / group size)
separates "many tokens" from "strong tokens"; BOS often acts as an attention sink.
Time profiles: segment_query -> Z_demo mass per slot (4 queries per slot), aligned to the segment centre / the grasp /
the place, as in the saliency diagnostic (profiles.py).

Usage: python -m lerobot.policies.BPV4CDX.attn_map --ckpt <pretrained_model> [--data heldout|probe] [--episodes 100] \
         [--generic] --out <dir> --name <run>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import collate, load_cfg, make_dataset, make_policy, reproducibility, to_device, write_result
from .gate_g0 import first_close
from .profiles import ANCHORS, anchor_times, episode_bins, plot_profiles, profile
from .saliency_zdemo import first_reopen

GROUPS = ("BOS", "text", "demo_start", "Z_demo", "demo_end", "segment_query", "state", "action")
FLOW_TIMES = (0.3, 0.7)


class AttentionRecorder:
    """Wraps backbone._run_attention; stores the head-mean attention probabilities of every call."""

    def __init__(self, backbone) -> None:
        self.backbone, self.maps = backbone, []
        self._orig = backbone._run_attention

    def __enter__(self):
        def wrapped(attention, query, key, value, additive_mask):
            q, k = query.float(), key.float()
            if k.shape[1] != q.shape[1]:
                k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            scores = torch.matmul(q, k.transpose(-1, -2)) * attention.scaling
            scores = scores + additive_mask[..., : k.shape[-2]].float()
            self.maps.append(torch.softmax(scores, dim=-1).mean(1).cpu())  # [B, L, L]
            return self._orig(attention, query, key, value, additive_mask)

        self.backbone._run_attention = wrapped
        return self

    def __exit__(self, *exc):
        del self.backbone._run_attention  # back to the class method
        return False


def group_index(cfg, text_valid: np.ndarray, has_demo: bool, horizon: int) -> tuple[list[np.ndarray], int]:
    """Token indices of each group for one sample (text: valid tokens only) and the prefix length."""
    pos, out = 0, {}
    if cfg.prefix_bos:
        out["BOS"], pos = np.array([0]), 1
    n_text = len(text_valid)
    out["text"] = pos + np.flatnonzero(text_valid)
    pos += n_text
    if has_demo:
        out["demo_start"] = np.array([pos])
        out["Z_demo"] = pos + 1 + np.arange(cfg.num_demo_tokens)
        out["demo_end"] = np.array([pos + 1 + cfg.num_demo_tokens])
        pos += cfg.num_demo_tokens + 2
    out["segment_query"] = np.array([pos])
    prefix = pos + 1
    out["state"] = np.array([prefix])
    out["action"] = prefix + 1 + np.arange(horizon)
    return [out.get(g, np.array([], dtype=int)) for g in GROUPS], prefix


@torch.no_grad()
def episode_attention(policy, ds, i: int, generic: bool) -> dict:
    model, cfg = policy.model, policy.config
    ri, ep = ds.items[i]
    length = len(ds.repos[ri].states[ep])
    starts = list(range(0, length, ds.cfg.segment_len))
    s = len(starts)
    item = {**ds.demo(i), **ds.target(i, starts, generic=[generic] * s)}
    batch = to_device(collate([item]))
    horizon = int(cfg.chunk_size)
    noise = torch.randn((s, horizon, int(cfg.max_action_dim)), generator=torch.Generator().manual_seed(i)).cuda()
    maps = None
    for t in FLOW_TIMES:
        with AttentionRecorder(model.core.backbone) as rec, torch.autocast("cuda", dtype=torch.bfloat16):
            model.forward_segments(batch, noise=noise, time=torch.full((s,), t, device="cuda"),
                                   use_demo=model.demo_encoder is not None)
        layer_maps = torch.stack(rec.maps)  # [6, S, L, L]
        maps = layer_maps if maps is None else maps + layer_maps
    maps = (maps / len(FLOW_TIMES)).numpy()
    text_valid = item["text.attention_mask"][0].bool().numpy()
    groups, prefix = group_index(cfg, text_valid, model.demo_encoder is not None, horizon)
    ng = len(GROUPS)
    # mass[layer, query group, key group] averaged over query tokens and segments
    mass = np.full((maps.shape[0], ng, ng), np.nan)
    for qi, qg in enumerate(groups):
        if not len(qg):
            continue
        rows = maps[:, :, qg]  # [6, S, |qg|, L]
        for ki, kg in enumerate(groups):
            if len(kg):
                mass[:, qi, ki] = rows[..., kg].sum(-1).mean((1, 2))
    zslot = None
    if len(groups[GROUPS.index("Z_demo")]):
        segq = groups[GROUPS.index("segment_query")][0]
        z = maps[:, :, segq][..., groups[GROUPS.index("Z_demo")]]  # [6, S, 128]
        z = z.reshape(z.shape[0], s, -1, 4).sum(-1)  # [6, S, 32] per slot
        zslot = z / np.maximum(z.sum(-1, keepdims=True), 1e-12)  # share within Z_demo
    return {"mass": mass, "zslot": zslot, "example": maps[:, 0], "groups": groups, "prefix": prefix,
            "starts": starts, "steps": [min(ds.cfg.segment_len, length - st) for st in starts], "length": length,
            "time": item["demo.time"].numpy(), "sizes": [len(g) for g in groups]}


def figures(out: Path, name: str, res: dict, ex: dict, profiles: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    files = []
    mass = np.array(res["mass_by_layer"], dtype=float)  # [6, G, G]
    layers = res["layers"]
    # 1) group -> group attention mass, per layer + mean
    fig, axes = plt.subplots(1, len(layers) + 1, figsize=(3.3 * (len(layers) + 1), 3.6))
    for ax, m, title in zip(axes, [*mass, np.nanmean(mass, 0)], [f"layer {l}" for l in layers] + ["mean of 6"]):
        ax.imshow(np.nan_to_num(m), vmin=0, vmax=1, cmap="viridis")
        for a in range(len(GROUPS)):
            for b in range(len(GROUPS)):
                if np.isfinite(m[a, b]) and m[a, b] >= 0.01:
                    ax.text(b, a, f"{m[a, b]:.2f}", ha="center", va="center", fontsize=5,
                            color="white" if m[a, b] < 0.5 else "black")
        ax.set_xticks(range(len(GROUPS)), GROUPS, rotation=90, fontsize=6)
        ax.set_yticks(range(len(GROUPS)), GROUPS, fontsize=6)
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("query (row) -> key (column)", fontsize=7)
    fig.suptitle(f"{name}: attention mass from each query group to each key group (rows sum to 1)", fontsize=10)
    fig.tight_layout()
    f = out / f"{name}_group_mass.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    # 2) stacked bars: what segment_query / state / action attend to, per layer
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    for ax, qname in zip(axes, ("segment_query", "state", "action")):
        qi = GROUPS.index(qname)
        bottom = np.zeros(len(layers))
        for ki, kname in enumerate(GROUPS):
            v = np.nan_to_num(mass[:, qi, ki])
            if v.max() < 0.005:
                continue
            ax.bar([str(l) for l in layers], v, bottom=bottom, label=kname)
            bottom += v
        ax.set_title(f"query = {qname}", fontsize=9)
        ax.set_xlabel("full-attention layer")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("attention mass")
    fig.tight_layout()
    f = out / f"{name}_bars.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    # 3) segment_query -> Z_demo per slot, aligned to segment / grasp / place
    if profiles:
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
        for ax, a in zip(axes, ANCHORS):
            plot_profiles(ax, {name: profiles[a]}, f"segment_query -> Z_demo slots, aligned to {a}")
            ax.set_xlabel("slot time - anchor (normalized)")
        axes[0].set_ylabel("attention per slot (x average slot)")
        fig.tight_layout()
        f = out / f"{name}_slot_profile.png"
        fig.savefig(f, dpi=120)
        plt.close(fig)
        files.append(str(f))
    # 4) one full attention map (example episode, first segment, last full-attention layer) with group borders
    m = ex["example"][-1]
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(np.log10(m + 1e-4), cmap="magma", vmin=-4, vmax=0)
    bounds = [0]
    for g in ex["groups"]:
        if len(g):
            bounds.append(int(g.max()) + 1)
    for b in sorted(set(bounds)):
        ax.axhline(b - 0.5, color="white", lw=0.4)
        ax.axvline(b - 0.5, color="white", lw=0.4)
    ticks = [(int(g.min()) + int(g.max())) / 2 for g in ex["groups"] if len(g)]
    labels = [n for n, g in zip(GROUPS, ex["groups"]) if len(g)]
    ax.set_xticks(ticks, labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_title(f"{name}: attention, layer {layers[-1]}, one segment (log10 probability)", fontsize=9)
    fig.tight_layout()
    f = out / f"{name}_example_map.png"
    fig.savefig(f, dpi=120)
    plt.close(fig)
    files.append(str(f))
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", choices=["heldout", "probe"], default="heldout")
    ap.add_argument("--probe_repo_file", default=None)
    ap.add_argument("--episodes", type=int, default=100, help="episodes (fixed-seed random choice; 0 = all)")
    ap.add_argument("--generic", action="store_true", help="generic text instead of the canonical task name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    args, overrides = ap.parse_known_args()
    cfg = load_cfg(args.ckpt, overrides)
    if args.data == "probe":
        ds = make_dataset(cfg, repo_id_file=args.probe_repo_file, data_root=None, split="all", segment_mode="fixed")
    else:
        ds = make_dataset(cfg, split="val", segment_mode="fixed")
    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(ds), size=min(args.episodes, len(ds)), replace=False).tolist()) if args.episodes else list(range(len(ds)))
    policy = make_policy(cfg, args.ckpt)
    layers = [i for i, layer in enumerate(policy.model.core.backbone.vlm_language_model.layers)
              if layer.block_type != "linear_attention"]
    masses, per = [], {a: [] for a in ANCHORS}
    sizes = None
    bins = np.linspace(-1, 1, 21)
    example = None
    for n, i in enumerate(picks):
        e = episode_attention(policy, ds, i, args.generic)
        masses.append(e["mass"])
        sizes = e["sizes"] if sizes is None else sizes
        example = example or e
        if e["zslot"] is not None:
            ri, ep = ds.items[i]
            st = ds.repos[ri].states[ep]
            fc = first_close(st)
            grasp = fc[1] if fc else None
            place = first_reopen(st, fc[0], fc[1]) if fc else None
            anchors = anchor_times(e["starts"], e["steps"], e["length"], grasp, place)
            zs = e["zslot"].mean(0)  # mean over the 6 layers -> [S, 32]
            for a in ANCHORS:
                if anchors[a] is not None:
                    per[a].append(episode_bins(e["time"], zs, anchors[a], bins))
        if (n + 1) % 20 == 0:
            print(f"{n + 1} / {len(picks)}", flush=True)
    mass = np.nanmean(np.stack(masses), 0)  # [6, G, G]
    profiles = {a: profile(per[a], bins) for a in ANCHORS} if per["segment"] else {}
    res = {"data": args.data, "episodes": len(picks), "generic_text": args.generic, "layers": layers,
           "groups": list(GROUPS), "group_size_example": dict(zip(GROUPS, sizes)),
           "mass_by_layer": np.round(mass, 5).tolist(),
           "mass_mean_over_layers": {q: {k: round(float(v), 4) for k, v in zip(GROUPS, row) if np.isfinite(v)}
                                     for q, row in zip(GROUPS, np.nanmean(mass, 0))},
           "slot_profiles": profiles}
    res["figures"] = figures(Path(args.out) / "figures", args.name, res, example, profiles)
    res["reproducibility"] = reproducibility(cfg, args.ckpt, policy, ds)
    print(write_result(args.out, args.name, res))
    for q in ("segment_query", "state", "action", "Z_demo", "text"):
        print(q, res["mass_mean_over_layers"].get(q))


if __name__ == "__main__":
    main()
