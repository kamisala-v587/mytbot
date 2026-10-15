#!/usr/bin/env python3
"""BP encoder 内部：query 对 BP 图像 / state 的 cross-attention，以及换图像 / 换 state 后输出的变化。

Usage:
  python encoder_attn.py --config config_encoder.yaml [--gpus 0,1]
  python encoder_attn.py --config config_encoder.yaml --analyze-only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bpv3_encoder_probe"))

from attn_probe import DataPool, load_cfg, run_workers  # noqa: E402


def encoder_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(cfg["manifest"]).read_text().splitlines() if line.strip()]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    tasks = sorted(by_task)
    rng = random.Random(int(cfg.get("seed", 0)))
    plan = []
    for task in tasks:
        for row in by_task[task][: int(cfg["episodes_per_task"])]:
            wrong = rng.choice([t for t in tasks if t != task])
            plan.append({
                "task": task, "repo": row["repo"], "episode": row["episode"], "progress": 0.0,
                "other_episode": row["episode"], "wrong_task": wrong, "wrong_repo": by_task[wrong][0]["repo"],
                "wrong_episode": by_task[wrong][0]["episode"],
            })
    return plan


class CrossCapture:
    """Record cross-attention of the chunk compressor (memory = [image tokens..., state, action...])."""

    def __init__(self, chunk_encoder):
        self.records: list[dict[str, torch.Tensor]] = []
        self.n_img = 0
        for layer in chunk_encoder.compressor_layers:
            layer.forward = self._wrap(layer)

    def _wrap(self, block):
        def forward(queries, memory, memory_valid):
            q = block.cross_norm(queries)
            m = block.memory_norm(memory)
            attn = block.cross_attn
            cross, weights = attn(q, m, m, key_padding_mask=~memory_valid, need_weights=True, average_attn_weights=False)
            self.records.append(self._reduce(attn, weights.float(), m.float()))
            queries = queries + cross
            s = block.self_norm(queries)
            self_attended, _ = block.self_attn(s, s, s, need_weights=False)
            queries = queries + self_attended
            return queries + block.ff(block.ff_norm(queries))

        return forward

    def _reduce(self, attn, weights: torch.Tensor, memory: torch.Tensor) -> dict[str, torch.Tensor]:
        # weights: [N, H, Q, M]; memory: [N, M, D]
        n_img = self.n_img
        heads, dim = attn.num_heads, attn.embed_dim
        w_v = attn.in_proj_weight[2 * dim :].float()
        b_v = attn.in_proj_bias[2 * dim :].float()
        v = (memory @ w_v.T + b_v).view(memory.shape[0], memory.shape[1], heads, dim // heads).transpose(1, 2)
        out_w = attn.out_proj.weight.float()
        segs = {"img": slice(0, n_img), "state": slice(n_img, n_img + 1)}
        mass = torch.stack([weights[..., s].sum(-1) for s in segs.values()], -1)  # [N,H,Q,2]
        contrib = []
        for s in segs.values():
            part = (weights[..., s] @ v[:, :, s]).transpose(1, 2).reshape(weights.shape[0], weights.shape[2], dim)
            contrib.append((part @ out_w.T).norm(dim=-1))  # [N,Q]
        return {
            "mass": mass.cpu(),
            "contrib": torch.stack(contrib, -1).cpu(),
            "img_map": weights[..., :n_img].mean(dim=(1, 2)).cpu(),  # [N, n_img]
            "img_map_per_query": weights[..., :n_img].mean(dim=1).cpu(),  # [N,Q,n_img]
        }


def swap(prompt: dict, donor: dict, part: str) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in prompt.items()}
    if part == "state":
        out["state"] = donor["state"]
        out["state_is_available"] = donor["state_is_available"]
    elif part == "image":
        out["bp_pixel_values"] = donor["bp_pixel_values"]
        out["bp_image_grid_thw"] = donor["bp_image_grid_thw"]
        out["image_masks"] = donor["image_masks"]
    elif part == "no_state":
        out["state_is_available"] = torch.zeros_like(prompt["state_is_available"])
    else:
        raise ValueError(part)
    return out


def run_checkpoint(cfg: dict[str, Any], ck: dict[str, str], out: Path) -> None:
    from probe import load_policy, move_to_device

    import lerobot.policies.BPVAv3.configuration_bpvav3  # noqa: F401
    from torch.utils.data import default_collate

    path = out / "shards" / f"{ck['name']}.pt"
    if path.is_file():
        return
    policy, pcfg, device = load_policy(ck["path"])
    model = policy.model
    plan = encoder_plan(cfg)
    pool = DataPool(cfg, pcfg, plan)
    chunk_encoder = model.bp_obs_encoder.chunk_encoder
    cap = CrossCapture(chunk_encoder)
    raw_out: list[torch.Tensor] = []
    chunk_encoder.register_forward_hook(lambda mod, inp, output: raw_out.append(output.chunk_tokens.float()))
    pos_emb = model.bp_obs_encoder.chunk_position_embedding.weight.detach().float()

    def encode(prompt: dict, record: bool):
        cap.records, raw_out[:] = [], []
        batch = move_to_device(default_collate([prompt]), device)
        with torch.inference_mode():
            tokens, mask = model.encode_bp_prompt(batch)
        recs = cap.records if record else None
        return raw_out[0][0].cpu(), tokens[0].float().cpu(), mask[0].cpu(), recs

    results = []
    for n, item in enumerate(plan):
        prompt = pool.prompt(item["repo"], item["episode"])
        wrong = pool.prompt(item["wrong_repo"], item["wrong_episode"])
        start, end = pool.episode(item["repo"], item["episode"])
        cap.n_img = int(cfg.get("tokens_per_image", 64))
        raw, tokens, mask, recs = encode(prompt, True)
        rec = {**item, "length": end - start, "raw": raw.half(), "mask": mask,
               "attn": {k: torch.stack([r[k] for r in recs]) for k in recs[0]}}  # [layer, chunks, ...]
        rec["pos_norm"] = float(pos_emb.norm(dim=-1).mean())
        for part in ("state", "image", "no_state"):
            donor = wrong if part != "no_state" else prompt
            r2, _, _, _ = encode(swap(prompt, donor, part), False)
            rec[f"delta_{part}"] = ((r2 - raw).norm(dim=-1) / raw.norm(dim=-1).clamp_min(1e-6)).mean(-1)  # [chunks]
        if n < int(cfg.get("render_examples", 0)):
            offsets = np.linspace(0, end - start - 1, raw.shape[0]).round().astype(int)
            rec["frames"] = torch.stack([pool.ds[item["repo"]][start + int(o)]["observation.images.image0"][0] for o in offsets]).half()
        results.append(rec)
        if (n + 1) % 20 == 0:
            print(f"  [{ck['name']}] {n + 1}/{len(plan)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint": ck, "pos_emb": pos_emb.cpu(), "results": results}, path)
    print(f"[done] {path}", flush=True)


# ---------------------------------------------------------------- analysis


def analyze(cfg: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [ck["name"] for ck in cfg["checkpoints"] if (out / "shards" / f"{ck['name']}.pt").is_file()]
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    summary = {}
    for name in names:
        data = torch.load(out / "shards" / f"{name}.pt", map_location="cpu", weights_only=False)
        res = data["results"]
        mass = torch.stack([r["attn"]["mass"] for r in res])  # [N, L, C, H, Q, 2]
        contrib = torch.stack([r["attn"]["contrib"] for r in res])  # [N, L, C, Q, 2]
        maps = torch.stack([r["attn"]["img_map"] for r in res])  # [N, L, C, 64]
        qmaps = torch.stack([r["attn"]["img_map_per_query"] for r in res])  # [N, L, C, Q, 64]
        raw = torch.stack([r["raw"].float() for r in res])  # [N, C, Q, D]
        s: dict[str, Any] = {}
        s["state_mass_by_layer"] = mass[..., 1].mean(dim=(0, 2, 3, 4)).tolist()
        s["state_mass_by_layer_head"] = mass[..., 1].mean(dim=(0, 2, 4)).tolist()
        s["state_mass_by_query"] = mass[..., 1].mean(dim=(0, 2, 3)).tolist()
        c = contrib.mean(dim=(0, 2, 3))  # [L, 2]
        s["state_contrib_share_by_layer"] = (c[:, 1] / c.sum(-1)).tolist()
        # 图像注意力是否依赖内容：同一 query 在不同样本 / 不同关键帧上的空间分布有多相似
        flat = maps.reshape(maps.shape[0] * maps.shape[2], maps.shape[1], -1)  # [N*C, L, 64]
        sims = []
        for layer in range(flat.shape[1]):
            m = F.normalize(flat[:, layer], dim=-1)
            sim = m @ m.T
            sims.append(float((sim.sum() - sim.diag().sum()) / (sim.numel() - sim.shape[0])))
        s["img_map_cross_sample_cos"] = sims
        ent = -(maps.clamp_min(1e-12) / maps.sum(-1, keepdim=True) * (maps.clamp_min(1e-12) / maps.sum(-1, keepdim=True)).log()).sum(-1)
        s["img_map_entropy_ratio"] = (ent.mean(dim=(0, 2)) / np.log(maps.shape[-1])).tolist()
        # query 分工：不同 query 的空间图相似度
        qm = F.normalize(qmaps, dim=-1)
        qsim = (qm @ qm.transpose(-1, -2))  # [N,L,C,Q,Q]
        q = qsim.shape[-1]
        s["query_map_mutual_cos"] = ((qsim.sum(dim=(-1, -2)) - q) / (q * q - q)).mean(dim=(0, 2)).tolist()
        # 关键帧之间输出有多相似（位置编码之前）
        pooled = F.normalize(raw.mean(dim=2), dim=-1)  # [N, C, D]
        csim = pooled @ pooled.transpose(-1, -2)
        cn = csim.shape[-1]
        s["chunk_mutual_cos"] = float(((csim.sum(dim=(-1, -2)) - cn) / (cn * cn - cn)).mean())
        center = F.normalize(raw.mean(dim=(0, 1, 2)), dim=-1)  # 全局均值方向
        s["chunk_cos_to_global_mean"] = float((pooled @ center).mean())
        s["content_norm"] = float(raw.mean(dim=2).norm(dim=-1).mean())
        pos = data["pos_emb"]
        s["pos_norm"] = float(pos.norm(dim=-1).mean())
        pn = F.normalize(pos, dim=-1)
        s["pos_mutual_cos"] = float(((pn @ pn.T).sum() - pos.shape[0]) / (pos.shape[0] ** 2 - pos.shape[0]))
        # 位置编码之后每个关键帧 token 的范数（VLM 实际看到的）
        s["chunk_content_norm_by_index"] = raw.mean(dim=2).norm(dim=-1).mean(0).tolist()
        for part in ("state", "image", "no_state"):
            d = torch.stack([r[f"delta_{part}"] for r in res])
            s[f"delta_{part}"] = float(d.mean())
            s[f"delta_{part}_by_chunk"] = d.mean(0).tolist()
        summary[name] = s

        examples = [r for r in res if "frames" in r]
        for k, r in enumerate(examples[: int(cfg.get("render_examples", 0))]):
            frames = r["frames"].float()
            amap = r["attn"]["img_map"]  # [L, C, 64]
            n_layers, n_chunks = amap.shape[0], amap.shape[1]
            fig, axes = plt.subplots(n_layers + 1, n_chunks, figsize=(2.2 * n_chunks, 2.3 * (n_layers + 1)))
            for c in range(n_chunks):
                img = frames[c].permute(1, 2, 0).numpy()
                img = (img - img.min()) / max(img.max() - img.min(), 1e-6)
                axes[0, c].imshow(img)
                axes[0, c].set_title(f"kf {c}  state {float(r['attn']['mass'][:, c, ..., 1].mean()):.2f}", fontsize=7)
                for layer in range(n_layers):
                    grid = amap[layer, c].reshape(8, 8).numpy()
                    axes[layer + 1, c].imshow(img)
                    axes[layer + 1, c].imshow(grid, cmap="jet", alpha=0.5, extent=(0, img.shape[1], img.shape[0], 0), interpolation="bilinear")
                    if c == 0:
                        axes[layer + 1, c].set_ylabel(f"layer {layer}")
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            fig.suptitle(f"{name}: {r['task']} ep{r['episode']} (row1 keyframes; rows2+ query->image attention)", fontsize=9)
            fig.tight_layout()
            fig.savefig(plots / f"enc_attn_{name}_{k}_{r['task']}.png", dpi=110)
            plt.close(fig)

    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        print(f"== {name}")
        for key in ("state_mass_by_layer", "state_contrib_share_by_layer", "img_map_cross_sample_cos", "img_map_entropy_ratio",
                    "query_map_mutual_cos", "chunk_mutual_cos", "chunk_cos_to_global_mean", "content_norm", "pos_norm",
                    "pos_mutual_cos", "delta_state", "delta_image", "delta_no_state"):
            v = s[key]
            print(f"  {key}: {[round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)}")
        print(f"  delta_image_by_chunk: {[round(x, 3) for x in s['delta_image_by_chunk']]}")
        print(f"  state_mass_by_query: {[[round(x, 3) for x in row] for row in s['state_mass_by_query']]}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpus", default="")
    parser.add_argument("--worker", default=None)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    cfg = load_cfg(args.config)
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    ckpts = {ck["name"]: ck for ck in cfg["checkpoints"]}
    if args.worker:
        run_checkpoint(cfg, ckpts[args.worker], out)
        return 0
    if not args.analyze_only:
        pending = [n for n in ckpts if not (out / "shards" / f"{n}.pt").is_file()]
        gpus = [g for g in args.gpus.split(",") if g.strip()]
        if gpus:
            run_workers(cfg, pending, gpus, out, script=Path(__file__).resolve())
        else:
            for n in pending:
                run_checkpoint(cfg, ckpts[n], out)
    analyze(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
