#!/usr/bin/env python3
"""Action-expert attention over prefix segments (current images / text / BP) and offline BP-swap flow loss.

Usage:
  python attn_probe.py --config config.yaml [--gpus 0,1,2,3]      # extract every checkpoint, then analyze
  python attn_probe.py --config config.yaml --analyze-only
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bpv3_encoder_probe"))

SEGMENTS = ("img_head", "img_wrist1", "img_wrist2", "text", "bp", "state", "action")
CONDITIONS = ("self", "other", "wrong", "mask")
ATTN_CONDITIONS = ("self", "other", "wrong")
# 通路拆分：BP 都用同任务另一条轨迹，只屏蔽某条注意力通路。
# direct：state/action 不能直接看 BP；prefix：当前图像/文本不能看 BP；both：两者都屏蔽。
ROUTE_CONDITIONS = ("other", "block_direct", "block_prefix", "block_both", "mask")
ROUTE_PROMPT = {"block_direct": "other", "block_prefix": "other", "block_both": "other"}
ROUTE_BLOCK = {"block_direct": "direct", "block_prefix": "prefix", "block_both": "both"}
# 模态拆分：BP 用同任务另一条轨迹，只把图像或 state 换成错任务的。
MODALITY_CONDITIONS = ("other", "img_wrong", "state_wrong", "wrong", "mask")
MODALITY_SWAP = {
    "img_wrong": ("bp_pixel_values", "bp_image_grid_thw", "image_masks"),
    "state_wrong": ("state", "state_is_available"),
}
# 时序拆分：BP 用同任务另一条轨迹，在加 chunk 位置编码之前改写各 chunk 的内容。
# reverse：关键帧内容倒序；chunk_mean：每个 chunk 都换成 8 个 chunk 的均值（去掉时序）；
# task_mean：换成同任务多条示教在每个 chunk 上的均值（只剩任务身份 + 平均模板）。
TEMPORAL_CONDITIONS = ("other", "reverse", "chunk_mean", "task_mean", "wrong", "mask")
TEMPORAL_EDITS = ("reverse", "chunk_mean", "task_mean")


def load_cfg(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path.resolve())
    return cfg


# ---------------------------------------------------------------- samples


def build_plan(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(cfg["manifest"]).read_text().splitlines() if line.strip()]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    tasks = sorted(by_task)[: int(cfg.get("max_tasks", 10**6))]
    rng = random.Random(int(cfg.get("seed", 0)))
    n_cur = int(cfg["episodes_per_task"])
    n_mean = int(cfg.get("task_mean_episodes", 0))
    plan = []
    for k, task in enumerate(tasks):
        mean_eps = [r["episode"] for r in by_task[task]][:n_mean]
        eps = [r["episode"] for r in by_task[task]][: n_cur + 1]
        for i in range(n_cur):
            wrong_task = rng.choice([t for t in tasks if t != task])
            for progress in cfg["progress"]:
                plan.append(
                    {
                        "task": task,
                        "repo": by_task[task][0]["repo"],
                        "episode": eps[i],
                        "progress": float(progress),
                        "other_episode": eps[i + 1],
                        "wrong_task": wrong_task,
                        "wrong_repo": by_task[wrong_task][0]["repo"],
                        "wrong_episode": by_task[wrong_task][0]["episode"],
                        "mean_episodes": [e for e in mean_eps if e != eps[i]],
                        "noise_seed": len(plan),
                    }
                )
    return plan


class DataPool:
    """One BPVAv3 behavior dataset per repo, restricted to the episodes the plan touches."""

    def __init__(self, cfg: dict[str, Any], pcfg: Any, plan: list[dict[str, Any]]):
        from probe import action_delta_timestamps, dataset_cfg

        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.BPVAv3.dataset import make_bpvav3_behavior_dataset

        need: dict[str, set[int]] = {}
        for p in plan:
            need.setdefault(p["repo"], set()).update({p["episode"], p["other_episode"], *p.get("mean_episodes", [])})
            need.setdefault(p["wrong_repo"], set()).add(p["wrong_episode"])
        from types import SimpleNamespace

        from lerobot.datasets.factory import _load_external_stats_for_dataset

        # 训练时 use_external_stats=true：state/action 归一化用 norm_stats 下的聚合 stats，而不是 repo 自身 stats。
        stats_cfg = SimpleNamespace(dataset=SimpleNamespace(
            use_external_stats=True, external_stats_path=None,
            external_stats_root=cfg["external_stats_root"], action_mode="delta",
        ))
        probe_cfg = dict(cfg)
        acs = int(getattr(pcfg, "bp_action_chunk_size", 50))
        self.ds = {}
        for repo, eps in need.items():
            eps = sorted(eps)
            kw = dict(episodes=eps, delta_timestamps=action_delta_timestamps(repo, acs),
                      video_backend=cfg.get("video_backend", "torchcodec"), skip_video_file_validation=True)
            current_ds, prompt_ds = LeRobotDataset(repo, **kw), LeRobotDataset(repo, **kw)
            _load_external_stats_for_dataset(stats_cfg, current_ds)
            prompt_ds.meta.stats.update(current_ds.meta.stats)
            self.ds[repo] = make_bpvav3_behavior_dataset(
                current_ds, prompt_ds, dataset_cfg(probe_cfg, repo, pcfg), bp_action_chunk_size=acs,
            )

    def episode(self, repo: str, episode: int) -> tuple[int, int]:
        start, end, _ = self.ds[repo]._episode_ranges[int(episode)]
        return int(start), int(end)

    def current(self, repo: str, episode: int, progress: float) -> tuple[dict, int, int]:
        start, end = self.episode(repo, episode)
        frame = int(round(progress * (end - start - 1)))
        return self.ds[repo][start + frame], frame, end - start

    def prompt(self, repo: str, episode: int) -> dict:
        start, _ = self.episode(repo, episode)
        return self.ds[repo][start]["behavior_prompt"]


# ---------------------------------------------------------------- attention capture


class Capture:
    def __init__(self, backbone):
        self.backbone = backbone
        self.original = backbone._run_attention
        self.active = False
        self.block: str | None = None
        self.spans: dict[str, Any] = {}
        self.records: list[dict[str, torch.Tensor]] = []
        backbone._run_attention = self._run_attention

    def _blocked(self, additive_mask: torch.Tensor) -> torch.Tensor:
        s = self.spans
        mask = additive_mask.clone()
        low = torch.finfo(mask.dtype).min
        bp = s["bp_index"]
        if self.block in ("direct", "both"):
            suffix = torch.cat([s["state"], s["action"]])
            mask[:, :, suffix[:, None], bp[None, :]] = low
        if self.block in ("prefix", "both"):
            prefix = torch.arange(int(bp[0]), device=mask.device)
            mask[:, :, prefix[:, None], bp[None, :]] = low
        return mask

    def _run_attention(self, attention, query, key, value, additive_mask):
        if self.block is not None:
            additive_mask = self._blocked(additive_mask)
        if not self.active:
            return self.original(attention, query, key, value, additive_mask)
        groups = query.shape[1] // key.shape[1]
        k = key.repeat_interleave(groups, dim=1).float()
        v = value.repeat_interleave(groups, dim=1).float()
        scores = query.float() @ k.transpose(-1, -2) * attention.scaling
        scores = scores + additive_mask[..., : key.shape[-2]].float()
        probs = scores.softmax(dim=-1)
        self.records.append(self._reduce(probs, v))
        return (probs @ v).to(query.dtype).transpose(1, 2).contiguous()

    def _reduce(self, probs: torch.Tensor, v: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.spans
        act = s["action"]
        pa = probs[0, :, act]  # [H, Qa, K]
        mass = torch.stack([pa[..., s[name]].sum(-1).mean(-1) for name in SEGMENTS], dim=-1)  # [H, S]
        contrib = torch.stack(
            [(pa[..., s[name]] @ v[0, :, s[name]]).norm(dim=-1).mean(-1) for name in SEGMENTS], dim=-1
        )  # [H, S]
        bp_idx = s["bp_index"]
        chunk = pa[..., bp_idx].mean(1).reshape(pa.shape[0], -1, s["queries_per_chunk"]).sum(-1)  # [H, C]
        head_img = pa[..., s["img_head"]].mean(dim=(0, 1))  # [P]
        out = {"mass": mass, "contrib": contrib, "chunk": chunk, "head_img": head_img}
        img_q = torch.cat([torch.arange(probs.shape[-1], device=probs.device)[s[n]] for n in ("img_head", "img_wrist1", "img_wrist2")])
        out["img_to_bp"] = probs[0][:, img_q][..., s["bp"]].sum(-1).mean(-1)  # [H]
        out["bp_to_img"] = torch.stack(
            [probs[0][:, s["bp"]][..., s[n]].sum(-1).mean(-1) for n in ("img_head", "img_wrist1", "img_wrist2")], -1
        ).sum(-1)  # [H]
        return {key: val.float().cpu() for key, val in out.items()}


def segment_spans(sample: dict, bp_mask: torch.Tensor, chunk_size: int, num_chunks: int) -> dict[str, Any]:
    ids_type = sample["observation.mm_token_type_ids"][0].cpu()
    valid = sample["observation.attention_mask"][0].cpu().bool()
    n_prefix = ids_type.shape[0]
    image_pos = torch.nonzero(ids_type > 0).flatten()
    per = image_pos.numel() // 3
    img = [image_pos[i * per : (i + 1) * per] for i in range(3)]
    is_img = torch.zeros(n_prefix, dtype=torch.bool)
    is_img[image_pos] = True
    text = torch.nonzero(~is_img & valid).flatten()
    n_bp = bp_mask.shape[-1]
    bp_valid = bp_mask[0].cpu().bool()
    bp_index = torch.arange(n_prefix, n_prefix + n_bp)
    return {
        "img_head": img[0], "img_wrist1": img[1], "img_wrist2": img[2], "text": text,
        "bp": bp_index[bp_valid], "bp_index": bp_index,
        "state": torch.tensor([n_prefix + n_bp]),
        "action": torch.arange(n_prefix + n_bp + 1, n_prefix + n_bp + 1 + chunk_size),
        "queries_per_chunk": n_bp // num_chunks,
        "image_grid": int(round(per ** 0.5)),
    }


class ChunkEdit:
    """Forward hook on the BP chunk encoder: rewrites chunk contents [B, C, Q, D] before the position embedding."""

    def __init__(self, chunk_encoder):
        self.edit: str | None = None
        self.task_mean: torch.Tensor | None = None
        self.last: tuple[torch.Tensor, torch.Tensor] | None = None
        chunk_encoder.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        tokens, mask = output.chunk_tokens, output.mask
        if mask is None:
            mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        self.last = (tokens.detach(), mask.detach())
        if self.edit is None:
            return output
        tokens = tokens.clone()
        valid = torch.nonzero(mask[0]).flatten()
        if self.edit == "reverse":
            tokens[0, valid] = tokens[0, valid.flip(0)]
        elif self.edit == "chunk_mean":
            tokens[0, valid] = tokens[0, valid].mean(0, keepdim=True)
        elif self.edit == "task_mean":
            tokens[0, valid] = self.task_mean[valid].to(tokens.dtype)
        output.chunk_tokens = tokens
        return output


def task_mean_tokens(model, edit: ChunkEdit, prompts: list[dict], device) -> torch.Tensor:
    from probe import move_to_device
    from torch.utils.data import default_collate

    edit.edit = None
    total, count = None, None
    for prompt in prompts:
        with torch.inference_mode():
            model.encode_bp_prompt(move_to_device(default_collate([prompt]), device))
        tokens, mask = edit.last
        tokens, mask = tokens[0].float(), mask[0].float()[:, None, None]
        total = tokens * mask if total is None else total + tokens * mask
        count = mask if count is None else count + mask
    return total / count.clamp_min(1)


# ---------------------------------------------------------------- per-checkpoint worker


def run_checkpoint(cfg: dict[str, Any], ck: dict[str, str], out: Path) -> None:
    from probe import load_policy, move_to_device

    import lerobot.policies.BPVAv3.configuration_bpvav3  # noqa: F401  registers bpvav3
    from torch.utils.data import default_collate

    path = out / "shards" / f"{ck['name']}.pt"
    if path.is_file():
        print(f"[skip] {path}", flush=True)
        return
    policy, pcfg, device = load_policy(ck["path"])
    model = policy.model
    plan = build_plan(cfg)
    pool = DataPool(cfg, pcfg, plan)
    cap = Capture(model.core.backbone)
    num_chunks = int(pcfg.bp_num_chunks)
    adim = int(pcfg.output_features["action"].shape[0])
    times = [float(t) for t in cfg["times"]]
    ablation = cfg.get("route_ablation") or cfg.get("modality_ablation") or cfg.get("temporal_ablation")
    conditions = (
        MODALITY_CONDITIONS if cfg.get("modality_ablation")
        else ROUTE_CONDITIONS if cfg.get("route_ablation")
        else TEMPORAL_CONDITIONS if cfg.get("temporal_ablation") else CONDITIONS
    )
    edit = ChunkEdit(model.bp_obs_encoder.chunk_encoder) if cfg.get("temporal_ablation") else None
    mean_cache: dict[tuple, torch.Tensor] = {}
    results = []
    for n, item in enumerate(plan):
        if edit is not None:
            key = (item["repo"], tuple(item["mean_episodes"]))
            if key not in mean_cache:
                mean_cache[key] = task_mean_tokens(model, edit, [pool.prompt(item["repo"], e) for e in key[1]], device)
            edit.task_mean = mean_cache[key]
        cur, frame, length = pool.current(item["repo"], item["episode"], item["progress"])
        prompts = {
            "self": cur["behavior_prompt"],
            "other": pool.prompt(item["repo"], item["other_episode"]),
            "wrong": pool.prompt(item["wrong_repo"], item["wrong_episode"]),
        }
        masked = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in prompts["other"].items()}
        masked["mask"] = torch.zeros_like(masked["mask"])
        prompts["mask"] = masked
        for cond, keys in MODALITY_SWAP.items():
            mixed = dict(prompts["other"])
            mixed.update({k: prompts["wrong"][k] for k in keys})
            prompts[cond] = mixed
        rec: dict[str, Any] = {**item, "frame": frame, "length": length, "loss": {}, "attn": {}}
        for cond in conditions:
            sample = dict(cur)
            sample["behavior_prompt"] = prompts["other" if cond in TEMPORAL_EDITS else ROUTE_PROMPT.get(cond, cond)]
            cap.block = ROUTE_BLOCK.get(cond)
            if edit is not None:
                edit.edit = cond if cond in TEMPORAL_EDITS else None
            batch = move_to_device(default_collate([sample]), device)
            state = F.pad(batch["observation.state"], (0, pcfg.max_state_dim - batch["observation.state"].shape[-1]))
            raw = batch["action"]
            actions = F.pad(raw, (0, pcfg.max_action_dim - raw.shape[-1]))
            valid = batch["action_is_valid"].bool()
            with torch.inference_mode():
                hidden, vmask = model.embed_prefix_with_bp(
                    pixel_values=batch["observation.pixel_values"], image_grid_thw=batch["observation.image_grid_thw"],
                    input_ids=batch["observation.input_ids"], attention_mask=batch["observation.attention_mask"],
                    behavior_prompt=batch["behavior_prompt"],
                )
                n_bp = hidden.shape[1] - batch["observation.input_ids"].shape[1]
                pos = model.build_position_ids(
                    batch["observation.input_ids"], batch["observation.mm_token_type_ids"],
                    batch["observation.image_grid_thw"], batch["observation.attention_mask"], 1 + actions.shape[1], n_bp,
                )
                bp_mask = vmask[:, -n_bp:]
                cap.spans = segment_spans(batch, bp_mask, actions.shape[1], num_chunks)
                cap.spans = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cap.spans.items()}
                gen = torch.Generator(device="cpu").manual_seed(int(item["noise_seed"]))
                noise = torch.randn(actions.shape, generator=gen).to(device=device, dtype=actions.dtype)
                losses, attn = [], []
                for t in times:
                    cap.records = []
                    cap.active = cond in ATTN_CONDITIONS and not ablation
                    out_flow = model.core.forward_embedded(
                        hidden, state, actions, noise=noise, time=torch.full((1,), t, device=device),
                        position_ids=pos, vlm_valid_mask=vmask,
                    )["flow"]
                    cap.active = False
                    flow = out_flow[0, :, :adim][valid[0]]
                    losses.append(float(flow.mean()))
                    if cap.records:
                        attn.append({k: torch.stack([r[k] for r in cap.records]) for k in cap.records[0]})
            rec["loss"][cond] = losses
            if attn:
                rec["attn"][cond] = {k: torch.stack([a[k] for a in attn]) for k in attn[0]}  # [T, L, ...]
            if cond == "other":
                rec["bp_valid_chunks"] = int(bp_mask[0].reshape(num_chunks, -1).any(-1).sum())
                rec["image_grid"] = cap.spans["image_grid"]
        results.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  [{ck['name']}] {n + 1}/{len(plan)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint": ck, "times": times, "segments": SEGMENTS, "results": results}, path)
    print(f"[done] {path}", flush=True)


def run_workers(cfg: dict[str, Any], names: list[str], gpus: list[str], out: Path, script: Path | None = None) -> None:
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue, running, free, failed = list(names), {}, list(gpus), []
    while queue or running:
        while queue and free:
            name, gpu = queue.pop(0), free.pop(0)
            log = (logs / f"{name}.log").open("w")
            cmd = [sys.executable, str(script or Path(__file__).resolve()), "--config", cfg["_config_path"], "--worker", name]
            proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu), stdout=log, stderr=subprocess.STDOUT)
            running[name] = (proc, gpu, log, time.time())
            print(f"[run] {name} -> GPU {gpu}", flush=True)
        time.sleep(3)
        for name, (proc, gpu, log, start) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[name]
            free.append(gpu)
            print(f"[run] {name} exit {proc.returncode} in {time.time() - start:.0f}s", flush=True)
            if proc.returncode != 0:
                failed.append(name)
    if failed:
        raise RuntimeError(f"failed: {failed}")


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
            run_workers(cfg, pending, gpus, out)
        else:
            for n in pending:
                run_checkpoint(cfg, ckpts[n], out)
    from analyze import analyze

    analyze(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
