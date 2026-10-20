#!/usr/bin/env python3
"""BPVAv3.1 vs taskcon offline probe.

Per sample (task x episode x progress) and BP condition it records:
  * flow loss at fixed noise / times (both model families);
  * action-expert attention over BP keyframes (self / other only);
  * v3.1 only: retrieval-head logits vs the next-keyframe target, and the next-keyframe state prediction.
Episodes are labelled left / right / both arm from raw parquet joint motion, so "other" BP can be split
into same-arm vs cross-arm demonstrations.

Usage:
  python probe_v31.py --config config.yaml --gpus 0,1,2,3
  python probe_v31.py --config config.yaml --analyze-only
"""
from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bpv3_attn_probe"))
import attn_probe as ap  # noqa: E402  (also puts bpv3_encoder_probe on sys.path)

# other: same-task demo, same arm when possible; other_x: same-task demo with the other arm, if any.
CONDITIONS = ("self", "other", "other_x", "reverse", "chunk_mean", "no_pos", "wrong", "mask")
EDITS = ("reverse", "chunk_mean")
ATTN = ("self", "other")
ON_OTHER = ("reverse", "chunk_mean", "no_pos", "mask")
STATE_DIM = 14
LEFT, RIGHT = slice(0, 7), slice(7, 14)
EVAL_TASKS = (
    "handover_mic", "place_cans_plasticbox", "place_mouse_pad", "place_shoe", "rotate_qrcode",
    "scan_object", "stack_blocks_two", "stack_bowls_two",
)


def load_cfg(path: Path) -> dict[str, Any]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path.resolve())
    return cfg


# ---------------------------------------------------------------- arm labels


def arm_labels(repo: str, ratio: float) -> dict[int, str]:
    import pandas as pd

    frames = [pd.read_parquet(p, columns=["episode_index", "observation.state"])
              for p in sorted(Path(repo, "data").rglob("*.parquet"))]
    df = pd.concat(frames)
    labels = {}
    for ep, g in df.groupby("episode_index"):
        s = np.stack(g["observation.state"].to_numpy())
        motion = np.abs(np.diff(s, axis=0)).sum(0)
        left, right = motion[LEFT].sum(), motion[RIGHT].sum()
        labels[int(ep)] = "left" if left > ratio * right else "right" if right > ratio * left else "both"
    return labels


def build_plan(cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[int, str]]]:
    rows = [json.loads(line) for line in Path(cfg["manifest"]).read_text().splitlines() if line.strip()]
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    tasks = sorted(by_task)[: int(cfg.get("max_tasks", 10**6))]
    arms = {by_task[t][0]["repo"]: arm_labels(by_task[t][0]["repo"], float(cfg.get("arm_ratio", 3.0))) for t in tasks}
    rng = random.Random(int(cfg.get("seed", 0)))
    plan = []
    for task in tasks:
        repo = by_task[task][0]["repo"]
        eps = [r["episode"] for r in by_task[task]]
        arm = arms[repo]
        for cur in eps[: int(cfg["episodes_per_task"])]:
            pool = [e for e in eps if e != cur]
            same = [e for e in pool if arm[e] == arm[cur]]
            cross = [e for e in pool if {arm[e], arm[cur]} == {"left", "right"}]
            other = same[0] if same else pool[0]
            wrong_task = rng.choice([t for t in tasks if t != task])
            for progress in cfg["progress"]:
                plan.append({
                    "task": task, "repo": repo, "episode": cur, "progress": float(progress),
                    "arm": arm[cur], "other_episode": other, "other_arm": arm[other],
                    "other_x_episode": cross[0] if cross else None,
                    "wrong_task": wrong_task, "wrong_repo": by_task[wrong_task][0]["repo"],
                    "wrong_episode": by_task[wrong_task][0]["episode"], "noise_seed": len(plan),
                })
    return plan, arms


# ---------------------------------------------------------------- data


def train_dataset_cfg(ckpt: str) -> dict[str, Any]:
    return json.loads((Path(ckpt) / "train_config.json").read_text())["dataset"]


class DataPool:
    def __init__(self, cfg: dict[str, Any], pcfg: Any, plan: list[dict[str, Any]], ckpt: str):
        from types import SimpleNamespace

        from probe import action_delta_timestamps, dataset_cfg

        from lerobot.datasets.factory import _load_external_stats_for_dataset
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        train_ds = train_dataset_cfg(ckpt)
        probe_cfg = dict(cfg, bp_action_use=train_ds["bp_action_use"], bp_state_use=train_ds["bp_state_use"])
        self.v31 = pcfg.type == "bpvav31"
        need: dict[str, set[int]] = {}
        for p in plan:
            need.setdefault(p["repo"], set()).update({p["episode"], p["other_episode"]})
            if p["other_x_episode"] is not None:
                need[p["repo"]].add(p["other_x_episode"])
            need.setdefault(p["wrong_repo"], set()).add(p["wrong_episode"])
        stats_cfg = SimpleNamespace(dataset=SimpleNamespace(
            use_external_stats=True, external_stats_path=None,
            external_stats_root=cfg["external_stats_root"], action_mode="delta",
        ))
        acs = int(getattr(pcfg, "bp_action_chunk_size", 50))
        self.ds, self.cache = {}, {}
        for repo, eps in need.items():
            kw = dict(episodes=sorted(eps), delta_timestamps=action_delta_timestamps(repo, acs),
                      video_backend=cfg.get("video_backend", "torchcodec"), skip_video_file_validation=True)
            current_ds, prompt_ds = LeRobotDataset(repo, **kw), LeRobotDataset(repo, **kw)
            _load_external_stats_for_dataset(stats_cfg, current_ds)
            prompt_ds.meta.stats.update(current_ds.meta.stats)
            v3_cfg = dataset_cfg(probe_cfg, repo, pcfg)
            if self.v31:
                from lerobot.policies.BPVAv31.configuration_bpvav31 import BPVAv31DatasetConfig
                from lerobot.policies.BPVAv31.dataset import make_bpvav31_behavior_dataset

                fields = {k: getattr(v3_cfg, k) for k in (
                    "repo_id", "action_mode", "height", "width", "max_state_dim", "max_action_dim",
                    "current_image_index", "qwen3_5_processor_path", "qwen3_5_processor_revision", "use_language",
                    "bp_num_chunks", "bp_fixed_num_chunks", "bp_seed", "bp_sampling_mode", "align_bp_encoder",
                    "bp_action_use", "bp_state_use", "bp_camera_keys")}
                dcfg = BPVAv31DatasetConfig(**fields, bp_attach_retrieval=True, bp_attach_task_id=False)
                self.ds[repo] = make_bpvav31_behavior_dataset(current_ds, prompt_ds, dcfg, bp_action_chunk_size=acs)
            else:
                from lerobot.policies.BPVAv3.dataset import make_bpvav3_behavior_dataset

                self.ds[repo] = make_bpvav3_behavior_dataset(current_ds, prompt_ds, v3_cfg, bp_action_chunk_size=acs)

    def span(self, repo: str, episode: int) -> tuple[int, int]:
        start, end, _ = self.ds[repo]._episode_ranges[int(episode)]
        return int(start), int(end)

    def current(self, repo: str, episode: int, progress: float) -> tuple[dict, int, int]:
        start, end = self.span(repo, episode)
        frame = int(round(progress * (end - start - 1)))
        return self.ds[repo][start + frame], frame, end - start

    def prompt(self, repo: str, episode: int) -> dict:
        key = (repo, int(episode))
        if key not in self.cache:
            self.cache[key] = self.ds[repo][self.span(repo, episode)[0]]["behavior_prompt"]
        return self.cache[key]


@contextlib.contextmanager
def no_position(encoder, active: bool):
    if not active:
        yield
        return
    saved = (getattr(encoder, "use_progress_encoding", None), encoder.chunk_position_embedding)
    if saved[0] is not None:
        encoder.use_progress_encoding = False
    encoder.chunk_position_embedding = None
    try:
        yield
    finally:
        if saved[0] is not None:
            encoder.use_progress_encoding = saved[0]
        encoder.chunk_position_embedding = saved[1]


def retrieval(model, hidden, vmask, input_ids, state, prompt, progress, query_progress=None):
    from lerobot.policies.BPVAv31.bp_encoder import sinusoidal_progress_encoding
    from lerobot.policies.BPVAv31.modeling_bpvav31 import next_keyframe_index

    n_bp = hidden.shape[1] - input_ids.shape[1]
    chunk, cmask = model._chunk_pool(hidden[:, -n_bp:].float(), vmask[:, -n_bp:])
    ratios = torch.as_tensor(prompt["source_time_ratio"], device=chunk.device, dtype=torch.float32).reshape(1, -1)
    target = next_keyframe_index(progress, ratios, cmask)
    prefix = hidden[:, :-n_bp].float()
    pv = vmask[:, :-n_bp].to(prefix.dtype).unsqueeze(-1)
    pooled = (prefix * pv).sum(1) / pv.sum(1).clamp_min(1.0)
    qp = progress if query_progress is None else query_progress
    enc = sinusoidal_progress_encoding(qp, prefix.shape[-1]).to(pooled.dtype)
    dtype = next(model.bp_retrieve_query.parameters()).dtype
    query = model.bp_retrieve_query(torch.cat([pooled, enc, state.float()], -1).to(dtype)).float()
    key = model.bp_retrieve_key(chunk.to(dtype)).float()
    logits = torch.einsum("bd,bcd->bc", query, key) / float(model.config.bp_retrieve_tau)
    logits = logits.masked_fill(~cmask, float("-inf"))
    attn = torch.nan_to_num(torch.softmax(logits, -1), nan=0.0)
    pred = model.bp_next_state_head(torch.einsum("bc,bcd->bd", attn, chunk).to(dtype)).float()
    return {"logits": logits[0].cpu(), "target": int(target[0]), "pred": pred[0, :STATE_DIM].cpu()}


# ---------------------------------------------------------------- worker


def run_checkpoint(cfg: dict[str, Any], ck: dict[str, str], out: Path) -> None:
    from probe import load_policy, move_to_device
    from torch.utils.data import default_collate

    import lerobot.policies.BPVAv3.configuration_bpvav3  # noqa: F401
    import lerobot.policies.BPVAv31.configuration_bpvav31  # noqa: F401
    from lerobot.policies.BPVAv31.modeling_bpvav31 import next_keyframe_index

    path = out / "shards" / f"{ck['name']}.pt"
    if path.is_file():
        print(f"[skip] {path}", flush=True)
        return
    policy, pcfg, device = load_policy(ck["path"])
    model = policy.model
    plan, _ = build_plan(cfg)
    pool = DataPool(cfg, pcfg, plan, ck["path"])
    has_head = hasattr(model, "bp_retrieve_query")
    has_content_head = hasattr(model, "bp_content_query")
    cap = ap.Capture(model.core.backbone)
    edit = ap.ChunkEdit(model.bp_obs_encoder.chunk_encoder)
    num_chunks = int(pcfg.bp_num_chunks)
    adim = int(pcfg.output_features["action"].shape[0])
    times = [float(t) for t in cfg["times"]]
    results = []
    for n, item in enumerate(plan):
        cur, frame, length = pool.current(item["repo"], item["episode"], item["progress"])
        progress = frame / max(1, length - 1)
        other = pool.prompt(item["repo"], item["other_episode"])
        masked = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in other.items()}
        masked["mask"] = torch.zeros_like(masked["mask"])
        prompts = {"self": cur["behavior_prompt"], "other": other, "mask": masked,
                   "wrong": pool.prompt(item["wrong_repo"], item["wrong_episode"])}
        if item["other_x_episode"] is not None:
            prompts["other_x"] = pool.prompt(item["repo"], item["other_x_episode"])
        rec: dict[str, Any] = {**item, "frame": frame, "length": length, "progress_exact": progress,
                               "loss": {}, "attn": {}, "ret": {}, "cret": {}, "content": {}}
        prog_t = torch.tensor([progress], device=device)
        for cond in CONDITIONS:
            source = "other" if cond in ON_OTHER and cond != "mask" else cond
            if source not in prompts:
                continue
            sample = dict(cur)
            sample["behavior_prompt"] = prompts[source]
            edit.edit = cond if cond in EDITS else None
            cap.block = None
            batch = move_to_device(default_collate([sample]), device)
            state = F.pad(batch["observation.state"], (0, pcfg.max_state_dim - batch["observation.state"].shape[-1]))
            raw = batch["action"]
            actions = F.pad(raw, (0, pcfg.max_action_dim - raw.shape[-1]))
            valid = batch["action_is_valid"].bool()
            with torch.inference_mode(), no_position(model.bp_obs_encoder, cond == "no_pos"):
                hidden, vmask = model.embed_prefix_with_bp(
                    pixel_values=batch["observation.pixel_values"], image_grid_thw=batch["observation.image_grid_thw"],
                    input_ids=batch["observation.input_ids"], attention_mask=batch["observation.attention_mask"],
                    behavior_prompt=batch["behavior_prompt"],
                )
            edit.edit = None
            if cond in ("self", "other", "other_x"):
                # 加时间编码前的 BP 内容（hook 在改写前记录），每个 chunk 对 query 求均值。
                rec["content"][cond] = edit.last[0][0].float().mean(1).half().cpu()
            if has_content_head and cond != "mask":
                with torch.inference_mode():
                    clogits, _, cmask = model.content_retrieval(batch_size=1, state=state)
                ratios = torch.as_tensor(batch["behavior_prompt"]["source_time_ratio"], device=device).float().reshape(-1)
                target = int((ratios - progress).abs().masked_fill(~cmask[0], float("inf")).argmin())
                rec["cret"][cond] = {"logits": clogits[0].float().cpu(), "target": target}
            with torch.inference_mode():
                n_bp = hidden.shape[1] - batch["observation.input_ids"].shape[1]
                pos = model.build_position_ids(
                    batch["observation.input_ids"], batch["observation.mm_token_type_ids"],
                    batch["observation.image_grid_thw"], batch["observation.attention_mask"], 1 + actions.shape[1], n_bp,
                )
                bp_mask = vmask[:, -n_bp:]
                cap.spans = ap.segment_spans(batch, bp_mask, actions.shape[1], num_chunks)
                cap.spans = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cap.spans.items()}
                gen = torch.Generator(device="cpu").manual_seed(int(item["noise_seed"]))
                noise = torch.randn(actions.shape, generator=gen).to(device=device, dtype=actions.dtype)
                losses, attn = [], []
                for t in times:
                    cap.records = []
                    cap.active = cond in ATTN
                    flow = model.core.forward_embedded(
                        hidden, state, actions, noise=noise, time=torch.full((1,), t, device=device),
                        position_ids=pos, vlm_valid_mask=vmask,
                    )["flow"]
                    cap.active = False
                    losses.append(float(flow[0, :, :adim][valid[0]].mean()))
                    if cap.records:
                        attn.append({k: torch.stack([r[k] for r in cap.records]) for k in ("mass", "chunk")})
                rec["loss"][cond] = losses
                if attn:
                    # [T, L, H, ...] -> head mean [T, L, ...]
                    rec["attn"][cond] = {k: torch.stack([a[k] for a in attn]).mean(2) for k in ("mass", "chunk")}
                if has_head and cond != "mask":
                    r = retrieval(model, hidden, vmask, batch["observation.input_ids"], state,
                                  batch["behavior_prompt"], prog_t)
                    if cond == "other":
                        fixed = retrieval(model, hidden, vmask, batch["observation.input_ids"], state,
                                          batch["behavior_prompt"], prog_t, query_progress=torch.full_like(prog_t, 0.5))
                        rec["ret"]["other_qfixed"] = fixed
                    rec["ret"][cond] = r
                if has_head and cond in ("self", "other", "other_x"):
                    demo = batch["behavior_prompt"]["state"][0].float()
                    ratios = torch.as_tensor(batch["behavior_prompt"]["source_time_ratio"], device=device).reshape(1, -1)
                    idx = int(next_keyframe_index(prog_t, ratios.float(), bp_mask.reshape(1, num_chunks, -1).any(-1))[0])
                    rec.setdefault("demo_next_state", {})[cond] = demo[idx, :STATE_DIM].cpu()
        if "bp_origin_state" in cur:
            origin = cur["bp_origin_state"].float()
            oratio = torch.as_tensor(cur["bp_origin_time_ratio"]).float().reshape(1, -1)
            oidx = int(next_keyframe_index(torch.tensor([progress]), oratio, None)[0])
            rec["target_state"] = origin[oidx, :STATE_DIM]
        rec["current_state"] = cur["observation.state"].float()[:STATE_DIM]
        results.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  [{ck['name']}] {n + 1}/{len(plan)}", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"checkpoint": ck, "times": times, "results": results}, path)
    print(f"[done] {path}", flush=True)


# ---------------------------------------------------------------- analysis


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation) if len(x) > 2 else float("nan")


def summarize(run: dict) -> dict[str, Any]:
    res = run["results"]
    s: dict[str, Any] = {"n": len(res)}
    loss = {}
    for c in CONDITIONS:
        rows = [r for r in res if c in r["loss"]]
        loss[c] = {"mean": float(np.mean([np.mean(r["loss"][c]) for r in rows])) if rows else None, "n": len(rows)}
    s["loss"] = loss
    # paired ratio vs other on the same items (median over tasks of per-task mean ratio)
    s["ratio_vs_other"] = {}
    tasks = sorted({r["task"] for r in res})
    for c in CONDITIONS:
        per_task = []
        for t in tasks:
            rows = [r for r in res if r["task"] == t and c in r["loss"]]
            if rows:
                per_task.append(np.mean([np.mean(r["loss"][c]) for r in rows]) /
                                np.mean([np.mean(r["loss"]["other"]) for r in rows]))
        s["ratio_vs_other"][c] = float(np.median(per_task)) if per_task else None
    eval_rows = [r for r in res if r["task"] in EVAL_TASKS]
    s["loss_eval"] = {c: float(np.mean([np.mean(r["loss"][c]) for r in eval_rows if c in r["loss"]]))
                      for c in CONDITIONS if any(c in r["loss"] for r in eval_rows)}

    # action attention: expected BP keyframe position vs current progress
    prog = np.array([r["progress_exact"] for r in res])
    s["attn_rho"] = {}
    for c in ATTN:
        chunk = torch.stack([r["attn"][c]["chunk"] for r in res]).mean(1)  # [N, L, C]
        c_n = chunk.shape[-1]
        rho = []
        for layer in range(chunk.shape[1]):
            m = chunk[:, layer].numpy()
            m = m / np.clip(m.sum(-1, keepdims=True), 1e-12, None)
            rho.append(_spearman(m @ np.linspace(0, 1, c_n), prog))
        s["attn_rho"][c] = {"by_layer": rho, "max_abs": float(np.nanmax(np.abs(rho))), "mean": float(np.nanmean(rho))}
        s.setdefault("bp_mass", {})[c] = float(torch.stack([r["attn"][c]["mass"][..., 4] for r in res]).mean())

    # retrieval head (v3.1)
    if res[0]["ret"]:
        s["retrieval"] = {}
        for c in [*CONDITIONS, "other_qfixed"]:
            rows = [r["ret"][c] for r in res if c in r["ret"]]
            if not rows:
                continue
            hit = [int(x["logits"].argmax()) == x["target"] for x in rows]
            near = [abs(int(x["logits"].argmax()) - x["target"]) <= 1 for x in rows]
            exp = [float((torch.softmax(x["logits"], 0) * torch.arange(len(x["logits"]))).sum()) for x in rows]
            s["retrieval"][c] = {"top1": float(np.mean(hit)), "within1": float(np.mean(near)),
                                 "rho_expected_vs_target": _spearman(exp, [x["target"] for x in rows]), "n": len(rows)}
        targets = [r["ret"]["other"]["target"] for r in res if "other" in r["ret"]]
        s["retrieval_majority_baseline"] = float(np.max(np.bincount(targets)) / len(targets))
        # next-keyframe state: 14 real dims
        tgt = {i: r["target_state"] for i, r in enumerate(res) if "target_state" in r}
        def mse(a, b):
            return float(((a - b) ** 2).mean())
        s["next_state_mse"] = {}
        for c in ("self", "other", "other_x", "reverse", "chunk_mean", "no_pos", "wrong"):
            vals = [mse(res[i]["ret"][c]["pred"], t) for i, t in tgt.items() if c in res[i]["ret"]]
            if vals:
                s["next_state_mse"][c] = float(np.mean(vals))
        s["next_state_mse"]["baseline_copy_current"] = float(np.mean([mse(res[i]["current_state"], t) for i, t in tgt.items()]))
        for c in ("self", "other", "other_x"):
            vals = [mse(res[i]["demo_next_state"][c], t) for i, t in tgt.items() if c in res[i].get("demo_next_state", {})]
            if vals:
                s["next_state_mse"][f"baseline_copy_demo_{c}"] = float(np.mean(vals))
    return s


def keyframe_align(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """示教 a 的每个关键帧在示教 b 里按余弦找最近邻，返回 (top1, within1) 命中率。"""
    sim = F.normalize(a.float(), dim=-1) @ F.normalize(b.float(), dim=-1).T
    j = sim.argmax(-1)
    i = torch.arange(len(j))
    return float((j == i).float().mean()), float(((j - i).abs() <= 1).float().mean())


def content_metrics(run: dict) -> dict[str, Any]:
    """BP encoder 内容本身（加时间编码前）：跨示教关键帧对齐 + 任务 kNN。所有 ckpt 可比。"""
    res = [r for r in run["results"] if r.get("content")]
    if not res:
        return {}
    out: dict[str, Any] = {}
    seen, rows = set(), []
    for r in res:
        key = (r["task"], r["episode"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    single = [r for r in rows if r["arm"] in ("left", "right")]
    for name, pick, subset in (
        ("same_arm_all", "other", [r for r in rows if r["other_arm"] == r["arm"]]),
        ("same_arm_single", "other", [r for r in single if r["other_arm"] == r["arm"]]),
        ("cross_arm_single", "other_x", [r for r in single if "other_x" in r["content"]]),
    ):
        vals = [keyframe_align(r["content"]["self"], r["content"][pick]) for r in subset if pick in r["content"]]
        if vals:
            out[f"kf_align_{name}"] = {"top1": float(np.mean([v[0] for v in vals])),
                                       "within1": float(np.mean([v[1] for v in vals])), "n": len(vals)}
    feats = torch.stack([F.normalize(r["content"]["other"].float().mean(0), dim=-1) for r in rows])
    labels = [r["task"] for r in rows]
    sim = feats @ feats.T
    sim.fill_diagonal_(-2)
    nn_idx = sim.argmax(-1).tolist()
    out["task_knn_acc"] = float(np.mean([labels[i] == labels[j] for i, j in enumerate(nn_idx)]))
    return out


def content_retrieval_metrics(run: dict) -> dict[str, Any]:
    res = [r for r in run["results"] if r.get("cret")]
    if not res:
        return {}
    out: dict[str, Any] = {}

    def acc(rows, cond):
        rows = [r["cret"][cond] for r in rows if cond in r["cret"]]
        if not rows:
            return None
        d = [abs(int(x["logits"].argmax()) - x["target"]) for x in rows]
        return {"top1": float(np.mean([v == 0 for v in d])), "within1": float(np.mean([v <= 1 for v in d])), "n": len(rows)}

    same = [r for r in res if r["other_arm"] == r["arm"]]
    for cond in CONDITIONS:
        v = acc(same, cond)
        if v:
            out[cond] = v
    single = [r for r in res if r["arm"] in ("left", "right") and "other_x" in r["cret"]]
    out["single_arm_other"] = acc(single, "other")
    out["single_arm_other_x"] = acc(single, "other_x")
    return out


def arm_split(run: dict) -> dict[str, Any]:
    """Single-arm current episodes only: same-arm demo (other) vs cross-arm demo (other_x) on identical items."""
    res = [r for r in run["results"] if r["arm"] in ("left", "right") and "other_x" in r["loss"]
           and r["other_arm"] == r["arm"]]
    if not res:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(res), "tasks": sorted({r["task"] for r in res})}
    out["loss"] = {c: float(np.mean([np.mean(r["loss"][c]) for r in res])) for c in ("self", "other", "other_x", "mask")}
    out["cross_worse_frac"] = float(np.mean([np.mean(r["loss"]["other_x"]) > np.mean(r["loss"]["other"]) for r in res]))
    if res[0]["ret"]:
        out["retrieval_top1"] = {c: float(np.mean([int(r["ret"][c]["logits"].argmax()) == r["ret"][c]["target"] for r in res]))
                                 for c in ("self", "other", "other_x")}
        def mse(a, b):
            return float(((a - b) ** 2).mean())
        out["next_state_mse"] = {c: float(np.mean([mse(r["ret"][c]["pred"], r["target_state"]) for r in res]))
                                 for c in ("self", "other", "other_x")}
        out["copy_demo_mse"] = {c: float(np.mean([mse(r["demo_next_state"][c], r["target_state"]) for r in res]))
                                for c in ("self", "other", "other_x")}
    return out


def analyze(cfg: dict[str, Any], out: Path) -> None:
    names = [ck["name"] for ck in cfg["checkpoints"] if (out / "shards" / f"{ck['name']}.pt").is_file()]
    summary = {}
    for n in names:
        run = torch.load(out / "shards" / f"{n}.pt", map_location="cpu", weights_only=False)
        summary[n] = {"all": summarize(run), "arm_split": arm_split(run),
                      "content": content_metrics(run), "content_retrieval": content_retrieval_metrics(run)}
        arms = [r["arm"] for r in run["results"][:: len(cfg["progress"])]]
        summary[n]["arm_counts"] = {a: arms.count(a) for a in ("left", "right", "both")}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    for n, s in summary.items():
        a = s["all"]
        print(f"== {n}  (n={a['n']})")
        print("  flow ratio vs other: " + " ".join(f"{c}={v:.3f}" for c, v in a["ratio_vs_other"].items() if v))
        print(f"  attn rho max|.| self={a['attn_rho']['self']['max_abs']:.2f} other={a['attn_rho']['other']['max_abs']:.2f}"
              f"  bp_mass other={a['bp_mass']['other']:.3f}")
        if "retrieval" in a:
            print("  retrieval top1: " + " ".join(f"{c}={v['top1']:.2f}" for c, v in a["retrieval"].items())
                  + f"  majority={a['retrieval_majority_baseline']:.2f}")
            print("  next-state mse: " + " ".join(f"{c}={v:.3f}" for c, v in a["next_state_mse"].items()))
        print(f"  arm split: {json.dumps(s['arm_split'], ensure_ascii=False)[:600]}")
        if s["content"]:
            print("  content: " + json.dumps(s["content"], ensure_ascii=False))
        if s["content_retrieval"]:
            print("  content retrieval: " + " ".join(
                f"{c}={v['top1']:.2f}/{v['within1']:.2f}" for c, v in s["content_retrieval"].items() if v))


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
            ap.run_workers(cfg, pending, gpus, out, script=Path(__file__).resolve())
        else:
            for n in pending:
                run_checkpoint(cfg, ckpts[n], out)
    analyze(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
