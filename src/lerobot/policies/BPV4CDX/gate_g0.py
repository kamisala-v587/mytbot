"""D0 / G0: canonical texts, within-task behaviour differences (for G5), sampler coverage and slot checks.

Outputs (under --out):
  canonical_text.json         task -> canonical text
  g5_differences.csv          per task: episodes, grasp-arm counts, joint-space spread at the first gripper close,
                              duration spread (state only, no FK, so it also runs on the training machine)
  sampler_coverage.csv        per episode and sampler: interpolation RMS error, gripper-flip hit rate, max slot gap,
                              repeats, slot checks
  sampler_summary.json        per sampler: means of the above; G0 pass / fail
  contact_sheets/*.png        head-camera 32-slot contact sheets for a few episodes per sampler (with --sheets)

S-V needs per-frame ViT features of every frame, so it runs on --sv_episodes episodes only (GPU).

Usage: python -m lerobot.policies.BPV4CDX.gate_g0 --repo_file <txt> [--data_root <dir>] --out <dir> \
         [--episodes 100] [--sv_episodes 20] [--sheets 3] [--qwen_path <base_800m>]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from lerobot.policies.BPV4BPEn.build_feature_cache import load_states
from lerobot.policies.BPV4BPEn.keyframes import gripper_flips

from .paths import canonical_text, read_repo_list, task_name
from .sampling import SAMPLERS, interpolation_error, sample

ARM_JOINTS = {"L": slice(0, 6), "R": slice(7, 13)}
GRIPPER = {"L": 6, "R": 13}


def first_close(state: np.ndarray) -> tuple[str, int] | None:
    best = None
    for arm, g in GRIPPER.items():
        closed = state[:, g] < 0.5
        idx = np.nonzero(closed[1:] & ~closed[:-1])[0]
        if len(idx) and (best is None or idx[0] + 1 < best[1]):
            best = (arm, int(idx[0] + 1))
    return best


def slot_checks(plan, segment_len: int) -> dict:
    f = plan.frames
    lo, hi = plan.active
    starts = np.arange(0, plan.length, segment_len)
    return {
        "n_slots": int(len(f)),
        "monotonic": bool(np.all(np.diff(f) >= 0)),
        "in_range": bool(f.min() >= 0 and f.max() < plan.length),
        "covers_start": bool(f[0] == lo),
        "covers_end": bool(f[-1] == hi),
        "flips_included": bool(all(x in set(f.tolist()) for x in plan.flips)) if plan.sampler == "sa" else None,
        "segments": int(len(starts)),
        "segments_in_range": bool(np.all(starts < plan.length)),
    }


def per_frame_features(repo: str, ep: int, qwen_path: str, vlm, processor, device) -> np.ndarray:
    """[T, 3*1024]: mean ViT token of each camera, for every frame of the episode (S-V proxy input)."""
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.BPV4BPEn.build_feature_cache import CAMERAS, visual_tokens
    from lerobot.transforms.utils import resize_with_pad

    ds = LeRobotDataset(repo, video_backend="torchcodec")
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    end = int(ds.meta.episodes[ep]["dataset_to_index"])
    out = []
    for s in range(start, end, 16):
        images = []
        for i in range(s, min(end, s + 16)):
            item = ds[i]
            images.extend(resize_with_pad(item[c][None], 224, 224)[0] for c in CAMERAS)
        proc = processor(images, do_rescale=False)
        with torch.no_grad():
            feats = visual_tokens(vlm, torch.as_tensor(proc.pixel_values).to(device, torch.bfloat16),
                                  torch.as_tensor(proc.image_grid_thw, dtype=torch.long).to(device))
        out.append(feats.float().mean(1).reshape(-1, 3 * feats.shape[-1]).cpu().numpy())
    return np.concatenate(out)


def contact_sheet(repo: str, ep: int, plans: dict, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.transforms.utils import resize_with_pad

    ds = LeRobotDataset(repo, video_backend="torchcodec")
    start = int(ds.meta.episodes[ep]["dataset_from_index"])
    names = list(plans)
    k = len(next(iter(plans.values())).frames)
    fig, axes = plt.subplots(len(names), k, figsize=(k * 0.9, len(names) * 1.0 + 0.6))
    axes = np.atleast_2d(axes)
    for r, name in enumerate(names):
        plan = plans[name]
        flips = set(plan.flips.tolist())
        for j, f in enumerate(plan.frames):
            img = resize_with_pad(ds[start + int(f)]["observation.images.cam_high"][None], 224, 224)[0]
            ax = axes[r, j]
            ax.imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if int(f) in flips:
                for sp in ax.spines.values():
                    sp.set_edgecolor("red"), sp.set_linewidth(2)
            ax.set_title(str(int(f)), fontsize=6)
        axes[r, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    fig.suptitle(f"{task_name(repo)} {Path(repo).name} ep{ep}: head camera, 32 slots per sampler (red = gripper flip)",
                 fontsize=9)
    fig.tight_layout(rect=(0.03, 0, 1, 0.93))
    fig.savefig(out, dpi=90)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo_file", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num_slots", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--segment_len", type=int, default=50)
    ap.add_argument("--episodes", type=int, default=100, help="episodes for the S-A / S-U coverage comparison")
    ap.add_argument("--sv_episodes", type=int, default=20, help="episodes for S-V (needs ViT on every frame)")
    ap.add_argument("--sheets", type=int, default=3)
    ap.add_argument("--qwen_path", default=None, help="Qwen3.5-0.8B base (needed for S-V and nothing else)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repos = read_repo_list(args.repo_file, args.data_root)
    rng = np.random.default_rng(args.seed)

    # canonical texts
    texts = {task_name(r): canonical_text(r) for r in repos}
    (out / "canonical_text.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))

    # within-task behaviour differences (state only)
    states = {r: load_states(r) for r in repos}
    by_task: dict[str, list[np.ndarray]] = {}
    for r in repos:
        by_task.setdefault(task_name(r), []).extend(states[r].values())
    rows = []
    for t, eps in sorted(by_task.items()):
        arms, configs, lengths = {"L": 0, "R": 0, "none": 0}, {"L": [], "R": []}, []
        for s in eps:
            lengths.append(len(s))
            fc = first_close(s)
            if fc is None:
                arms["none"] += 1
                continue
            arm, f = fc
            arms[arm] += 1
            configs[arm].append(s[f, ARM_JOINTS[arm]])
        spread = {a: float(np.mean(np.std(np.stack(v), axis=0))) if len(v) >= 3 else float("nan") for a, v in configs.items()}
        rows.append({"task": t, "episodes": len(eps), "arm_L": arms["L"], "arm_R": arms["R"], "no_close": arms["none"],
                     "joint_spread_L_rad": round(spread["L"], 4), "joint_spread_R_rad": round(spread["R"], 4),
                     "length_mean": round(float(np.mean(lengths)), 1),
                     "length_cv": round(float(np.std(lengths) / max(1e-9, np.mean(lengths))), 3),
                     # position class usable: >= 3 episodes with the same grasp arm and a visible spread
                     "position_class": bool(any(len(v) >= 3 and spread[a] > 0.02 for a, v in configs.items())),
                     # arm class usable: >= 3 episodes with each arm
                     "arm_class": bool(min(arms["L"], arms["R"]) >= 3)})
    with (out / "g5_differences.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(), w.writerows(rows)

    # sampler coverage
    pool = [(r, ep) for r in repos for ep in sorted(states[r])]
    pick = [pool[i] for i in rng.choice(len(pool), size=min(args.episodes, len(pool)), replace=False)]
    sv_pick = set(pick[: args.sv_episodes]) if args.qwen_path else set()
    vlm = processor = device = None
    if sv_pick:
        import torch

        from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit
        from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

        device = torch.device("cuda")
        vlm = load_vit(None, args.qwen_path, device)
        processor, _ = load_qwen35_vision(args.qwen_path)
    cov_rows, sheets_done = [], 0
    for r, ep in pick:
        s = states[r][ep]
        plans = {name: sample(name, s, args.num_slots, args.alpha) for name in ("sa", "su")}
        if (r, ep) in sv_pick:
            plans["sv"] = sample("sv", s, args.num_slots, args.alpha,
                                 frame_features=per_frame_features(r, ep, args.qwen_path, vlm, processor, device))
        flips = gripper_flips(s)
        for name, plan in plans.items():
            fr = set(plan.frames.tolist())
            hit = np.mean([any(abs(x - y) <= 1 for y in fr) for x in flips]) if flips else float("nan")
            cov_rows.append({"repo": r, "episode": ep, "sampler": name, "length": plan.length,
                             "interp_rms": interpolation_error(s, plan.frames), "flip_hit_rate": hit,
                             "max_gap_frac": float(np.max(np.diff(np.r_[plan.active[0], plan.frames, plan.active[1]]))
                                                   / max(1, plan.active[1] - plan.active[0] + 1)),
                             "repeats": plan.repeats, **slot_checks(plan, args.segment_len)})
        if sheets_done < args.sheets and len(plans) == 3:
            (out / "contact_sheets").mkdir(exist_ok=True)
            contact_sheet(r, ep, plans, out / "contact_sheets" / f"{task_name(r)}_{Path(r).name}_ep{ep}.png")
            sheets_done += 1
        print(r, ep, flush=True)
    with (out / "sampler_coverage.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cov_rows[0]))
        w.writeheader(), w.writerows(cov_rows)

    summary = {}
    for name in SAMPLERS:
        rs = [c for c in cov_rows if c["sampler"] == name]
        if not rs:
            continue
        mean = lambda k: float(np.nanmean([c[k] for c in rs]))  # noqa: E731
        checks = ["monotonic", "in_range", "covers_start", "covers_end", "segments_in_range"]
        summary[name] = {"episodes": len(rs), "interp_rms": mean("interp_rms"), "flip_hit_rate": mean("flip_hit_rate"),
                         "max_gap_frac": mean("max_gap_frac"), "repeats_total": int(sum(c["repeats"] for c in rs)),
                         "all_32_slots": all(c["n_slots"] == args.num_slots for c in rs),
                         **{k: all(c[k] for c in rs) for k in checks}}
        if name == "sa":
            summary[name]["flips_included"] = all(c["flips_included"] for c in rs)
    structural = all(v["all_32_slots"] and v["monotonic"] and v["in_range"] and v["covers_start"] and v["covers_end"]
                     for v in summary.values())
    summary["G0_structural_pass"] = bool(structural and summary.get("sa", {}).get("flips_included", True))
    summary["note"] = "G0 also needs a manual spot check of contact_sheets/ (timestamps, cameras, coverage)."
    (out / "sampler_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
