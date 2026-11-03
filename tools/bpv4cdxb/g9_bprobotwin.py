"""Compact G9 on BP RoboTwin candidate sets (gate_g9_main with a G9 index), seen vs unseen.

  prepare   choose the candidate sets and write the G9 index (all their episodes are queries / demos, one bucket per
            candidate set) and the repo -> group map:
              heldout_sets    the split's held-out candidate sets (never trained)
              heldout_tasks   every candidate set of the split's held-out tasks (task never trained)
              train_sibling   per held-out set, a trained set of the same anchor with another goal (same scenes, seen)
  summary   one Markdown table of the per-checkpoint results (matched action MSE, blank / alt / wrong ratios)

usage (my_tbot/, env.local.sh):
  python tools/bpv4cdxb/g9_bprobotwin.py prepare --out outputs/BPv4CDX/stage_b/g9_bp
  bash tools/bpv4cdxb/g9_bprobotwin.sh <run>/checkpoints        # runs gate_g9_main per checkpoint, then summary
  python tools/bpv4cdxb/g9_bprobotwin.py summary --out outputs/BPv4CDX/stage_b/g9_bp
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPLIT = Path("/vla/workspace/my_robotwin/bp_tasks/splits/v1.yaml")
DATA = Path("/vla/workspace/data")
BP_ROOT = "robotwin2.5_bp"
MANIFEST = ROOT / "configs/ds_ids/local/robotwin2.5_bp_split.json"
TRAIN_LIST = ROOT / "configs/ds_ids/local/robotwin2.5_bp_train_v1.txt"
CACHE = DATA / "robotwin2.5_vit_cache"
ZDEMO = "/vla/workspace/cache/bpv4cdxb_zdemo/89ef6e438045"


def rel_of(cs: str) -> str:
    """'place_shoe_bp/I1/L1/toe_left' -> 'robotwin2.5_bp/place_shoe_bp/I1_L1_toe_left'."""
    task, inst, anchor, goal = cs.split("/", 3)
    return f"{BP_ROOT}/{task}/{inst}_{anchor}_{goal}"


def prepare(out: Path) -> None:
    split = yaml.safe_load(SPLIT.read_text())
    train = {l.strip().split(f"/{BP_ROOT}/", 1)[1] for l in TRAIN_LIST.read_text().splitlines() if l.strip()}
    groups: dict[str, str] = {}
    for sets in split["heldout_sets"].values():
        for cs in sets:
            groups[rel_of(cs)] = "heldout_sets"
            task, inst, anchor, _ = cs.split("/", 3)
            prefix = f"{task}/{inst}_{anchor}_"
            siblings = sorted(t for t in train if t.startswith(prefix))
            if siblings:
                groups[f"{BP_ROOT}/{siblings[0]}"] = "train_sibling"
    for task in split["heldout_tasks"]:
        for d in sorted((DATA / BP_ROOT / task).iterdir()):
            if (d / "meta/info.json").is_file():
                groups[f"{BP_ROOT}/{task}/{d.name}"] = "heldout_tasks"
    manifest = json.loads(MANIFEST.read_text())
    entries = {e["rel"]: e for e in manifest["repos"]}
    missing = sorted(r for r in groups if r not in entries)
    if missing:
        raise SystemExit(f"not in {MANIFEST}: {missing[:3]}")
    # every episode of a chosen repo is held out -> it lands in the val pool (queries and demos)
    g9_manifest = {**manifest, "rule": "G9: all episodes of the chosen candidate sets held out",
                   "repos": [{**entries[r], "heldout": list(range(int(entries[r]["total_episodes"])))} for r in sorted(groups)]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "g9_manifest.json").write_text(json.dumps(g9_manifest, indent=1))
    (out / "g9_repos.txt").write_text("".join(f"{DATA / r}\n" for r in sorted(groups)))
    (out / "g9_groups.json").write_text(json.dumps(groups, indent=1, sort_keys=True))
    subprocess.run([sys.executable, "-m", "lerobot.policies.BPV4CDXB.pairs_main", "--repo_file", str(out / "g9_repos.txt"),
                    "--manifest", str(out / "g9_manifest.json"), "--zdemo_dir", ZDEMO, "--demo_cache_dir", str(CACHE),
                    "--bucket_by", "repo", "--out", str(out / "g9_index.json"), "--workers", "16"], check=True)
    counts = {g: sum(v == g for v in groups.values()) for g in sorted(set(groups.values()))}
    print("candidate sets per group:", counts, "->", out / "g9_index.json")


def fmt(r: dict | None) -> str:
    return "—" if not r else f"{r['ratio']:.2f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def summary(out: Path) -> None:
    rows = []
    for p in sorted((out / "results").glob("*.json")):
        res = json.loads(p.read_text())
        for grp, e in sorted(res.get("by_group", {}).items()):
            rows.append(f"| {p.stem} | {grp} | {e['queries']} | {e['matched_mse']:.4f} | {fmt(e.get('blank'))} | "
                        f"{fmt(e.get('alt'))} | {fmt(e.get('wrong'))} |")
        cx = res.get("cross_arm")
        rows.append(f"| {p.stem} | **all** | {res['queries']} | {res['mean_error']['matched']:.4f} | "
                    f"{fmt(res['ratios'].get('blank'))} | {fmt(res['ratios'].get('alt'))} | {fmt(res['ratios'].get('wrong'))} |"
                    + (f" cross-arm follow {cx['follow_rate']:.2f}" if cx else ""))
    text = ("# G9 on BP RoboTwin (split v1)\n\n"
            "matched MSE: 50-step chunk vs the true chunk (normalized delta space); ratios = E_cond / E_matched with a "
            "query-episode bootstrap 95% CI; > 1 means the policy does worse with that demo (it reads the demo).\n\n"
            "| ckpt | group | queries | matched MSE | blank / matched | alt / matched | wrong / matched |\n"
            "| --- | --- | ---: | ---: | --- | --- | --- |\n" + "\n".join(rows) + "\n")
    (out / "summary.md").write_text(text)
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "summary"])
    ap.add_argument("--out", default=str(ROOT / "outputs/BPv4CDX/stage_b/g9_bp"))
    args = ap.parse_args()
    (prepare if args.cmd == "prepare" else summary)(Path(args.out))


if __name__ == "__main__":
    main()
