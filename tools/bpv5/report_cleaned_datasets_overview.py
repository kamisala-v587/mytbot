#!/usr/bin/env python3
"""Overview stats + figures for cleaned datasets (+ MolmoAct + Droid-in-progress)."""
from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/vla/workspace/code/conclusion/bpv5")
ASSETS = OUT / "assets" / "cleaned_data_overview_20261116"
REC = OUT / "records" / "cleaned_data_overview_20261116"
ACCEPT = OUT / "records" / "cleaned_accept_20261116" / "accept_p0_results.json"

SOURCES = {
    "AgiBotWorld": {
        "root": Path("/physical-ai-data/cleaned_dataset/agibotworld/full-20261106-1943"),
        "kind_default": "real",
        "pipeline": "cleaned_lerobot_v3_80d",
    },
    "RoboTwin": {
        "root": Path("/physical-ai-data/cleaned_dataset/RoboTwin-LeRobot-v3.0/full-20261105-1329"),
        "kind_default": "sim",
        "pipeline": "cleaned_lerobot_v3_80d",
    },
    "InternData-A1": {
        "root": Path("/physical-ai-data/cleaned_dataset/full-InternData-A1-v3.0/full-20261114-2126"),
        "kind_default": "sim",
        "pipeline": "cleaned_lerobot_v3_80d",
    },
    "DROID-v3": {
        "root": Path("/physical-ai-data/cleaned_dataset/droid-v3/data_clean2.0-20261114"),
        "kind_default": "real",
        "pipeline": "cleaning_in_progress",
    },
    "MolmoAct": {
        "root": Path("/physical-ai-data/download_tmp_dengjun/allenai__MolmoAct-Dataset"),
        "kind_default": "real",
        "pipeline": "raw_hf_lerobot_not_80d",
    },
}

RECENT_H = 48  # mark if root/marker touched within this many hours


def shape_dim(shape) -> int | None:
    if not shape:
        return None
    if isinstance(shape, list):
        n = 1
        for x in shape:
            n *= int(x)
        return n
    return None


def native_action_dim(feats: dict) -> int | None:
    prefer = [
        "actions.joint.position",
        "action",
        "actions",
        "actions.effector.position",
    ]
    for k in prefer:
        if k in feats:
            d = shape_dim(feats[k].get("shape"))
            if d:
                return d
    # fallback: first actions.* float vector
    for k, v in feats.items():
        if k.startswith("actions.") and v.get("dtype") in ("float32", "float64"):
            d = shape_dim(v.get("shape"))
            if d:
                return d
    return None


def freshness(path: Path) -> dict:
    now = time.time()
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    age_h = (now - st.st_mtime) / 3600
    markers = {}
    for name in ("_SUCCESS", "cleaning-result.json", ".data_clean", "IN_PROGRESS"):
        p = path / name
        if p.exists():
            markers[name] = {
                "mtime": time.strftime("%F %T", time.localtime(p.stat().st_mtime)),
                "age_h": round((now - p.stat().st_mtime) / 3600, 2),
            }
    # newest staging sibling for droid
    staging = []
    parent = path.parent if path.name.startswith("data_clean") else path
    if parent.exists():
        for c in parent.iterdir():
            if ".staging" in c.name or c.name.startswith(".data_clean"):
                try:
                    staging.append(
                        {
                            "name": c.name,
                            "mtime": time.strftime("%F %T", time.localtime(c.stat().st_mtime)),
                            "age_h": round((now - c.stat().st_mtime) / 3600, 2),
                        }
                    )
                except OSError:
                    pass
    staging.sort(key=lambda x: x["age_h"])
    return {
        "exists": True,
        "root_mtime": time.strftime("%F %T", time.localtime(st.st_mtime)),
        "root_age_h": round(age_h, 2),
        "recent": age_h <= RECENT_H or any(m["age_h"] <= RECENT_H for m in markers.values()),
        "markers": markers,
        "staging_newest": staging[:5],
    }


def scan_cleaned(name: str, root: Path, kind_default: str) -> dict:
    fr = freshness(root)
    cr_path = root / "cleaning-result.json"
    out = {
        "name": name,
        "root": str(root),
        "freshness": fr,
        "status": "MISSING",
        "n_repos": 0,
        "n_episodes": 0,
        "n_frames": 0,
        "has_80d": False,
        "native_action_dims": Counter(),
        "robot_types": Counter(),  # weighted by episodes
        "robot_types_repos": Counter(),
        "kind": kind_default,
        "fps": Counter(),
        "leaf_status": Counter(),
        "sample_repos": [],
    }
    if not root.exists():
        return out
    if not cr_path.exists():
        out["status"] = "NO_CLEANING_RESULT"
        return out
    cr = json.loads(cr_path.read_text())
    out["status"] = cr.get("status") or "UNKNOWN"
    out["n_episodes_claimed"] = cr.get("episodes")
    leaves = cr.get("leaves") or []
    out["n_repos"] = len(leaves)
    frames = 0
    eps = 0
    n_80 = 0
    for leaf in leaves:
        out["leaf_status"][leaf.get("status") or "?"] += 1
        e = int(leaf.get("episodes") or 0)
        f = int(leaf.get("frames") or 0)
        eps += e
        frames += f
        rel = leaf.get("output") or leaf.get("leaf")
        info_p = root / rel / "meta" / "info.json"
        robot = "?"
        native = None
        has80 = False
        fps = None
        if info_p.is_file():
            try:
                meta = json.loads(info_p.read_text())
                robot = meta.get("robot_type") or "?"
                fps = meta.get("fps")
                feats = meta.get("features") or {}
                has80 = "action_80D" in feats and "state_80D" in feats and "action_mask" in feats
                native = native_action_dim(feats)
                # prefer info totals if present
                e = int(meta.get("total_episodes") or e)
                f = int(meta.get("total_frames") or f)
            except Exception:
                pass
        out["robot_types"][str(robot)] += e
        out["robot_types_repos"][str(robot)] += 1
        if native is not None:
            out["native_action_dims"][int(native)] += e
        if fps is not None:
            out["fps"][int(fps)] += e
        if has80:
            n_80 += 1
        if len(out["sample_repos"]) < 5:
            out["sample_repos"].append(
                {"rel": rel, "episodes": e, "frames": f, "robot": robot, "native_dim": native, "has_80d": has80}
            )
    out["n_episodes"] = eps
    out["n_frames"] = frames
    out["has_80d"] = n_80 == len(leaves) and len(leaves) > 0
    out["n_repos_with_80d"] = n_80
    out["n_repos_info_ok"] = n_80  # approx; recount below if needed
    # recount info-readable
    info_ok = 0
    for leaf in leaves:
        rel = leaf.get("output") or leaf.get("leaf")
        if (root / rel / "meta" / "info.json").is_file():
            info_ok += 1
    out["n_repos_info_ok"] = info_ok
    out["has_80d"] = n_80 == info_ok and info_ok > 0
    if fr.get("recent") and out["status"] == "COMPLETED":
        out["convert_note"] = "RECENTLY_FINISHED"
    elif fr.get("recent"):
        out["convert_note"] = "RECENT_ACTIVITY"
    else:
        out["convert_note"] = "STABLE"
    return out


def scan_molmoact(name: str, root: Path, kind_default: str) -> dict:
    fr = freshness(root)
    out = {
        "name": name,
        "root": str(root),
        "freshness": fr,
        "status": "RAW_HF" if root.exists() else "MISSING",
        "n_repos": 0,
        "n_episodes": 0,
        "n_frames": 0,
        "has_80d": False,
        "native_action_dims": Counter(),
        "robot_types": Counter(),
        "robot_types_repos": Counter(),
        "kind": kind_default,
        "fps": Counter(),
        "leaf_status": Counter({"RAW": 0}),
        "sample_repos": [],
        "convert_note": "NOT_IN_CLEAN_PIPELINE",
        "pipeline": "raw_hf_lerobot_not_80d",
    }
    if not root.exists():
        return out
    repos = []
    for split in ("molmoact_dataset_tabletop/train", "molmoact_dataset_household/train"):
        info = root / split / "meta" / "info.json"
        if info.is_file():
            repos.append((split, info))
    out["n_repos"] = len(repos)
    for rel, info in repos:
        meta = json.loads(info.read_text())
        e = int(meta.get("total_episodes") or 0)
        f = int(meta.get("total_frames") or 0)
        robot = meta.get("robot_type") or "?"
        feats = meta.get("features") or {}
        native = native_action_dim(feats) or shape_dim((feats.get("actions") or {}).get("shape"))
        has80 = "action_80D" in feats
        out["n_episodes"] += e
        out["n_frames"] += f
        out["robot_types"][str(robot)] += e
        out["robot_types_repos"][str(robot)] += 1
        if native is not None:
            out["native_action_dims"][int(native)] += e
        if meta.get("fps") is not None:
            out["fps"][int(meta["fps"])] += e
        out["leaf_status"]["RAW"] += 1
        out["sample_repos"].append(
            {"rel": rel, "episodes": e, "frames": f, "robot": robot, "native_dim": native, "has_80d": has80}
        )
    out["has_80d"] = False
    out["n_repos_with_80d"] = 0
    out["n_repos_info_ok"] = len(repos)
    return out


def scan_droid(name: str, root: Path, kind_default: str) -> dict:
    fr = freshness(root)
    out = {
        "name": name,
        "root": str(root),
        "freshness": fr,
        "status": "CONVERTING",
        "n_repos": 0,
        "n_episodes": 0,
        "n_frames": 0,
        "has_80d": False,
        "native_action_dims": Counter(),
        "robot_types": Counter({"Franka (expected)": 0}),
        "robot_types_repos": Counter(),
        "kind": kind_default,
        "fps": Counter(),
        "leaf_status": Counter(),
        "sample_repos": [],
        "convert_note": "CONVERTING_NOW",
        "pipeline": "cleaning_in_progress",
        "droid_progress": {},
    }
    # published leaves?
    infos = []
    if root.exists():
        for p in root.iterdir():
            if p.name.startswith("."):
                continue
            if (p / "meta" / "info.json").is_file():
                infos.append(p)
    out["n_repos_published"] = len(infos)
    db = root / ".data_clean" / "results.sqlite"
    if db.is_file():
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = con.execute("select processed, kept, reviewed, rejected from summary").fetchone()
            if row:
                out["droid_progress"] = {
                    "processed": row[0],
                    "kept": row[1],
                    "reviewed": row[2],
                    "rejected": row[3],
                }
            dec = dict(con.execute("select decision, count(*) from decisions group by decision").fetchall())
            out["droid_progress"]["decisions"] = dec
            mn, avg, mx, sm = con.execute(
                "select min(length), avg(length), max(length), sum(length) from episodes"
            ).fetchone()
            out["n_episodes"] = int(con.execute("select count(*) from episodes").fetchone()[0])
            out["n_frames"] = int(sm or 0)
            out["droid_progress"]["length_min"] = mn
            out["droid_progress"]["length_avg"] = float(avg) if avg is not None else None
            out["droid_progress"]["length_max"] = mx
            # no published repos yet
            out["robot_types"] = Counter({"Franka (expected, not yet exported)": out["n_episodes"]})
            out["native_action_dims"] = Counter({"TBD_until_export": out["n_episodes"]})
        finally:
            con.close()
    # active heartbeats
    hb = []
    cluster = root / ".data_clean" / "cluster"
    if cluster.is_dir():
        for p in cluster.glob("heartbeat-*.json"):
            try:
                hb.append({"file": p.name, "age_h": round((time.time() - p.stat().st_mtime) / 3600, 3)})
            except OSError:
                pass
    out["droid_progress"]["heartbeats"] = hb
    out["freshness"]["recent"] = True
    return out


def counter_to_dict(c: Counter) -> dict:
    return {str(k): int(v) for k, v in c.most_common()}


def plot_bars(rows: list[dict], assets: Path):
    assets.mkdir(parents=True, exist_ok=True)
    names = [r["name"] for r in rows]
    eps = [r["n_episodes"] for r in rows]
    frames = [r["n_frames"] for r in rows]
    repos = [r["n_repos"] if r["n_repos"] else r.get("n_repos_published", 0) for r in rows]

    def bar(vals, title, fname, ylabel, colors=None):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x = np.arange(len(names))
        cols = colors or ["#3b6ea5"] * len(names)
        # mark converting
        for i, r in enumerate(rows):
            if r.get("convert_note") in ("CONVERTING_NOW",):
                cols[i] = "#c45c26"
            elif r.get("convert_note") == "RECENTLY_FINISHED":
                cols[i] = "#2a9d8f"
            elif r.get("convert_note") == "NOT_IN_CLEAN_PIPELINE":
                cols[i] = "#6c757d"
        ax.bar(x, vals, color=cols)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(assets / fname, dpi=140)
        plt.close(fig)

    bar(eps, "Episodes (trajectories) by dataset", "F1_episodes.png", "episodes")
    bar(frames, "Frames by dataset", "F2_frames.png", "frames")
    bar(repos, "Repos (leaves) by dataset", "F3_repos.png", "repos")

    # stacked robot share for cleaned completed
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    cleaned = [r for r in rows if r.get("pipeline", "").startswith("cleaned") or r["name"] in ("AgiBotWorld", "RoboTwin", "InternData-A1")]
    for ax, r in zip(axes, cleaned):
        rt = r["robot_types"]
        if isinstance(rt, Counter):
            items = rt.most_common(8)
        else:
            items = list(Counter(rt).most_common(8))
        if not items:
            ax.set_title(r["name"])
            continue
        labels, vals = zip(*items)
        ax.pie(vals, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 7})
        ax.set_title(f"{r['name']} robots\n(by episodes)")
    fig.suptitle("Robot type share (episode-weighted)")
    fig.tight_layout()
    fig.savefig(assets / "F4_robot_share.png", dpi=140)
    plt.close(fig)

    # real vs sim
    fig, ax = plt.subplots(figsize=(7, 4))
    kinds = defaultdict(int)
    for r in rows:
        if r["n_episodes"] <= 0:
            continue
        kinds[r["kind"]] += r["n_episodes"]
    labels = list(kinds.keys())
    vals = [kinds[k] for k in labels]
    ax.bar(labels, vals, color=["#2a9d8f", "#3b6ea5"][: len(labels)])
    ax.set_title("Real vs Sim episode mass (all listed sources)")
    ax.set_ylabel("episodes")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(assets / "F5_real_sim.png", dpi=140)
    plt.close(fig)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    REC.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, cfg in SOURCES.items():
        root = cfg["root"]
        if name == "MolmoAct":
            r = scan_molmoact(name, root, cfg["kind_default"])
        elif name == "DROID-v3":
            r = scan_droid(name, root, cfg["kind_default"])
        else:
            r = scan_cleaned(name, root, cfg["kind_default"])
        r["pipeline"] = cfg["pipeline"]
        # serialize counters
        serial = dict(r)
        for k in ("native_action_dims", "robot_types", "robot_types_repos", "fps", "leaf_status"):
            if isinstance(serial.get(k), Counter):
                serial[k] = counter_to_dict(serial[k])
        rows.append(serial)
        print(name, serial.get("status"), "repos", serial.get("n_repos"), "eps", serial.get("n_episodes"), "note", serial.get("convert_note"))

    plot_bars(
        [
            {
                **r,
                "robot_types": Counter(r["robot_types"]),
                "pipeline": r.get("pipeline", ""),
            }
            for r in rows
        ],
        ASSETS,
    )

    accept = json.loads(ACCEPT.read_text()) if ACCEPT.exists() else {}
    payload = {
        "generated_at": time.strftime("%F %T"),
        "sources": rows,
        "accept_p0p1": {k: {"overall": v.get("overall"), "checks": v.get("checks")} for k, v in accept.items()},
    }
    (REC / "overview.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print("wrote", REC / "overview.json")
    return payload


if __name__ == "__main__":
    main()
