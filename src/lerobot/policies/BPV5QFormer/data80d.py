"""80-D unified-action data (cleaned root, default /vla/workspace/cache/bpv5_cleaned_root) for BPv5: repos, group norm stats, camera slots, per-episode
meta and states. Decisions: code/conclusion/bpv5/开发文档.md (0.1) and 相机映射待确认.md.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# Override with BPV5_DATA_ROOT / BPV5_STATS_ROOT, or call configure() from the dataset.
ROOT = Path(os.environ.get("BPV5_DATA_ROOT", "/vla/workspace/cache/bpv5_cleaned_root"))
STATS = Path(os.environ.get("BPV5_STATS_ROOT", "/physical-ai-data/norm_stats/action80_new/delta"))
# Optional single stats JSON (tbot 14-D or mixed 80-D). When set, every repo uses this file.
STATS_FILE: Path | None = None
# action80d: team 80-D parquet.
# native14_pad32: AE-3/tbot style — pad native 14-D to 32 (E1).
# native14_scatter80: scatter native 14-D into 80-D semantic slots (E2/E3 path).
ACTION_LAYOUT = "action80d"
# bpv5_80d: the production BPv5 rule from code/conclusion/bpv5/80D_action_norm_stats_delta_abs.md.
# state_mask and tbot_aloha are retained only for historical E1/E2 comparisons.
DELTA_MODE = "bpv5_80d"
IMAGE_PREFIX = "observation.images."
GENERIC_TEXT = "reconstruct the demonstrated action segment"
# dims with std≈0 still need a floor so delta / std stays finite (r1lite torso_joint_3, r1pro base_yaw, …)
STD_EPS = 1e-2
ACTION_DIM = 80

# BPv5 80-D normalization policy. These are semantic slots, not all valid
# state slots: quaternion, gripper, and velocity targets are absolute.
BPV5_DELTA_DIMS = tuple(range(0, 10)) + (29, 39) + tuple(range(40, 50)) + (74, 75)
BPV5_ABS_DIMS = tuple(range(10, 17)) + tuple(range(50, 57)) + tuple(range(77, 80))
NATIVE_ALOHA_WIDTH = 14
PAD32_DIM = 32
# Native RoboTwin 14-D (L6, L gripper, R6, R gripper) laid into the 80-D slots (scatter80 only).
NATIVE_ALOHA_SLOTS = (0, 1, 2, 3, 4, 5, 16, 40, 41, 42, 43, 44, 45, 56)
# tbot MASK_MAPPING aloha = make_bool_mask(6, -1, 6, -1): joints True, grippers False.
# Indices below are in *native 14-D* order; pad32 uses them directly; scatter80 remaps via SLOTS.
NATIVE_ALOHA_DELTA_IDX = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
NATIVE_ALOHA_ABS_IDX = (6, 13)
# 80-D slot indices for the same joints/grippers (scatter80).
NATIVE_ALOHA_DELTA_DIMS = (0, 1, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45)
NATIVE_ALOHA_ABS_DIMS = (16, 56)
_LAYOUTS = {"action80d", "native14_pad32", "native14_scatter80"}


def configure(
    *,
    root: str | Path | None = None,
    stats_root: str | Path | None = None,
    stats_file: str | Path | None = None,
    action_layout: str = "action80d",
    delta_mode: str = "bpv5_80d",
) -> None:
    """Rebind data/stats roots and clear caches. Call once before building the dataset."""
    global ROOT, STATS, STATS_FILE, ACTION_LAYOUT, DELTA_MODE, ACTION_DIM
    if action_layout not in _LAYOUTS:
        raise ValueError(f"action_layout must be {'|'.join(sorted(_LAYOUTS))}, got {action_layout}")
    if delta_mode not in {"bpv5_80d", "state_mask", "tbot_aloha"}:
        raise ValueError(f"delta_mode must be bpv5_80d|state_mask|tbot_aloha, got {delta_mode}")
    if root is not None:
        ROOT = Path(root)
    if stats_root is not None:
        STATS = Path(stats_root)
    STATS_FILE = Path(stats_file) if stats_file else None
    ACTION_LAYOUT = action_layout
    DELTA_MODE = delta_mode
    ACTION_DIM = PAD32_DIM if action_layout == "native14_pad32" else 80
    for fn in (stats_index, group_stats, info, data_files, episodes, episode_tasks, _native_stats_bundle):
        cache = getattr(fn, "cache_clear", None)
        if cache:
            cache()

# camera slots head / left wrist / right wrist per source: accepted camera keys (without the observation.images.
# prefix), first present one wins; single-arm robots use the left slots of the 80-D layout, so their wrist camera
# is the left slot and the right slot stays empty
CAMERAS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "abc": (["mcap.top-camera", "mcap.top-left-camera"], ["mcap.left-wrist-camera"], ["mcap.right-wrist-camera"]),
    "agibot": (["head"], ["hand_left"], ["hand_right"]),
    "dom": (["opst_cam"], ["wrist_cam"], []),
    "galaxea": (["head_rgb"], ["left_wrist_rgb"], ["right_wrist_rgb"]),
    "hifi_umi": (["head_main"], ["left_hand_up"], ["right_hand_up"]),
    "hiw": (["head"], ["left_wrist"], ["right_wrist"]),
    "icra_wbc": (["head_color"], ["hand_left"], ["hand_right"]),
    "interndata": (["images.rgb.head"], ["images.rgb.hand_left", "images.rgb.hand"], ["images.rgb.hand_right"]),
    "maniparena": (["faceImg"], ["leftImg"], ["rightImg"]),
    "openneo": (["third_view"], ["left_wrist_view"], ["right_wrist_view"]),
    "robotwin": (["cam_high"], ["cam_left_wrist"], ["cam_right_wrist"]),
    # robotwin2.5_bp under cleaned_root/robotwin25/<task>/<leaf>
    "robotwin25": (["cam_high"], ["cam_left_wrist"], ["cam_right_wrist"]),
    # DROID-v3 clean: single-arm Franka; exterior≈head, wrist→left, right empty
    "droid": (["exterior_1_left", "exterior_2_left"], ["wrist_left"], []),
    # MolmoAct V3: first_view≈head, wrist→left; second_view unused for wrist slot
    "molmoact": (["first_view"], ["wrist_image"], []),
    "worldengine": (["cam_high"], ["cam_left_wrist"], ["cam_right_wrist"]),
}


@lru_cache(maxsize=1)
def stats_index() -> dict[str, str]:
    """relative repo id -> stats file (relative to STATS), or absolute when STATS_FILE is set."""
    if STATS_FILE is not None:
        # one shared stats file for every readable leaf under ROOT (filtered by sources in repos())
        return {"__single__": str(STATS_FILE)}
    return {k: v for k, v in json.loads((STATS / "stats_index.json").read_text()).items() if not k.startswith("/")}


@lru_cache(maxsize=64)
def group_stats(rel_stats: str) -> dict:
    path = Path(rel_stats)
    return json.loads((path if path.is_absolute() else (STATS / rel_stats)).read_text())


def _iter_leaf_repos() -> list[str]:
    """Discover LeRobot leaves under ROOT (any depth; follows symlinks).

    A leaf is any directory containing meta/info.json. Used for ViT cache item
    lists and for sources that are absent from team stats_index (droid /
    molmoact / robotwin25 / deep InternData paths).
    """
    out: list[str] = []
    if not ROOT.is_dir():
        return out
    for src in sorted(p.name for p in ROOT.iterdir() if p.is_dir() or p.is_symlink()):
        src_path = ROOT / src
        if not src_path.is_dir():
            continue
        for dirpath, dirnames, _filenames in os.walk(src_path, followlinks=True):
            p = Path(dirpath)
            if (p / "meta" / "info.json").is_file():
                out.append(str(p.relative_to(ROOT)))
                dirnames[:] = []  # do not descend into a LeRobot leaf
    return sorted(set(out))


def repos(sources: list[str] | None = None) -> list[str]:
    """Relative repo ids with readable meta.

    Prefer filesystem walk for native14 / single-stats; otherwise union
    stats_index with filesystem so new cleaned-root sources still appear.
    """
    if STATS_FILE is not None or ACTION_LAYOUT.startswith("native14_"):
        candidates = _iter_leaf_repos()
    else:
        candidates = sorted(set(stats_index()) | set(_iter_leaf_repos()))
    out = []
    for rel in candidates:
        if sources and rel.split("/")[0] not in sources:
            continue
        if (ROOT / rel / "meta" / "info.json").is_file():
            out.append(rel)
    return out


@lru_cache(maxsize=256)
def info(rel: str) -> dict:
    return json.loads((ROOT / rel / "meta" / "info.json").read_text())


def camera_slots(rel: str) -> list[str | None]:
    """[head, left wrist, right wrist] video feature keys of this repo (None: no camera in that slot). Some sources
    name their cameras without the observation.images. prefix (abc: mcap.top-camera, interndata: images.rgb.head)."""
    keys = {k[len(IMAGE_PREFIX):] if k.startswith(IMAGE_PREFIX) else k: k
            for k, v in info(rel)["features"].items() if v.get("dtype") in ("video", "image")}
    slots = []
    for accepted in CAMERAS[rel.split("/")[0]]:
        hit = next((k for k in accepted if k in keys), None)
        slots.append(keys[hit] if hit else None)
    return slots


@lru_cache(maxsize=8)
def episodes(rel: str) -> dict[int, dict]:
    """episode_index -> data file / row range and, per video key, file and start timestamp."""
    cams = [c for c in camera_slots(rel) if c]
    cols = ["episode_index", "length", "data/chunk_index", "data/file_index", "dataset_from_index", "dataset_to_index"]
    for c in cams:
        cols += [f"videos/{c}/chunk_index", f"videos/{c}/file_index", f"videos/{c}/from_timestamp"]
    out = {}
    for f in sorted((ROOT / rel / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f, columns=cols).to_pydict()
        for i, ep in enumerate(t["episode_index"]):
            out[int(ep)] = {
                "length": int(t["length"][i]), "from": int(t["dataset_from_index"][i]), "to": int(t["dataset_to_index"][i]),
                "data": (int(t["data/chunk_index"][i]), int(t["data/file_index"][i])),
                "videos": {c: (int(t[f"videos/{c}/chunk_index"][i]), int(t[f"videos/{c}/file_index"][i]),
                               float(t[f"videos/{c}/from_timestamp"][i])) for c in cams},
            }
    return out


def data_path(rel: str, chunk: int, file: int) -> Path:
    return ROOT / rel / info(rel)["data_path"].format(chunk_index=chunk, file_index=file)


def video_path(rel: str, key: str, chunk: int, file: int) -> Path:
    return ROOT / rel / info(rel)["video_path"].format(video_key=key, chunk_index=chunk, file_index=file)


@lru_cache(maxsize=8)
def data_files(rel: str) -> list[Path]:
    return sorted((ROOT / rel / "data").rglob("*.parquet"))


def cache_path(cache_dir: str | Path, rel: str, ep: int) -> Path:
    return Path(cache_dir) / rel / f"ep{int(ep):06d}.pt"


@lru_cache(maxsize=1)
def _native_stats_bundle() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand tbot 14-D stats.json into pad32 or scatter80 vectors + masks."""
    if STATS_FILE is None:
        raise RuntimeError(f"{ACTION_LAYOUT} requires configure(stats_file=...)")
    st = json.loads(Path(STATS_FILE).read_text())
    if "action" not in st or "observation.state" not in st:
        raise ValueError(f"native stats must have action + observation.state keys: {STATS_FILE}")
    a14, s14 = st["action"], st["observation.state"]
    if len(a14["mean"]) != NATIVE_ALOHA_WIDTH or len(s14["mean"]) != NATIVE_ALOHA_WIDTH:
        raise ValueError(
            f"native stats dim {len(a14['mean'])}/{len(s14['mean'])} != {NATIVE_ALOHA_WIDTH}"
        )
    a_mean = np.zeros(ACTION_DIM, dtype=np.float32)
    a_std = np.ones(ACTION_DIM, dtype=np.float32)
    s_mean = np.zeros(ACTION_DIM, dtype=np.float32)
    s_std = np.ones(ACTION_DIM, dtype=np.float32)
    a_mask = np.zeros(ACTION_DIM, dtype=bool)
    s_mask = np.zeros(ACTION_DIM, dtype=bool)
    d_mask = np.zeros(ACTION_DIM, dtype=bool)

    if ACTION_LAYOUT == "native14_pad32":
        # AE-3/tbot: keep native order, zero-pad to 32
        for i in range(NATIVE_ALOHA_WIDTH):
            a_mean[i] = float(a14["mean"][i])
            a_std[i] = max(float(a14["std"][i]), STD_EPS)
            s_mean[i] = float(s14["mean"][i])
            s_std[i] = max(float(s14["std"][i]), STD_EPS)
            a_mask[i] = True
            s_mask[i] = True
        d_mask[list(NATIVE_ALOHA_DELTA_IDX)] = True
    else:
        # scatter into 80-D semantic slots
        for i, slot in enumerate(NATIVE_ALOHA_SLOTS):
            a_mean[slot] = float(a14["mean"][i])
            a_std[slot] = max(float(a14["std"][i]), STD_EPS)
            s_mean[slot] = float(s14["mean"][i])
            s_std[slot] = max(float(s14["std"][i]), STD_EPS)
            a_mask[slot] = True
            s_mask[slot] = True
        d_mask[list(NATIVE_ALOHA_DELTA_DIMS)] = True
    return a_mean, a_std, s_mean, s_std, a_mask, s_mask, d_mask


def norm_vectors(
    rel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return action mean/std, state mean/std, action_mask, state_mask, delta_mask (bool [ACTION_DIM])."""
    if ACTION_LAYOUT.startswith("native14_"):
        return _native_stats_bundle()

    key = "__single__" if STATS_FILE is not None else rel
    if key not in stats_index() and STATS_FILE is None:
        raise KeyError(f"no stats index entry for {rel}")
    st = group_stats(stats_index()[key] if STATS_FILE is None else str(STATS_FILE))
    meta = st.get("metadata") or {}
    if "action_80D" in st:
        a, s = st["action_80D"], st["state_80D"]
        a_mask = np.asarray(meta["action_mask"], dtype=np.float32) > 0.5
        s_mask = np.asarray(meta["state_mask"], dtype=np.float32) > 0.5
    else:
        raise ValueError(f"unsupported stats schema in {STATS_FILE or stats_index().get(rel)}: need action_80D")

    if DELTA_MODE == "bpv5_80d":
        # Production semantics must not silently inherit historical all-state delta stats.
        if meta.get("action_mode") != "mixed" or "delta_mask" not in meta or "abs_mask" not in meta:
            raise ValueError(
                f"{STATS_FILE or stats_index().get(rel)}: BPV5 requires mixed stats with delta_mask and abs_mask"
            )
        d_mask = np.zeros(ACTION_DIM, dtype=bool)
        d_mask[list(BPV5_DELTA_DIMS)] = True
        d_mask &= s_mask
        expected_abs = np.zeros(ACTION_DIM, dtype=bool)
        expected_abs[list(BPV5_ABS_DIMS)] = True
        expected_abs &= a_mask
        declared = np.asarray(meta["delta_mask"], dtype=np.float32) > 0.5
        declared_abs = np.asarray(meta["abs_mask"], dtype=np.float32) > 0.5
        if declared.shape != d_mask.shape or not np.array_equal(declared, d_mask):
            raise ValueError(f"{STATS_FILE or stats_index().get(rel)}: delta_mask conflicts with BPV5 80D policy")
        if declared_abs.shape != expected_abs.shape or not np.array_equal(declared_abs, expected_abs):
            raise ValueError(f"{STATS_FILE or stats_index().get(rel)}: abs_mask conflicts with BPV5 80D policy")
    elif "delta_mask" in meta:
        d_mask = np.asarray(meta["delta_mask"], dtype=np.float32) > 0.5
    elif DELTA_MODE == "tbot_aloha":
        d_mask = np.zeros(ACTION_DIM, dtype=bool)
        d_mask[list(NATIVE_ALOHA_DELTA_DIMS)] = True
    else:
        # old team rule: every state-valid dim is delta
        d_mask = s_mask.copy()

    return (
        np.asarray(a["mean"], dtype=np.float32),
        np.maximum(np.asarray(a["std"], dtype=np.float32), STD_EPS),
        np.asarray(s["mean"], dtype=np.float32),
        np.maximum(np.asarray(s["std"], dtype=np.float32), STD_EPS),
        a_mask,
        s_mask,
        d_mask,
    )

def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _pick_english(text: str) -> str:
    """Prefer non-CJK side when a string mixes Chinese and English (galaxea-style bilingual)."""
    text = str(text).strip()
    if not text or text.lower() == "none":
        return ""
    if _has_cjk(text) and any(c.isascii() and c.isalpha() for c in text):
        for sep in ("\n", " | ", "|", " / ", "/"):
            if sep in text:
                english = [p.strip() for p in text.split(sep) if p.strip() and not _has_cjk(p)]
                if english:
                    return english[0]
    return text


def dom_json_to_text(obj: dict) -> str:
    task = str(obj.get("task", "task"))
    objects = ", ".join(str(x) for x in (obj.get("objects") or []))
    containers = ", ".join(str(x) for x in (obj.get("containers") or []))
    parts = [task]
    if objects:
        parts.append(f"objects: {objects}")
    if containers:
        parts.append(f"containers: {containers}")
    return "; ".join(parts)


def normalize_task_text(raw) -> str:
    """Episode `tasks` field -> one English natural-language string for Stage A."""
    if raw is None:
        return "robot demonstration"
    if isinstance(raw, dict):
        return dom_json_to_text(raw)
    if isinstance(raw, list):
        parts = [normalize_task_text(x) for x in raw]
        parts = [p for p in parts if p and p != "robot demonstration"]
        return " ".join(parts) if parts else "robot demonstration"
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                return dom_json_to_text(json.loads(s))
            except json.JSONDecodeError:
                pass
        return _pick_english(s) or "robot demonstration"
    return _pick_english(str(raw)) or "robot demonstration"


@lru_cache(maxsize=8)
def episode_tasks(rel: str) -> dict[int, object]:
    """episode_index -> raw `tasks` field from meta/episodes."""
    out: dict[int, object] = {}
    for f in sorted((ROOT / rel / "meta" / "episodes").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "tasks"])
        for ep, task in zip(t.column("episode_index").to_pylist(), t.column("tasks").to_pylist()):
            out[int(ep)] = task
    return out


def task_text(rel: str, ep_idx: int) -> str:
    return normalize_task_text(episode_tasks(rel).get(int(ep_idx)))


def delta_chunk(
    state: np.ndarray,
    action: np.ndarray,
    delta_mask: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    start: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """action[t+k] - where(delta_mask, state[t], 0); then mean/std normalize.

    delta_mask True  -> joint-style delta (subtract state at chunk start)
    delta_mask False -> absolute target (gripper / velocity / empty)

    Returns normalized action [H,80], step_valid [H], raw state at t [80].
    """
    length = int(action.shape[0])
    base = np.where(delta_mask, state[start], 0.0).astype(np.float32)
    out = np.zeros((horizon, ACTION_DIM), dtype=np.float32)
    step_valid = np.zeros(horizon, dtype=bool)
    last = np.zeros(ACTION_DIM, dtype=np.float32)
    for k in range(horizon):
        t = start + k
        if t >= length:
            out[k] = last
            step_valid[k] = False
        else:
            last = (action[t] - base).astype(np.float32)
            out[k] = last
            step_valid[k] = True
    out = (out - action_mean) / action_std
    return out, step_valid, state[start].astype(np.float32)


def normalize_state(state_t: np.ndarray, state_mean: np.ndarray, state_std: np.ndarray) -> np.ndarray:
    return ((state_t - state_mean) / state_std).astype(np.float32)


class StateReader:
    """state_80D / state_mask rows of an episode; keeps the loaded data files in memory (episodes are read in order).
    An episode may continue in the next data file."""

    def __init__(self) -> None:
        self._files: dict[Path, tuple] = {}

    def _load(self, path: Path) -> tuple:
        if path not in self._files:
            if len(self._files) >= 2:
                self._files.pop(next(iter(self._files)))
            names = set(pq.read_schema(path).names)
            if "state_80D" in names:
                t = pq.read_table(path, columns=["index", "state_80D", "state_mask"])
                self._files[path] = (t.column("index").to_numpy(),
                                     np.stack(t.column("state_80D").to_numpy(zero_copy_only=False)).astype(np.float32),
                                     np.stack(t.column("state_mask").to_numpy(zero_copy_only=False)) > 0)
            elif "observation.state" in names:
                t = pq.read_table(path, columns=["index", "observation.state"])
                raw = np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
                state, mask = _layout_native14(raw, with_mask=True)
                self._files[path] = (t.column("index").to_numpy(), state, mask)
            else:
                raise ValueError(f"{path}: neither state_80D nor observation.state")
        return self._files[path]

    def episode(self, rel: str, ep: dict) -> tuple[np.ndarray, np.ndarray]:
        files = data_files(rel)
        pos = files.index(data_path(rel, *ep["data"]))
        states, mask, need = [], None, ep["from"]
        while need < ep["to"]:
            if pos >= len(files):
                raise ValueError(f"{rel}: rows {need}..{ep['to']} missing")
            index, state, m = self._load(files[pos])
            lo = int(np.searchsorted(index, need))
            hi = int(np.searchsorted(index, ep["to"]))
            if lo >= len(index) or index[lo] != need or (hi - lo) != int(index[hi - 1] - need + 1):
                raise ValueError(f"{rel}: rows {need}..{ep['to']} not contiguous in {files[pos].name}")
            states.append(state[lo:hi])
            mask = m[lo] if mask is None else mask
            need += hi - lo
            pos += 1
        return np.concatenate(states), mask


def _layout_native14(raw: np.ndarray, *, with_mask: bool = False):
    """Map native [T,14] into current ACTION_DIM (pad32 or scatter80)."""
    if raw.ndim != 2 or raw.shape[1] != NATIVE_ALOHA_WIDTH:
        raise ValueError(f"native vector shape {raw.shape}, want [T, {NATIVE_ALOHA_WIDTH}]")
    raw = raw.astype(np.float32)
    out = np.zeros((len(raw), ACTION_DIM), dtype=np.float32)
    if ACTION_LAYOUT == "native14_pad32":
        out[:, :NATIVE_ALOHA_WIDTH] = raw
        if with_mask:
            mask = np.zeros((len(raw), ACTION_DIM), dtype=bool)
            mask[:, :NATIVE_ALOHA_WIDTH] = True
            return out, mask
        return out
    # scatter80 (default for native when not pad32)
    out[:, NATIVE_ALOHA_SLOTS] = raw
    if with_mask:
        mask = np.zeros((len(raw), ACTION_DIM), dtype=bool)
        mask[:, NATIVE_ALOHA_SLOTS] = True
        return out, mask
    return out


class FrameReader:
    """Per-episode state/action: action_80D, or native 14-D (pad32 / scatter80)."""

    def __init__(self) -> None:
        self._files: dict[Path, tuple] = {}

    def _load(self, path: Path) -> tuple:
        if path not in self._files:
            if len(self._files) >= 2:
                self._files.pop(next(iter(self._files)))
            names = set(pq.read_schema(path).names)
            use_native = ACTION_LAYOUT.startswith("native14_") or (
                "action_80D" not in names and "observation.state" in names and "action" in names
            )
            if use_native:
                t = pq.read_table(path, columns=["index", "observation.state", "action"])
                state = _layout_native14(np.stack(t.column("observation.state").to_numpy(zero_copy_only=False)))
                action = _layout_native14(np.stack(t.column("action").to_numpy(zero_copy_only=False)))
                self._files[path] = (t.column("index").to_numpy(), state, action)
            elif "action_80D" in names and "state_80D" in names:
                t = pq.read_table(path, columns=["index", "state_80D", "action_80D"])
                self._files[path] = (
                    t.column("index").to_numpy(),
                    np.stack(t.column("state_80D").to_numpy(zero_copy_only=False)).astype(np.float32),
                    np.stack(t.column("action_80D").to_numpy(zero_copy_only=False)).astype(np.float32),
                )
            else:
                raise ValueError(f"{path}: need action_80D/state_80D or observation.state/action")
        return self._files[path]

    def episode(self, rel: str, ep: dict) -> tuple[np.ndarray, np.ndarray]:
        files = data_files(rel)
        pos = files.index(data_path(rel, *ep["data"]))
        states, actions, need = [], [], ep["from"]
        while need < ep["to"]:
            if pos >= len(files):
                raise ValueError(f"{rel}: rows {need}..{ep['to']} missing")
            index, state, action = self._load(files[pos])
            lo = int(np.searchsorted(index, need))
            hi = int(np.searchsorted(index, ep["to"]))
            if lo >= len(index) or index[lo] != need or (hi - lo) != int(index[hi - 1] - need + 1):
                raise ValueError(f"{rel}: rows {need}..{ep['to']} not contiguous in {files[pos].name}")
            states.append(state[lo:hi])
            actions.append(action[lo:hi])
            need += hi - lo
            pos += 1
        return np.concatenate(states), np.concatenate(actions)
