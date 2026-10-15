#!/usr/bin/env python3
"""Extract and analyze fixed BPV3 behavior-prompt encoder representations."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def repo_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def task_from_repo(repo: str) -> str:
    path = Path(repo)
    if path.parent.name:
        return path.parent.name
    raise ValueError(f"无法从 repo 路径推断任务名: {repo}")


def digest_json(payload: Any) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def output_dir(cfg: dict[str, Any], *, smoke_test: bool = False) -> Path:
    base = Path(cfg["output_dir"]).expanduser().resolve()
    return base.with_name(base.name + "_smoke") if smoke_test else base


def checkpoint_meta(cfg: dict[str, Any], *, smoke_test: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    checkpoints = list(cfg["checkpoints"])
    if smoke_test:
        checkpoints = checkpoints[: int(cfg.get("smoke_checkpoints", 1))]
    for item in checkpoints:
        path = Path(item["path"]).expanduser().resolve()
        config_path = path / "config.json"
        train_config_path = path / "train_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"checkpoint 缺少 config.json: {path}")
        if not any((path / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
            raise FileNotFoundError(f"checkpoint 缺少实际权重: {path}")
        model_config = read_json(config_path)
        train_config = read_json(train_config_path) if train_config_path.is_file() else {}
        rows.append(
            {
                "name": str(item["name"]),
                "path": str(path),
                "model_config": model_config,
                "train_config": train_config,
                "fingerprint": digest_json(
                    {
                        "name": item["name"],
                        "path": str(path),
                        "config": model_config,
                        "train_config": train_config,
                    }
                ),
            }
        )
    return rows


def dataset_plan(cfg: dict[str, Any], *, smoke_test: bool = False) -> list[dict[str, Any]]:
    repos = repo_list(Path(cfg["data_repo_file"]).expanduser())
    clean = [repo for repo in repos if "clean" in Path(repo).name]
    if not clean:
        raise ValueError(f"未在 {cfg['data_repo_file']} 找到 clean repo")
    by_task = {task_from_repo(repo): repo for repo in clean}
    tasks = sorted(by_task)
    samples = int(cfg["samples_per_task"])
    if smoke_test:
        tasks = tasks[: int(cfg.get("smoke_tasks", 2))]
        samples = min(samples, int(cfg.get("smoke_samples_per_task", 2)))
    return [{"task": task, "repo": by_task[task], "samples": samples} for task in tasks]


def manifest_fingerprint(cfg: dict[str, Any], *, smoke_test: bool = False) -> str:
    return digest_json(
        {
            "data_repo_file": str(Path(cfg["data_repo_file"]).expanduser().resolve()),
            "repos": dataset_plan(cfg, smoke_test=smoke_test),
            "seed": int(cfg["seed"]),
            "samples_per_task": int(cfg["samples_per_task"]),
            "smoke_test": bool(smoke_test),
        }
    )


def make_manifest(cfg: dict[str, Any], output: Path, *, smoke_test: bool = False) -> list[dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    manifest = output / "manifest.jsonl"
    expected_fp = manifest_fingerprint(cfg, smoke_test=smoke_test)
    if manifest.is_file():
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows and rows[0].get("manifest_fingerprint") == expected_fp:
            return rows
        print(f"[manifest] existing manifest fingerprint mismatch, rebuilding: {manifest}", flush=True)

    rng = random.Random(int(cfg["seed"]))
    rows: list[dict[str, Any]] = []
    for item in dataset_plan(cfg, smoke_test=smoke_test):
        ds = LeRobotDataset(
            item["repo"],
            video_backend=cfg.get("video_backend", "torchcodec"),
            skip_video_file_validation=True,
        )
        total = int(ds.meta.total_episodes)
        if total <= 0:
            raise ValueError(f"repo 没有 episode: {item['repo']}")
        episode_ids = list(range(total))
        rng.shuffle(episode_ids)
        selected = sorted(episode_ids[: min(item["samples"], total)])
        for episode in selected:
            rows.append(
                {
                    "manifest_fingerprint": expected_fp,
                    "task": item["task"],
                    "repo": item["repo"],
                    "episode": int(episode),
                    "bp_repo": item["repo"],
                    "bp_episode": int(episode),
                }
            )
        del ds
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return rows


def select_policy_value(policy_cfg: Any, cfg: dict[str, Any], key: str, default: Any = None) -> Any:
    return getattr(policy_cfg, key, cfg.get(key, default))


def dataset_cfg(cfg: dict[str, Any], repo: str, policy_cfg: Any):
    from lerobot.policies.BPVAv3.configuration_bpvav3 import BPVAv3DatasetConfig

    resolution = select_policy_value(policy_cfg, cfg, "image_resolution", [cfg.get("height", 224), cfg.get("width", 224)])
    return BPVAv3DatasetConfig(
        repo_id=repo,
        action_mode=str(cfg.get("action_mode", "delta")),
        height=int(resolution[0]),
        width=int(resolution[1]),
        max_state_dim=int(select_policy_value(policy_cfg, cfg, "max_state_dim", 32)),
        max_action_dim=int(select_policy_value(policy_cfg, cfg, "max_action_dim", 32)),
        current_image_index=0,
        qwen3_5_processor_path=str(select_policy_value(policy_cfg, cfg, "qwen3_5_processor_path")),
        qwen3_5_processor_revision=select_policy_value(policy_cfg, cfg, "qwen3_5_processor_revision", None),
        use_language=bool(select_policy_value(policy_cfg, cfg, "use_language", False)),
        bp_num_chunks=int(select_policy_value(policy_cfg, cfg, "bp_num_chunks", cfg["bp_num_chunks"])),
        bp_fixed_num_chunks=bool(select_policy_value(policy_cfg, cfg, "bp_fixed_num_chunks", False)),
        bp_seed=int(cfg["seed"]),
        bp_sampling_mode="origin",
        align_bp_encoder=False,
        bp_action_use=bool(cfg.get("bp_action_use", False)),
        bp_state_use=bool(cfg.get("bp_state_use", True)),
        bp_camera_keys=list(select_policy_value(policy_cfg, cfg, "bp_camera_keys", cfg["bp_camera_keys"])),
    )


def load_policy(ckpt: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    config = PreTrainedConfig.from_pretrained(ckpt, local_files_only=True)
    cls = get_policy_class(config.type)
    policy = cls.from_pretrained(ckpt, config=config, local_files_only=True)
    requested = str(getattr(config, "device", ""))
    if requested.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(requested if requested and requested != "None" else ("cuda" if torch.cuda.is_available() else "cpu"))
    policy.to(device).eval()
    return policy, config, device


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(val, device) for key, val in value.items()}
    if isinstance(value, list):
        return [move_to_device(val, device) for val in value]
    return value


def action_delta_timestamps(repo: str, chunk_size: int) -> dict[str, list[float]]:
    """Match the prompt action window used by the BPV3 dataset factory."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.transforms.constants import get_feature_mapping
    from lerobot.utils.constants import ACTION

    meta = LeRobotDatasetMetadata(repo)
    action_keys = set(get_feature_mapping(meta.robot_type, meta.features).get(ACTION, []))
    action_keys.update(key for key in meta.features if key == ACTION)
    if not action_keys:
        raise KeyError(f"无法在 repo 中找到 action feature: {repo}")
    return {key: [step / meta.fps for step in range(chunk_size)] for key in sorted(action_keys)}


def row_input_fingerprint(row: dict[str, Any], cfg: dict[str, Any], policy_cfg: Any) -> str:
    payload = {
        "repo": row["repo"],
        "episode": row["episode"],
        "bp_repo": row["bp_repo"],
        "bp_episode": row["bp_episode"],
        "bp_num_chunks": int(select_policy_value(policy_cfg, cfg, "bp_num_chunks", cfg["bp_num_chunks"])),
        "bp_action_chunk_size": int(select_policy_value(policy_cfg, cfg, "bp_action_chunk_size", cfg["bp_action_chunk_size"])),
        "bp_action_use": bool(cfg.get("bp_action_use", False)),
        "bp_state_use": bool(cfg.get("bp_state_use", True)),
        "bp_camera_keys": list(select_policy_value(policy_cfg, cfg, "bp_camera_keys", cfg["bp_camera_keys"])),
        "processor": str(select_policy_value(policy_cfg, cfg, "qwen3_5_processor_path")),
        "processor_revision": select_policy_value(policy_cfg, cfg, "qwen3_5_processor_revision", None),
    }
    return digest_json(payload)


def build_prompt(
    row: dict[str, Any],
    cfg: dict[str, Any],
    policy_cfg: Any,
    dataset_cache: dict[tuple[str, str], Any],
    repo_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.BPVAv3.dataset import make_bpvav3_behavior_dataset

    repo = str(row["bp_repo"])
    processor_key = str(select_policy_value(policy_cfg, cfg, "qwen3_5_processor_path"))
    cache_key = (repo, processor_key)
    if cache_key not in dataset_cache:
        episodes = sorted({int(item["bp_episode"]) for item in repo_rows})
        action_chunk_size = int(
            select_policy_value(policy_cfg, cfg, "bp_action_chunk_size", cfg["bp_action_chunk_size"])
        )
        delta_timestamps = action_delta_timestamps(repo, action_chunk_size)
        current_ds = LeRobotDataset(
            repo,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.get("video_backend", "torchcodec"),
            skip_video_file_validation=True,
        )
        prompt_ds = LeRobotDataset(
            repo,
            episodes=episodes,
            delta_timestamps=delta_timestamps,
            video_backend=cfg.get("video_backend", "torchcodec"),
            skip_video_file_validation=True,
        )
        dcfg = dataset_cfg(cfg, repo, policy_cfg)
        dataset_cache[cache_key] = make_bpvav3_behavior_dataset(
            current_ds,
            prompt_ds,
            dcfg,
            bp_action_chunk_size=action_chunk_size,
        )

    wrapped = dataset_cache[cache_key]
    episode = int(row["bp_episode"])
    try:
        local_start, local_end, _ = wrapped._episode_ranges[episode]
    except KeyError as exc:
        raise KeyError(f"episode={episode} 不在已缓存的 repo 数据集: {repo}") from exc
    sample = wrapped[int(local_start)]
    return sample["behavior_prompt"], int(local_end - local_start)


def pool_tokens(tokens: torch.Tensor, mask: torch.Tensor, num_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tokens.ndim != 2 or mask.ndim != 1:
        raise ValueError(f"unexpected token/mask shape: {tuple(tokens.shape)}, {tuple(mask.shape)}")
    if tokens.shape[0] % num_chunks != 0:
        raise ValueError(f"token count {tokens.shape[0]} is not divisible by num_chunks={num_chunks}")
    queries = tokens.shape[0] // num_chunks
    chunk_tokens = tokens.reshape(num_chunks, queries, tokens.shape[-1])
    chunk_mask = mask.reshape(num_chunks, queries)
    masked = chunk_tokens.masked_fill(~chunk_mask[..., None], 0)
    denom = chunk_mask.sum(dim=1).clamp_min(1).to(masked.dtype).unsqueeze(-1)
    chunk_mean = masked.sum(dim=1) / denom
    global_denom = mask.sum().clamp_min(1).to(tokens.dtype)
    global_mean = tokens.masked_fill(~mask[:, None], 0).sum(dim=0) / global_denom
    return chunk_mean, global_mean


def extraction_signature(cfg: dict[str, Any], rows: list[dict[str, Any]], ckpts: list[dict[str, Any]], smoke_test: bool) -> str:
    return digest_json(
        {
            "config_path": cfg.get("_config_path"),
            "rows": rows,
            "checkpoints": [{"name": ck["name"], "path": ck["path"], "fingerprint": ck["fingerprint"]} for ck in ckpts],
            "smoke_test": smoke_test,
        }
    )


def shard_signature(rows: list[dict[str, Any]], ck: dict[str, Any], smoke_test: bool) -> str:
    return digest_json(
        {
            "rows": rows,
            "checkpoint": {"name": ck["name"], "path": ck["path"], "fingerprint": ck["fingerprint"]},
            "smoke_test": smoke_test,
        }
    )


def shard_path(output: Path, name: str) -> Path:
    return output / "shards" / f"{name}.pt"


def shard_is_valid(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        existing = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[extract] unreadable shard, rebuilding: {path} ({exc})", flush=True)
        return False
    return existing.get("signature") == signature


def extract_checkpoint(cfg: dict[str, Any], rows: list[dict[str, Any]], ck: dict[str, Any], output: Path, *, smoke_test: bool = False) -> Path:
    """Encode every manifest row with one checkpoint and save it as an independent shard."""
    path = shard_path(output, ck["name"])
    signature = shard_signature(rows, ck, smoke_test)
    if shard_is_valid(path, signature):
        print(f"[extract] shard already matches signature: {path}", flush=True)
        return path
    print(f"[extract] {ck['name']}", flush=True)
    policy, pcfg, device = load_policy(ck["path"])
    num_chunks = int(select_policy_value(pcfg, cfg, "bp_num_chunks", cfg["bp_num_chunks"]))
    action_chunk_size = int(select_policy_value(pcfg, cfg, "bp_action_chunk_size", cfg["bp_action_chunk_size"]))
    tokens_all: list[torch.Tensor] = []
    masks_all: list[torch.Tensor] = []
    chunks_all: list[torch.Tensor] = []
    globals_all: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    dataset_cache: dict[tuple[str, str], Any] = {}
    rows_by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_repo.setdefault(str(row["bp_repo"]), []).append(row)
    for idx, row in enumerate(rows):
        repo = str(row["bp_repo"])
        prompt, trajectory_len = build_prompt(row, cfg, pcfg, dataset_cache, rows_by_repo[repo])
        prompt = move_to_device(prompt, device)
        with torch.inference_mode():
            tokens, mask = policy.model.encode_bp_prompt(prompt)
        tokens = tokens[0].float().cpu()
        mask = mask[0].cpu().bool()
        chunk_mean, global_mean = pool_tokens(tokens, mask, num_chunks)
        meta = dict(row)
        meta["input_fingerprint"] = row_input_fingerprint(row, cfg, pcfg)
        meta["bp_trajectory_len"] = trajectory_len
        meta["bp_natural_chunks"] = min(math.ceil(trajectory_len / action_chunk_size), num_chunks)
        tokens_all.append(tokens.masked_fill(~mask[:, None], 0))
        masks_all.append(mask)
        chunks_all.append(chunk_mean)
        globals_all.append(global_mean)
        metadata.append(meta)
        if (idx + 1) % int(cfg.get("progress_every", 25)) == 0:
            print(f"  [{ck['name']}] {idx + 1}/{len(rows)}", flush=True)
    run = {
        "name": ck["name"],
        "path": ck["path"],
        "checkpoint_fingerprint": ck["fingerprint"],
        "checkpoint_summary": summarize_checkpoint(ck),
        "tokens": torch.stack(tokens_all),
        "token_mask": torch.stack(masks_all),
        "chunks": torch.stack(chunks_all),
        "global_mean": torch.stack(globals_all),
        "metadata": metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save({"signature": signature, "run": run}, tmp)
    tmp.replace(path)
    del policy
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path


def run_workers(cfg: dict[str, Any], names: list[str], gpus: list[str], output: Path, *, smoke_test: bool) -> None:
    """Dynamic scheduler: each GPU runs one checkpoint subprocess at a time."""
    import subprocess
    import time

    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    queue = list(names)
    running: dict[str, tuple[subprocess.Popen, str, Any, float]] = {}
    failed: list[str] = []
    free = list(gpus)
    while queue or running:
        while queue and free:
            name, gpu = queue.pop(0), free.pop(0)
            cmd = [sys.executable, str(Path(__file__).resolve()), "--config", cfg["_config_path"], "--worker-checkpoint", name]
            if smoke_test:
                cmd.append("--smoke-test")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
            env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 8) // (2 * len(gpus)))))
            log_path = logs / f"{name}.log"
            log = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            running[name] = (proc, gpu, log, time.time())
            print(f"[extract] {name} -> GPU {gpu} (log: {log_path})", flush=True)
        time.sleep(2)
        for name, (proc, gpu, log, start) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[name]
            free.append(gpu)
            status = "done" if code == 0 else f"FAILED (exit {code})"
            print(f"[extract] {name} {status} on GPU {gpu} in {time.time() - start:.0f}s", flush=True)
            if code != 0:
                failed.append(name)
    if failed:
        raise RuntimeError(f"以下 checkpoint 推理失败，详见 {logs}: {failed}")


def extract(
    cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    output: Path,
    *,
    smoke_test: bool = False,
    gpus: list[str] | None = None,
) -> Path:
    ckpts = checkpoint_meta(cfg, smoke_test=smoke_test)
    result = output / "embeddings.pt"
    signature = extraction_signature(cfg, rows, ckpts, smoke_test)
    if result.is_file():
        existing = torch.load(result, map_location="cpu", weights_only=False)
        if existing.get("signature") == signature:
            print(f"[extract] embeddings already match signature: {result}", flush=True)
            return result
        print("[extract] embeddings signature mismatch, rebuilding", flush=True)

    output.mkdir(parents=True, exist_ok=True)
    pending = [ck for ck in ckpts if not shard_is_valid(shard_path(output, ck["name"]), shard_signature(rows, ck, smoke_test))]
    print(f"[extract] {len(ckpts) - len(pending)} cached shards, {len(pending)} to run", flush=True)
    if pending and gpus and len(gpus) > 1:
        run_workers(cfg, [ck["name"] for ck in pending], gpus, output, smoke_test=smoke_test)
    else:
        for ck in pending:
            extract_checkpoint(cfg, rows, ck, output, smoke_test=smoke_test)

    runs = []
    for ck in ckpts:
        run = torch.load(shard_path(output, ck["name"]), map_location="cpu", weights_only=False)["run"]
        # Raw tokens stay in shards/; the merged file only keeps pooled features used by analyze().
        run.pop("tokens", None)
        runs.append(run)
    payload = {"config": cfg, "signature": signature, "runs": runs}
    torch.save(payload, result)
    print(f"[extract] merged {len(runs)} shards -> {result}", flush=True)
    return result


def summarize_checkpoint(ck: dict[str, Any]) -> dict[str, Any]:
    model_cfg = ck.get("model_config", {})
    train_cfg = ck.get("train_config", {})
    dataset = train_cfg.get("dataset", {}) if isinstance(train_cfg, dict) else {}
    policy = train_cfg.get("policy", {}) if isinstance(train_cfg, dict) else {}
    return {
        "lambda_bp_encoder": policy.get("lambda_bp_encoder", model_cfg.get("lambda_bp_encoder")),
        "frozen_bp_encoder": policy.get("frozen_bp_encoder", model_cfg.get("frozen_bp_encoder")),
        "frozen_vit": policy.get("frozen_vit", model_cfg.get("frozen_vit")),
        "bp_sampling_mode": dataset.get("bp_sampling_mode"),
        "bp_action_use": dataset.get("bp_action_use"),
        "bp_state_use": dataset.get("bp_state_use"),
        "bp_camera_keys": model_cfg.get("bp_camera_keys"),
        "bp_num_chunks": model_cfg.get("bp_num_chunks"),
        "bp_num_query_tokens": model_cfg.get("bp_num_query_tokens"),
    }


def cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    y = torch.nn.functional.normalize(y.float(), dim=-1)
    return 1 - x @ y.T


def effective_rank(x: torch.Tensor) -> float:
    centered = x.float() - x.float().mean(0)
    if centered.shape[0] < 2:
        return 1.0
    s = torch.linalg.svdvals(centered)
    energy = s * s
    probs = energy / energy.sum().clamp_min(1e-12)
    return float(torch.exp(-(probs * probs.clamp_min(1e-12).log()).sum()))


def _label_onehot(labels: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    tasks = sorted(set(labels))
    index = {task: i for i, task in enumerate(tasks)}
    y = torch.tensor([index[label] for label in labels], dtype=torch.long)
    return y, torch.nn.functional.one_hot(y, len(tasks)).double()


def leave_one_out_accuracy(x: torch.Tensor, labels: list[str]) -> float:
    """Assign each sample to the nearest task center computed without that sample."""
    x = x.double()
    y, onehot = _label_onehot(labels)
    sums = onehot.T @ x
    counts = onehot.sum(0)
    xn = torch.nn.functional.normalize(x, dim=-1)
    sim = xn @ torch.nn.functional.normalize(sums, dim=-1).T
    rows = torch.arange(len(labels))
    own_center = sums[y] - x
    sim[rows, y] = torch.nn.functional.cosine_similarity(x, own_center, dim=-1)
    sim[rows[counts[y] < 2], y[counts[y] < 2]] = -math.inf
    return float((sim.argmax(dim=1) == y).double().mean())


def silhouette_cosine(x: torch.Tensor, labels: list[str]) -> float | None:
    if len(set(labels)) < 2 or len(labels) < 3:
        return None
    d = cosine_distance(x, x).double()
    y, onehot = _label_onehot(labels)
    rows = torch.arange(len(labels))
    sums = d @ onehot
    counts = onehot.sum(0)
    own_n = counts[y] - 1
    a = (sums[rows, y] - d[rows, rows]) / own_n.clamp_min(1)
    mean_to_task = sums / counts
    mean_to_task[rows, y] = math.inf
    b = mean_to_task.min(dim=1).values
    scores = (b - a) / torch.maximum(torch.maximum(a, b), torch.full_like(a, 1e-12))
    valid = own_n > 0
    return float(scores[valid].mean()) if valid.any() else None


def valid_chunk_counts(run: dict[str, Any]) -> np.ndarray:
    """Episode-length bucket: K under the variable-K rule, so fixed-K checkpoints stay comparable."""
    natural = [m.get("bp_natural_chunks") for m in run["metadata"]]
    if all(k is not None for k in natural):
        return np.asarray(natural, dtype=np.int64)
    num_chunks = int(run["chunks"].shape[1])
    mask = run["token_mask"].bool()
    return mask.reshape(mask.shape[0], num_chunks, -1).any(-1).sum(-1).numpy()


def nearest_neighbor_agreement(x: torch.Tensor, keys: np.ndarray) -> float:
    d = cosine_distance(x, x)
    d.fill_diagonal_(math.inf)
    nn = d.argmin(dim=1).numpy()
    return float((keys[nn] == keys).mean())


def length_stratified_metrics(x: torch.Tensor, labels: list[str], lengths: np.ndarray) -> dict[str, Any]:
    """Task separability measured only among samples with the same valid-chunk count."""
    labels_np = np.array(labels)
    correct = 0.0
    total = 0
    silhouette_sum = 0.0
    silhouette_n = 0
    chance_sum = 0.0
    for length in sorted(set(lengths.tolist())):
        group = np.where(lengths == length)[0]
        group_labels = labels_np[group]
        tasks = [task for task in set(group_labels.tolist()) if (group_labels == task).sum() >= 2]
        if len(tasks) < 2:
            continue
        group = group[np.isin(group_labels, tasks)]
        sub_labels = labels_np[group].tolist()
        sub_x = x[torch.as_tensor(group)]
        correct += leave_one_out_accuracy(sub_x, sub_labels) * len(group)
        total += len(group)
        chance_sum += len(group) / len(tasks)
        sil = silhouette_cosine(sub_x, sub_labels)
        if sil is not None:
            silhouette_sum += sil * len(group)
            silhouette_n += len(group)
    return {
        "length_stratified_samples": total,
        "length_stratified_center_accuracy": correct / total if total else None,
        "length_stratified_silhouette_cosine": silhouette_sum / silhouette_n if silhouette_n else None,
        "length_stratified_chance_accuracy": chance_sum / total if total else None,
    }


def run_metrics(run: dict[str, Any], feature_key: str) -> dict[str, Any]:
    x = run[feature_key].float()
    labels = [m["task"] for m in run["metadata"]]
    lengths = valid_chunk_counts(run)
    d = cosine_distance(x, x)
    intra: list[float] = []
    inter: list[float] = []
    for i, label in enumerate(labels):
        same = [j for j, other in enumerate(labels) if other == label and j != i]
        other = [j for j, other_label in enumerate(labels) if other_label != label]
        if same:
            intra.extend(d[i, same].tolist())
        if other:
            inter.extend(d[i, other].tolist())
    intra_mean = float(np.mean(intra)) if intra else float("nan")
    inter_mean = float(np.mean(inter)) if inter else float("nan")
    norm = x.norm(dim=-1)
    return {
        "checkpoint": run["name"],
        "feature": feature_key,
        "samples": len(labels),
        "tasks": len(set(labels)),
        "intra_task_cosine_distance": intra_mean,
        "inter_task_cosine_distance": inter_mean,
        "inter_intra_ratio": float(inter_mean / max(intra_mean, 1e-8)) if np.isfinite(intra_mean) else float("nan"),
        "leave_one_out_nearest_center_accuracy": leave_one_out_accuracy(x, labels),
        "silhouette_cosine": silhouette_cosine(x, labels),
        "mean_norm": float(norm.mean()),
        "std_norm": float(norm.std()),
        "pca_effective_rank": effective_rank(x),
        "nearest_neighbor_same_task": nearest_neighbor_agreement(x, np.array(labels)),
        "nearest_neighbor_same_length": nearest_neighbor_agreement(x, lengths),
        "length_chance_same_length": float(sum((lengths == k).mean() ** 2 for k in set(lengths.tolist()))),
        **length_stratified_metrics(x, labels, lengths),
        "checkpoint_summary": run.get("checkpoint_summary", {}),
    }


FEATURES = ("chunk_flat", "global_mean")


def feature_matrix(run: dict[str, Any], feature: str) -> torch.Tensor:
    """chunk_flat keeps keyframe order (per-chunk mean, concatenated); global_mean averages all valid tokens."""
    if feature == "chunk_flat":
        return run["chunks"].reshape(run["chunks"].shape[0], -1).float()
    if feature == "global_mean":
        return run["global_mean"].float()
    raise ValueError(f"unknown feature {feature!r}, expected one of {FEATURES}")


def analysis_features(cfg: dict[str, Any]) -> list[str]:
    features = list(cfg.get("features", ["chunk_flat"]))
    for feature in features:
        if feature not in FEATURES:
            raise ValueError(f"unknown feature {feature!r}, expected one of {FEATURES}")
    return features


def analyze(cfg: dict[str, Any], output: Path) -> None:
    data = torch.load(output / "embeddings.pt", map_location="cpu", weights_only=False)
    features = analysis_features(cfg)
    metrics: list[dict[str, Any]] = []
    for run in data["runs"]:
        for feature in features:
            metrics.append(run_metrics({**run, feature: feature_matrix(run, feature)}, feature))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        plot(data, metrics, output, cfg)
    except ImportError as exc:
        print(f"[analyze] 可视化依赖缺失，指标仍已保存: {exc}", flush=True)


def pca2(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(x, full_matrices=False)
    return x @ vh[:2].T


def plot(data: dict[str, Any], metrics: list[dict[str, Any]], output: Path, cfg: dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    feature = analysis_features(cfg)[0]
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    for run in data["runs"]:
        labels = np.array([m["task"] for m in run["metadata"]])
        x = feature_matrix(run, feature)
        local = pca2(x.numpy())
        fig, ax = plt.subplots(figsize=(10, 7))
        for task in sorted(set(labels)):
            mask = labels == task
            ax.scatter(local[mask, 0], local[mask, 1], s=14, alpha=0.75, label=task)
        ax.set_title(f"PCA of {feature} BP embeddings: {run['name']}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(plots / f"pca_{run['name']}.png", dpi=150)
        plt.close(fig)

        center_labels = sorted(set(labels))
        centers = torch.stack([x[labels == task].mean(0) for task in center_labels])
        dist = cosine_distance(centers, centers).numpy()
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(dist, cmap="viridis")
        ax.set_title(f"Task center cosine distance ({feature}): {run['name']}")
        ax.set_xticks(range(len(center_labels)), center_labels, rotation=90, fontsize=5)
        ax.set_yticks(range(len(center_labels)), center_labels, fontsize=5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(plots / f"center_distance_{run['name']}.png", dpi=150)
        plt.close(fig)

    rows = [m for m in metrics if m["feature"] == feature]
    names = [m["checkpoint"] for m in rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(names, [m["intra_task_cosine_distance"] for m in rows], marker="o", label="intra task")
    ax.plot(names, [m["inter_task_cosine_distance"] for m in rows], marker="o", label="inter task")
    ax.set_title(f"BPV3 {feature} cosine distance by checkpoint")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Cosine distance")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots / "task_distance.png", dpi=150)
    plt.close(fig)

    plot_metrics_summary(rows, feature, plots / "metrics_summary.png")


def plot_metrics_summary(rows: list[dict[str, Any]], feature: str, path: Path) -> None:
    """One figure with every table metric across checkpoints for a single feature."""
    import matplotlib.pyplot as plt

    names = [m["checkpoint"] for m in rows]
    first = rows[0]
    panels = [
        ("Intra / inter task cosine distance", [("intra_task_cosine_distance", "intra"), ("inter_task_cosine_distance", "inter")], []),
        ("Inter / intra ratio", [("inter_intra_ratio", "ratio")], []),
        ("Center accuracy (leave-one-out)", [("leave_one_out_nearest_center_accuracy", "all samples"), ("length_stratified_center_accuracy", "same length")],
         [(1 / max(1, first["tasks"]), "chance (all)", ":"), (first["length_stratified_chance_accuracy"], "chance (same length)", "-.")]),
        ("Silhouette (cosine)", [("silhouette_cosine", "all samples"), ("length_stratified_silhouette_cosine", "same length")], [(0.0, "0", ":")]),
        ("PCA effective rank", [("pca_effective_rank", "erank")], []),
        ("Nearest neighbor agreement", [("nearest_neighbor_same_task", "same task"), ("nearest_neighbor_same_length", "same length")],
         [(first["length_chance_same_length"], "chance (same length)", ":")]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for ax, (title, keys, hlines) in zip(axes.flat, panels):
        for key, label in keys:
            ax.plot(names, [r[key] for r in rows], marker="o", markersize=3, label=label)
        for value, label, style in hlines:
            if value is not None:
                ax.axhline(value, color="gray", linestyle=style, linewidth=1, label=label)
        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=60, labelsize=7)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"BPV3 BP encoder probe metrics by checkpoint ({feature})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_plan(cfg: dict[str, Any], out: Path, *, smoke_test: bool = False) -> None:
    ckpts = checkpoint_meta(cfg, smoke_test=smoke_test)
    plan = {
        "output_dir": str(out),
        "data_repos": dataset_plan(cfg, smoke_test=smoke_test),
        "checkpoints": [
            {"name": ck["name"], "path": ck["path"], "summary": summarize_checkpoint(ck)} for ck in ckpts
        ],
        "manifest_fingerprint": manifest_fingerprint(cfg, smoke_test=smoke_test),
    }
    print(json.dumps(plan, indent=2, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true", help="只验证配置、数据路径和 checkpoint 文件，不导入训练依赖")
    parser.add_argument("--manifest-only", action="store_true", help="生成固定 clean episode manifest 后退出")
    parser.add_argument("--analyze-only", action="store_true", help="已有 embeddings.pt 时只重算指标和图片")
    parser.add_argument("--smoke-test", action="store_true", help="使用独立小输出目录和少量任务/episode/checkpoint 做快速验证")
    parser.add_argument("--gpus", default=os.environ.get("BPV3_PROBE_GPUS", ""), help="多卡并行，如 0,1,2,3；每张卡一次跑一个 checkpoint")
    parser.add_argument("--worker-checkpoint", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]

    cfg = load_cfg(args.config.resolve())
    out = output_dir(cfg, smoke_test=args.smoke_test)
    os.environ.setdefault("PYTHONPYCACHEPREFIX", "/tmp/bpv3_probe_pycache")

    if args.dry_run:
        write_plan(cfg, out, smoke_test=args.smoke_test)
        return 0

    if args.worker_checkpoint:
        rows = make_manifest(cfg, out, smoke_test=args.smoke_test)
        ck = next(ck for ck in checkpoint_meta(cfg, smoke_test=args.smoke_test) if ck["name"] == args.worker_checkpoint)
        extract_checkpoint(cfg, rows, ck, out, smoke_test=args.smoke_test)
        return 0

    rows = make_manifest(cfg, out, smoke_test=args.smoke_test)
    print(f"manifest: {len(rows)} samples, {len(set(row['task'] for row in rows))} tasks -> {out / 'manifest.jsonl'}", flush=True)
    if args.manifest_only:
        return 0
    if args.analyze_only:
        analyze(cfg, out)
        return 0
    extract(cfg, rows, out, smoke_test=args.smoke_test, gpus=gpus)
    analyze(cfg, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
