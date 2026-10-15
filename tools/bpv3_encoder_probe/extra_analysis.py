#!/usr/bin/env python3
"""Length confound, position-embedding removal and per-task analysis on top of probe.py outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from probe import (
    cosine_distance,
    effective_rank,
    leave_one_out_accuracy,
    length_stratified_metrics,
    load_cfg,
    nearest_neighbor_agreement,
    output_dir,
    silhouette_cosine,
    valid_chunk_counts,
)

POS_KEY = "model.bp_obs_encoder.chunk_position_embedding.weight"


def position_embedding(ckpt: str) -> torch.Tensor:
    with safe_open(str(Path(ckpt) / "model.safetensors"), "pt") as f:
        return f.get_tensor(POS_KEY).float()


def pc1_length_corr(x: torch.Tensor, lengths: np.ndarray) -> float:
    xc = x.double() - x.double().mean(0)
    _, _, vh = torch.linalg.svd(xc, full_matrices=False)
    pc1 = (xc @ vh[0]).numpy()
    return float(abs(np.corrcoef(pc1, lengths)[0, 1]))


def cross_length_ratio(x: torch.Tensor, labels: np.ndarray, lengths: np.ndarray) -> float | None:
    d = cosine_distance(x, x).numpy()
    same_task = labels[:, None] == labels[None, :]
    same_len = lengths[:, None] == lengths[None, :]
    off_diag = ~np.eye(len(labels), dtype=bool)
    diff = d[same_task & ~same_len & off_diag]
    same = d[same_task & same_len & off_diag]
    if diff.size == 0 or same.size == 0:
        return None
    return float(diff.mean() / max(same.mean(), 1e-12))


def summary(x: torch.Tensor, labels: np.ndarray, lengths: np.ndarray) -> dict:
    strat = length_stratified_metrics(x, labels.tolist(), lengths)
    return {
        "nn_same_task": nearest_neighbor_agreement(x, labels),
        "nn_same_length": nearest_neighbor_agreement(x, lengths),
        "pc1_length_corr": pc1_length_corr(x, lengths),
        "cross_length_ratio": cross_length_ratio(x, labels, lengths),
        "center_acc": leave_one_out_accuracy(x, labels.tolist()),
        "silhouette": silhouette_cosine(x, labels.tolist()),
        "strat_center_acc": strat["length_stratified_center_accuracy"],
        "strat_silhouette": strat["length_stratified_silhouette_cosine"],
        "strat_chance": strat["length_stratified_chance_accuracy"],
        "strat_samples": strat["length_stratified_samples"],
    }


def loo_predict(x: torch.Tensor, labels: np.ndarray) -> np.ndarray:
    """Leave-one-out nearest task center (cosine) prediction for every sample."""
    tasks = sorted(set(labels.tolist()))
    y = np.array([tasks.index(t) for t in labels])
    xd = x.double()
    sums = torch.stack([xd[torch.as_tensor(y == i)].sum(0) for i in range(len(tasks))])
    counts = np.bincount(y, minlength=len(tasks))
    pred = []
    for i in range(len(labels)):
        centers = sums.clone()
        centers[y[i]] = centers[y[i]] - xd[i]
        sim = torch.nn.functional.cosine_similarity(xd[i][None], centers, dim=-1)
        if counts[y[i]] < 2:
            sim[y[i]] = -1e9
        pred.append(tasks[int(sim.argmax())])
    return np.array(pred)


def per_task(x: torch.Tensor, labels: np.ndarray, lengths: np.ndarray, top_pairs: int = 8) -> dict:
    """Per-task accuracy and confusions, predicted only among samples with the same valid-chunk count."""
    pred = np.full(len(labels), "", dtype=object)
    used = np.zeros(len(labels), dtype=bool)
    within_erank = []
    for length in sorted(set(lengths.tolist())):
        group = np.where(lengths == length)[0]
        tasks = [t for t in set(labels[group].tolist()) if (labels[group] == t).sum() >= 2]
        group = group[np.isin(labels[group], tasks)]
        if len(tasks) < 2:
            continue
        pred[group] = loo_predict(x[torch.as_tensor(group)], labels[group])
        used[group] = True
        within_erank.append((effective_rank(x[torch.as_tensor(group)]), len(group)))
    acc = {t: float((pred[used & (labels == t)] == t).mean()) for t in sorted(set(labels[used].tolist()))}
    confusions: dict[tuple[str, str], int] = {}
    for true, guess in zip(labels[used], pred[used]):
        if true != guess:
            key = tuple(sorted((true, guess)))
            confusions[key] = confusions.get(key, 0) + 1
    pairs = sorted(confusions.items(), key=lambda kv: -kv[1])[:top_pairs]
    erank = sum(e * n for e, n in within_erank) / max(1, sum(n for _, n in within_erank))
    return {
        "strat_task_acc": acc,
        "strat_confusions": [(a, b, n) for (a, b), n in pairs],
        "within_length_erank": erank,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_cfg(args.config.resolve())
    out = output_dir(cfg)
    data = torch.load(out / "embeddings.pt", map_location="cpu", weights_only=False)
    result = {}
    for run in data["runs"]:
        labels = np.array([m["task"] for m in run["metadata"]])
        lengths = valid_chunk_counts(run)
        chunks = run["chunks"].float()
        valid = torch.as_tensor(np.arange(chunks.shape[1])[None, :] < lengths[:, None])
        pos = position_embedding(run["path"])[: chunks.shape[1]]
        no_pos = (chunks - pos[None]) * valid[..., None]
        raw_x = chunks.reshape(len(labels), -1)
        nopos_x = no_pos.reshape(len(labels), -1)
        result[run["name"]] = {
            "length_distribution": {int(k): int((lengths == k).sum()) for k in sorted(set(lengths.tolist()))},
            "raw": summary(raw_x, labels, lengths),
            "no_pos": summary(nopos_x, labels, lengths),
            **per_task(raw_x, labels, lengths),
        }
        r = result[run["name"]]
        print(
            f"{run['name']:>14} raw: nnT={r['raw']['nn_same_task']:.2f} nnL={r['raw']['nn_same_length']:.3f} "
            f"pc1={r['raw']['pc1_length_corr']:.2f} ratio={r['raw']['cross_length_ratio']} "
            f"strat={r['raw']['strat_center_acc']:.2f}/{r['raw']['strat_silhouette']:+.2f} | "
            f"nopos: nnL={r['no_pos']['nn_same_length']:.3f} pc1={r['no_pos']['pc1_length_corr']:.2f} "
            f"acc={r['no_pos']['center_acc']:.2f} sil={r['no_pos']['silhouette']:+.2f} strat={r['no_pos']['strat_center_acc']:.2f}",
            flush=True,
        )
    (out / "extra_metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
