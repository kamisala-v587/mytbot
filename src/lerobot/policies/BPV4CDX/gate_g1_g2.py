"""G1 (no action / state / text leakage into Z_demo) and G2 (shapes, cache vs online ViT, batch invariance).

G1
  - Z_demo computed from a batch equals Z_demo computed from the demo-side keys only, and is unchanged when the
    actions / states / texts are replaced by random tensors and by zero tensors;
  - after a backward pass the Q-Former has gradients, Qwen has none, and Z_demo receives a gradient;
  - texts fed to the model (not dropped) are only the canonical texts (task names) or the generic sentence.
G2
  - demo features [B, 32, 3, P, 1024], Z_demo [B, num_demo_tokens, hidden]; prefix length
    [BOS] + text + demo_start + Z_demo + demo_end + segment_query (180 for the main branch);
  - Z_demo of one trajectory is the same alone and inside a batch (no padding / batch leakage);
  - an invalid camera is masked (its features do not change Z_demo); P5 pooling order;
  - cached ViT features match a fresh ViT pass: mean token cosine >= 0.99 and frame offset 0 for a few episodes.

Usage: python -m lerobot.policies.BPV4CDX.gate_g1_g2 --config <jsonc> [--ckpt <pretrained_model>] --out <dir> \
         [--online_episodes 2] [--dataset.cache_dir=... other draccus overrides]
"""

from __future__ import annotations

import argparse

import torch

from .common import collate, load_cfg, make_dataset, make_policy, reproducibility, to_device, write_result
from .paths import GENERIC_TEXT, is_robotwin_like

DEMO_KEYS = ("demo.feats", "demo.time", "demo.camera_valid")


def g1(policy, ds, batch) -> dict:
    model = policy.model
    res = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z_full = model.encode_demo(batch)
        z_demo_only = model.encode_demo({k: batch[k] for k in DEMO_KEYS})
        same = []
        for fill in (torch.randn_like, torch.zeros_like):  # random labels, zero labels
            tampered = dict(batch)
            tampered["action"] = fill(batch["action"])
            tampered["observation.state"] = fill(batch["observation.state"])
            tampered["text.input_ids"] = torch.randint_like(batch["text.input_ids"], 0, 1000)
            same.append(bool(torch.equal(z_full, model.encode_demo(tampered))))
    res["z_equal_demo_keys_only"] = bool(torch.equal(z_full, z_demo_only))
    res["z_unchanged_by_labels_state_text"] = bool(all(same))
    res["state_slot"] = "segment-start state (AE-S0)" if model.config.use_segment_start_state else "zeros"

    policy.train()
    model.zero_grad(set_to_none=True)
    z_grad = {}

    def record(g: torch.Tensor) -> None:
        z_grad["norm"] = float(g.float().norm())

    with torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode_demo(batch)
        z.register_hook(record)
        losses = model.forward_segments(batch, z_demo=z)
        loss = losses["flow"].mean()
    loss.backward()
    policy.eval()
    enc_grad = sum(float(p.grad.float().norm()) for p in model.demo_encoder.parameters() if p.grad is not None)
    vlm_grads = [n for n, p in model.core.backbone.vlm.named_parameters() if p.grad is not None]
    res["z_demo_grad_norm"] = z_grad.get("norm", 0.0)
    res["qformer_grad_norm"] = enc_grad
    res["qwen_params_with_grad"] = len(vlm_grads)
    res["gradient_path_ok"] = bool(res["z_demo_grad_norm"] > 0 and enc_grad > 0 and not vlm_grads)

    allowed = {GENERIC_TEXT} | {r.text for r in ds.repos}
    tok = ds._tokenizer
    dropped = batch["text.is_dropped"].reshape(-1).cpu()
    texts = {tok.decode(ids[m.bool()], skip_special_tokens=True).strip()
             for ids, m, d in zip(batch["text.input_ids"].reshape(-1, batch["text.input_ids"].shape[-1]).cpu(),
                                  batch["text.attention_mask"].reshape(-1, batch["text.attention_mask"].shape[-1]).cpu(),
                                  dropped) if not d}
    res["texts_seen"] = sorted(texts)
    res["dropped_segments"] = int(dropped.sum())
    res["texts_only_canonical_or_generic"] = bool(texts <= allowed)
    # no arm words in the canonical texts outside RoboTwin ("left" / "right" alone may name the object, e.g.
    # "rotate the left hearth", and RoboTwin names such as place_a2b_left name the goal)
    arm_words = {"arm", "arms"}
    leaking = sorted({r.text for r in ds.repos if not is_robotwin_like(r.repo) and arm_words & set(r.text.split())})
    res["canonical_texts_with_arm_words"] = leaking[:50]
    res["canonical_texts_no_arm_words"] = not leaking
    res["pass"] = all(res[k] for k in ("z_equal_demo_keys_only", "z_unchanged_by_labels_state_text", "gradient_path_ok",
                                       "texts_only_canonical_or_generic", "canonical_texts_no_arm_words"))
    return res


def online_vit_check(ds, n_episodes: int, qwen_path: str) -> dict:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.BPV4BPEn.build_feature_cache import load_vit, visual_tokens
    from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision
    from lerobot.transforms.utils import resize_with_pad

    from .ae3_data import source_of
    from .dataset import pool_spatial

    device = torch.device("cuda")
    vlm = load_vit(None, qwen_path, device)
    processor, _ = load_qwen35_vision(qwen_path)

    def encode(lds, start: int, frame: int, cam: str) -> torch.Tensor:
        proc = processor([resize_with_pad(lds[start + frame][cam][None], 224, 224)[0]], do_rescale=False)
        with torch.no_grad():
            return visual_tokens(vlm, torch.as_tensor(proc.pixel_values).to(device, torch.bfloat16),
                                 torch.as_tensor(proc.image_grid_thw, dtype=torch.long).to(device)).float().cpu()[0]

    # decode noise makes a few tokens differ; the criteria are the mean token cosine and frame alignment: among the
    # checked frames that differ from their neighbours (online cosine to frame f -1 / f +1 below 0.995), the two
    # least similar to the cache must match best at offset 0. Static frames (e.g. the motionless first frames) are
    # indistinguishable from their neighbours, so an offset there is decode noise, not misalignment; they are skipped.
    cos_fn = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())  # noqa: E731
    # one episode per source first (AE-3: check every data source), then more of the same
    by_source: dict[str, list[int]] = {}
    for i, (ri, _) in enumerate(ds.items):
        by_source.setdefault(source_of(ds.repos[ri].rel), []).append(i)
    order = [v[j] for j in range(max(map(len, by_source.values()))) for v in by_source.values() if j < len(v)]
    per_episode = []
    for i in order[: min(n_episodes, len(ds))]:
        ri, ep = ds.items[i]
        repo = ds.repos[ri].repo
        rec = torch.load(ds.cache_file(ri, ep), weights_only=False)
        stored = int(rec.get("spatial", rec["feats"].shape[2]))
        cams = rec.get("cameras") or ds.repos[ri].emb.cameras
        lds = LeRobotDataset(repo, video_backend="torchcodec")
        start, length = int(lds.meta.episodes[ep]["dataset_from_index"]), int(rec["length"])
        checked = []
        for k in range(0, len(rec["frames"]), 4):
            f = int(rec["frames"][k])
            for ci, cam in enumerate(cams):
                if cam is None:
                    continue  # missing camera: zeros in the cache, nothing to compare
                c = rec["feats"][k, ci].float()

                def online_tokens(frame):
                    t = encode(lds, start, frame, cam)
                    return pool_spatial(t[None, None], 5)[0, 0] if stored == 5 else t

                online = {d: online_tokens(f + d) for d in (-1, 0, 1) if 0 <= f + d < length}
                score = {d: cos_fn(o, c) for d, o in online.items()}
                distinct = all(cos_fn(online[0], o) < 0.995 for d, o in online.items() if d != 0)
                checked.append({"slot": k, "camera": ci, "cos": score[0], "distinct": distinct,
                                "best_offset": max(score, key=score.get)})
        movers = sorted((x for x in checked if x["distinct"]), key=lambda x: x["cos"])[:2]
        per_episode.append({"rel": ds.repos[ri].rel, "episode": int(ep), "spatial": stored,
                            "mean_token_cos": float(sum(x["cos"] for x in checked) / len(checked)),
                            "aligned": bool(movers) and all(x["best_offset"] == 0 for x in movers),
                            "aligned_all_distinct": float(sum(x["best_offset"] == 0 for x in checked if x["distinct"])
                                                          / max(1, sum(x["distinct"] for x in checked))),
                            "static_skipped": int(sum(not x["distinct"] for x in checked)), "checked": len(checked)})
    ok = bool(per_episode) and all(e["mean_token_cos"] >= 0.99 and e["aligned"] for e in per_episode)
    return {"episodes": per_episode, "pass": ok}


def g2(policy, ds, batch, online_episodes: int, qwen_path: str) -> dict:
    model = policy.model
    res = {"demo_feats_shape": list(batch["demo.feats"].shape)}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        z = model.encode_demo(batch)
        z0 = model.encode_demo({k: batch[k][:1] for k in DEMO_KEYS})
    cfg = policy.config
    hidden = int(model.core.backbone.vlm.config.text_config.hidden_size)
    res["z_demo_shape"] = list(z.shape)
    res["shape_ok"] = bool(list(batch["demo.feats"].shape[1:4]) == [cfg.num_slots, cfg.num_cameras, cfg.spatial_tokens]
                           and list(z.shape[1:]) == [cfg.num_demo_tokens, hidden])
    res["batch_invariance_max_abs"] = float((z[:1].float() - z0.float()).abs().max())
    res["batch_invariance_ok"] = bool(res["batch_invariance_max_abs"] < 5e-2)

    # missing view: an invalid camera is masked out (its features do not matter), not copied from another camera
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        demo = {k: batch[k] for k in DEMO_KEYS}
        miss = {**demo, "demo.camera_valid": demo["demo.camera_valid"].clone()}
        miss["demo.camera_valid"][:, :, -1] = False
        noisy = {**miss, "demo.feats": miss["demo.feats"].clone()}
        noisy["demo.feats"][:, :, -1] = torch.randn_like(noisy["demo.feats"][:, :, -1]) * 10
        z_miss, z_noisy = model.encode_demo(miss), model.encode_demo(noisy)
    res["missing_view_masked"] = bool((z_miss.float() - z_noisy.float()).abs().max() < 5e-2
                                      and (z_miss.float() - z.float()).abs().max() > 1e-3)
    res["all_cameras_valid_in_batch"] = bool(batch["demo.camera_valid"].all())

    # P5 order: 4 x 4 block means (top-left, top-right, bottom-left, bottom-right) + global mean
    from .dataset import pool_spatial

    grid = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64, 1)
    pooled = pool_spatial(grid, 5).flatten().tolist()
    res["p5_order"] = pooled
    res["p5_order_ok"] = pooled == [13.5, 17.5, 45.5, 49.5, 31.5]

    # prefix layout: [BOS] + text + demo_start + Z_demo + demo_end + segment_query
    prefix, _ = model.build_prefix(batch, z)
    expect = int(cfg.prefix_bos) + int(ds.cfg.language_max_length) + (cfg.num_demo_tokens + 2 if z is not None else 0) + 1
    res["prefix_len"] = int(prefix.shape[1])
    res["prefix_len_ok"] = bool(prefix.shape[1] == expect)
    lengths = batch["demo.length"].tolist()
    res["trajectory_lengths_in_batch"] = lengths
    res["fixed_token_count_across_lengths"] = bool(len(set(lengths)) >= 1 and z.shape[1] == cfg.num_demo_tokens)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        flow = model.forward_segments(batch)["flow"]
    res["forward_finite"] = bool(torch.isfinite(flow).all())
    res["online_vit"] = online_vit_check(ds, online_episodes, cfg.qwen3_5_pretrained_path or qwen_path) if online_episodes else None
    res["pass"] = bool(res["shape_ok"] and res["batch_invariance_ok"] and res["forward_finite"]
                       and res["missing_view_masked"] and res["p5_order_ok"] and res["prefix_len_ok"]
                       and (res["online_vit"] is None or res["online_vit"]["pass"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--online_episodes", type=int, default=2)
    args, overrides = ap.parse_known_args()  # unknown args are draccus overrides, e.g. --dataset.cache_dir=...
    cfg = load_cfg(args.ckpt or args.config, overrides)
    ds = make_dataset(cfg, split="train")
    torch.manual_seed(0)
    lengths = [len(ds.repos[ri].states[ep]) for ri, ep in ds.items]
    order = sorted(range(len(ds)), key=lambda i: lengths[i])
    pick = [order[0], order[-1]] + order[1 : args.batch - 1]  # shortest and longest trajectories in one batch
    batch = to_device(collate([ds[i] for i in pick[: args.batch]]))
    policy = make_policy(cfg, args.ckpt)
    result = {"G1": g1(policy, ds, batch), "G2": g2(policy, ds, batch, args.online_episodes, cfg.dataset.qwen3_5_processor_path)}
    result["pass"] = bool(result["G1"]["pass"] and result["G2"]["pass"])
    result["reproducibility"] = reproducibility(cfg, args.ckpt, policy, ds)
    path = write_result(args.out, "g1_g2", result)
    print(path)
    print({k: result[k]["pass"] if k in ("G1", "G2") else result[k] for k in ("G1", "G2", "pass")})


if __name__ == "__main__":
    main()
