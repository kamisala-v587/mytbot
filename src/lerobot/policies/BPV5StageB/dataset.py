"""Stage B dataset: query j, demo i≠j under M1 L0/L1 pairing, current frames decoded online.

Reuses the Stage A scan index, P5 demo cache, and 80-D action pipeline.
History frames are not loaded.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

from lerobot.policies.BPV5QFormer import data80d as D
from lerobot.policies.BPV5QFormer.dataset import BPV5QFormerDataset
from lerobot.policies.BPV5StageB.configuration_bpv5stageb import BPV5StageBDatasetConfig
from lerobot.policies.BPV5StageB.pairing import build_partner_map, choose_partner

DEMO_GENERIC = "A robot demonstration."
CURRENT_GENERIC = "Perform the task."


class BPV5StageBDataset(BPV5QFormerDataset):
    cfg: BPV5StageBDatasetConfig

    def _apply_split(self, by_repo: dict) -> None:
        min_n = int(self.cfg.min_repo_episodes)
        by_repo = {rel: rows for rel, rows in by_repo.items() if len(rows) >= min_n}
        self._pool_eps = {rel: [int(ep) for _, ep, _ in rows] for rel, rows in by_repo.items()}
        super()._apply_split(by_repo)

        partners, stats = build_partner_map(
            self.items,
            self._pool_eps,
            min_bucket=int(self.cfg.pairing_min_bucket),
            tier_overrides=self.cfg.pairing_tiers,
            storage=getattr(self.cfg, "pairing_storage", "list"),
            timing=getattr(self.cfg, "init_timing", False),
        )
        before = len(self.items)
        self.items = [(rel, ep, length) for rel, ep, length in self.items if (rel, int(ep)) in partners]
        self.partners = partners
        self.num_episodes = len(self.items)
        self.num_frames = sum(length for _, _, length in self.items)
        self._split_eps: dict[str, list[int]] = {}
        for rel, ep, _ in self.items:
            self._split_eps.setdefault(rel, []).append(int(ep))
        if not self.items:
            raise RuntimeError(
                "bpv5 stage B empty after M1 pairing filter "
                f"(before={before}, kept={stats['kept']}, dropped={stats['dropped']})"
            )
        logging.info(
            "bpv5 stage B split=%s M1: %d→%d items, %d leaves with >= %d episodes "
            "(kept=%d dropped=%d)",
            self.cfg.split,
            before,
            len(self.items),
            len(self._pool_eps),
            min_n,
            stats["kept"],
            stats["dropped"],
        )

    def _mask_text(self, text: str, prob: float, generic_ratio: float, generic: str) -> tuple[str, bool]:
        if float(prob) <= 0 or float(torch.rand(())) >= float(prob):
            return text, False
        if float(torch.rand(())) < float(generic_ratio):
            return generic, False
        return generic, True

    def _vision(self):
        if getattr(self, "_processor", None) is None:
            from lerobot.policies.BPVAv31.transform_bpvav31 import load_qwen35_vision

            self._processor, self._token_ids = load_qwen35_vision(self.cfg.qwen3_5_processor_path)
        return self._processor, self._token_ids

    def current_images(self, rel: str, ep: int, frames: list[int]) -> dict[str, torch.Tensor]:
        from lerobot.datasets.video_utils import decode_video_frames
        from lerobot.transforms.utils import resize_with_pad

        meta = D.episodes(rel)[int(ep)]
        fps = float(D.info(rel).get("fps") or 30)
        slots = D.camera_slots(rel)
        size = int(self.cfg.image_size)
        images: list[torch.Tensor] = []
        valid: list[bool] = []
        n_ok = 0
        for frame in frames:
            for cam in slots:
                if cam is None or cam not in meta["videos"]:
                    images.append(torch.zeros(3, size, size))
                    valid.append(False)
                    continue
                chunk, file, start = meta["videos"][cam]
                path = D.video_path(rel, cam, chunk, file)
                ts = [start + int(frame) / fps]
                decoded = None
                last_exc: Exception | None = None
                for backend in (None, "pyav"):
                    try:
                        decoded = decode_video_frames(path, ts, 1e-4, backend=backend)
                        break
                    except Exception as exc:
                        last_exc = exc
                        logging.warning(
                            "bpv5 stage B decode failed backend=%s rel=%s ep=%s cam=%s frame=%s path=%s err=%s",
                            backend or "default",
                            rel,
                            ep,
                            cam,
                            frame,
                            path,
                            exc,
                        )
                if decoded is not None:
                    images.append(resize_with_pad(decoded, size, size)[0])
                    valid.append(True)
                    n_ok += 1
                else:
                    # RoboTwin / cleaned mp4s occasionally trip torchcodec; don't kill the run.
                    _ = last_exc
                    images.append(torch.zeros(3, size, size))
                    valid.append(False)
        if n_ok == 0:
            raise RuntimeError(f"bpv5 stage B: all current cameras failed for {rel} ep={ep} frames={frames}")
        processor, _token_ids = self._vision()
        out = processor(images, do_rescale=False)
        grid = torch.as_tensor(out["image_grid_thw"], dtype=torch.long)
        pixels = torch.as_tensor(out["pixel_values"])
        s = len(frames)
        return {
            "obs.pixel_values": pixels,
            "obs.image_grid_thw": grid,
            "obs.camera_valid": torch.tensor(valid, dtype=torch.bool).view(s, 3),
        }

    def assemble(
        self,
        query_i: int,
        demo_ep: int,
        starts: list[int],
        *,
        demo_rel: str | None = None,
        mask_text: bool = True,
        pad_language: bool = False,
    ) -> dict:
        rel, ep, _ = self.items[query_i]
        ep_meta = D.episodes(rel)[int(ep)]
        state, action = self._frames.episode(rel, ep_meta)
        length = int(state.shape[0])
        target = self.target(
            query_i,
            starts,
            state=state,
            action=action,
            length=length,
            generic=[False] * len(starts),
            dropped=[False] * len(starts),
        )
        demo_rel = demo_rel or rel
        demo = self._demo_episode(demo_rel, int(demo_ep))
        demo_text = D.task_text(demo_rel, int(demo_ep))
        current_text = D.task_text(rel, int(ep))
        if pad_language:
            # Force-drop demo + current instruction text (attn all-zero). For HOM / ablations.
            demo_pad = current_pad = True
            demo_text, current_text = DEMO_GENERIC, CURRENT_GENERIC
        elif mask_text:
            demo_text, demo_pad = self._mask_text(
                demo_text, self.cfg.demo_text_mask_prob, self.cfg.demo_text_generic_ratio, DEMO_GENERIC
            )
            current_text, current_pad = self._mask_text(
                current_text, self.cfg.current_text_mask_prob, self.cfg.current_text_generic_ratio, CURRENT_GENERIC
            )
        else:
            demo_pad = current_pad = False
        demo_ids, demo_attn = self.tokens(demo_text)
        cur_ids, cur_attn = self.tokens(current_text)
        if demo_pad:
            demo_attn = torch.zeros_like(demo_attn)
        if current_pad:
            cur_attn = torch.zeros_like(cur_attn)
        s = len(starts)
        out = {
            **demo,
            **target,
            "demo_text.input_ids": demo_ids[None].expand(s, -1).clone(),
            "demo_text.attention_mask": demo_attn[None].expand(s, -1).clone(),
            "current_text.input_ids": cur_ids[None].expand(s, -1).clone(),
            "current_text.attention_mask": cur_attn[None].expand(s, -1).clone(),
            "demo.episode": torch.tensor(int(demo_ep), dtype=torch.long),
            "query.episode": torch.tensor(int(ep), dtype=torch.long),
            "query.repo": rel,
            "query.robot": str(D.info(rel).get("robot_type") or rel.split("/")[0]),
        }
        out.update(self.current_images(rel, int(ep), [int(f) for f in starts]))
        return out

    def _demo_episode(self, rel: str, ep: int) -> dict[str, torch.Tensor]:
        rec = torch.load(D.cache_path(self.cfg.cache_dir, rel, ep), map_location="cpu", weights_only=True)
        feats = rec["feats"]
        valid = rec.get("camera_valid")
        return {
            "demo.feats": feats,
            "demo.time": rec["time_ratio"].float(),
            "demo.camera_valid": valid.bool() if valid is not None else torch.ones(feats.shape[:2], dtype=torch.bool),
        }

    def _partner_eps(self, rel: str, ep: int) -> Sequence[int]:
        cands = self.partners.get((rel, int(ep)))
        if not cands:
            raise RuntimeError(f"no M1 partner for {rel} ep={ep}")
        return cands

    def __getitem__(self, i: int) -> dict:
        n = len(self.items)
        last_exc: Exception | None = None
        for attempt in range(16):
            idx = i if attempt == 0 else int(torch.randint(0, n, ()))
            rel, ep, length = self.items[idx]
            try:
                partner = choose_partner(
                    self._partner_eps(rel, int(ep)), int(ep), int(torch.randint(0, 2**31 - 1, ()))
                )
                starts = self.segment_starts(int(length))
                return self.assemble(idx, partner, starts, mask_text=True)
            except Exception as exc:
                last_exc = exc
                logging.warning(
                    "bpv5 stage B __getitem__ retry %d/16 idx=%d rel=%s ep=%s: %s",
                    attempt + 1,
                    idx,
                    rel,
                    ep,
                    exc,
                )
        raise RuntimeError(f"bpv5 stage B __getitem__ failed after retries (seed idx={i})") from last_exc
    def make_heldout_monitor(self):
        if not self.cfg.heldout_monitor or self.cfg.split != "train":
            return None
        from .heldout_monitor import StageBHeldoutMonitor

        # Pad language so blank/alt/wrong ratios isolate Z_demo (not task-text shortcuts).
        return StageBHeldoutMonitor(
            self.derive(split="val", segment_mode="fixed"),
            pad_language=True,
        )


def make_bpv5_stageb_dataset(train_cfg) -> tuple[BPV5StageBDataset, dict]:
    cfg = train_cfg.dataset
    for key in ("num_slots", "spatial_tokens", "max_action_dim", "max_state_dim"):
        if int(getattr(cfg, key)) != int(getattr(train_cfg.policy, key)):
            raise ValueError(f"dataset.{key} != policy.{key}")
    phase = train_cfg.policy.phase
    if phase in {"ae", "b1"} and not train_cfg.policy.stage_a_checkpoint and not train_cfg.policy.pretrained_path:
        raise ValueError(f"{phase} requires policy.stage_a_checkpoint, or policy.pretrained_path when resuming")
    if phase in {"b2", "b3"} and not train_cfg.policy.pretrained_path:
        raise ValueError(f"{phase} requires policy.pretrained_path pointing at the previous phase")
    ds = BPV5StageBDataset(cfg, chunk_size=int(train_cfg.policy.chunk_size))
    return ds, ds.stats
