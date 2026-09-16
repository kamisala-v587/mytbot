"""TBot-SA1 wrapper for the training dataloader benchmark.

Delegates to ``train_dataloader_benchmark`` so TBot and BPVA share construction,
iteration timing, per-sample metadata, distributed behavior, and output format.
Only validates ``policy.type`` is TBot_SA1.
"""

from __future__ import annotations

from typing import Any, Callable

from lerobot.policies.names import is_tbot_sa1

from . import train_dataloader_benchmark


def _load_tbot_cfg(path: str, base_loader: Callable[[str], Any] | None = None) -> Any:
    loader = train_dataloader_benchmark._load_cfg if base_loader is None else base_loader
    cfg = loader(path)
    if not is_tbot_sa1(getattr(cfg.policy, "type", None)):
        raise ValueError(
            "train_tbot_dataloader_benchmark 仅支持 policy.type=TBot_SA1/tbot_sa1；"
            f"当前为 {getattr(cfg.policy, 'type', None)!r}"
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    train_dataloader_benchmark.main(argv, load_config=_load_tbot_cfg)


if __name__ == "__main__":
    main()
