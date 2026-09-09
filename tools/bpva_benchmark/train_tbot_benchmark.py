"""Short TBot-SA1 training-throughput benchmark without eval/checkpoint/W&B.

This entrypoint is intentionally a thin TBot-specific wrapper around
``train_benchmark`` so BPVA and TBot baselines share the same timing,
instrumentation, Accelerate, reporting, and dataloader override behavior.
"""

from __future__ import annotations

from typing import Any, Callable

from lerobot.policies.names import is_tbot_sa1

from . import train_benchmark


def _load_tbot(path: str, base_loader: Callable[[str], Any] | None = None) -> Any:
    loader = train_benchmark._load if base_loader is None else base_loader
    cfg = loader(path)
    if not is_tbot_sa1(getattr(cfg.policy, "type", None)):
        raise ValueError(
            "train_tbot_benchmark 仅支持 policy.type=TBot_SA1/tbot_sa1；"
            f"当前为 {getattr(cfg.policy, 'type', None)!r}"
        )
    return cfg


def main(argv: list[str] | None = None) -> None:
    original_load = train_benchmark._load

    def load_checked(path: str) -> Any:
        return _load_tbot(path, base_loader=original_load)

    train_benchmark._load = load_checked
    try:
        train_benchmark.main(argv)
    finally:
        train_benchmark._load = original_load


if __name__ == "__main__":
    main()
