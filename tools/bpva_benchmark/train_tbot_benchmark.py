"""TBot-SA1 短程训练吞吐测评（不评测、不存 checkpoint、不启 W&B）。

薄封装：复用 ``train_benchmark`` 的计时、插桩、Accelerate、报告与 dataloader
覆盖逻辑，仅额外校验 ``policy.type`` 为 TBot_SA1。
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
    train_benchmark.main(argv, load_config=_load_tbot)


if __name__ == "__main__":
    main()
