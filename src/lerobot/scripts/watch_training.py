#!/usr/bin/env python
"""Watch training logs and notify admins when training stalls or errors."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lerobot.scripts.send_error_to_dingtalk import send_error_to_dingtalk

TAIL_BYTES = 8192
TRAIN_ERROR_MARKERS = ("Traceback (most recent call last)", "ERROR ")
TRAIN_DONE_MARKER = "End of training"


@dataclass
class LossSnapshot:
    step: int | None
    timestamp: str | None
    line: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor loss.log / train.log and alert via DingTalk when training stalls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Training output directory containing loss.log (and optionally train.log).",
    )
    parser.add_argument(
        "--loss-log",
        type=Path,
        default=None,
        help="Path to loss.log. Overrides --output-dir/loss.log when set.",
    )
    parser.add_argument(
        "--train-log",
        type=Path,
        default=None,
        help="Optional train.log for error/done detection. Defaults to <output-dir>/train.log.",
    )
    parser.add_argument(
        "--check-interval",
        type=float,
        default=60.0,
        help="Seconds between checks (default: 60).",
    )
    parser.add_argument(
        "--stale-threshold",
        type=float,
        default=600.0,
        help="Alert if loss.log mtime is older than this many seconds (default: 600).",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=1800.0,
        help="Minimum seconds between duplicate alerts (default: 1800).",
    )
    parser.add_argument(
        "--heartbeat-steps",
        type=int,
        default=0,
        help="Send a heartbeat DingTalk message every N new steps (0 = disabled).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check and exit (useful for cron).",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None]:
    if args.loss_log is not None:
        loss_log = args.loss_log.expanduser().resolve()
        train_log = args.train_log.expanduser().resolve() if args.train_log else None
        return loss_log, train_log

    if args.output_dir is None:
        print("Error: provide --output-dir or --loss-log.", file=sys.stderr)
        sys.exit(2)

    output_dir = args.output_dir.expanduser().resolve()
    loss_log = output_dir / "loss.log"
    train_log = (
        args.train_log.expanduser().resolve()
        if args.train_log is not None
        else output_dir / "train.log"
    )
    return loss_log, train_log


def read_tail_text(path: Path, nbytes: int = TAIL_BYTES) -> str:
    with path.open("rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - nbytes))
        return f.read().decode("utf-8", errors="replace")


def parse_last_loss_line(text: str) -> LossSnapshot:
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped or stripped.startswith("step,"):
            continue
        parts = stripped.split(",", 2)
        if len(parts) < 2:
            return LossSnapshot(step=None, timestamp=None, line=stripped)
        try:
            step = int(parts[0])
        except ValueError:
            return LossSnapshot(step=None, timestamp=None, line=stripped)
        return LossSnapshot(step=step, timestamp=parts[1], line=stripped)
    return LossSnapshot(step=None, timestamp=None, line=None)


def scan_train_log(train_log: Path, seen_error_lines: set[str]) -> list[str]:
    if not train_log.is_file():
        return []

    alerts: list[str] = []
    tail = read_tail_text(train_log)
    for line in tail.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if TRAIN_DONE_MARKER in stripped:
            alerts.append(f"训练已正常结束（检测到 '{TRAIN_DONE_MARKER}'）")
            break
        if any(marker in stripped for marker in TRAIN_ERROR_MARKERS):
            if stripped not in seen_error_lines:
                seen_error_lines.add(stripped)
                alerts.append(f"train.log 异常:\n{stripped}")
    return alerts


def notify(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message.splitlines()[0]}")
    send_error_to_dingtalk(message)


def maybe_alert(
    message: str,
    last_alert_at: float,
    cooldown: float,
) -> float:
    now = time.time()
    if now - last_alert_at < cooldown:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] (cooldown) {message.splitlines()[0]}")
        return last_alert_at
    notify(message)
    return now


def check_once(
    loss_log: Path,
    train_log: Path | None,
    stale_threshold: float,
    cooldown: float,
    heartbeat_steps: int,
    state: dict,
) -> None:
    seen_error_lines: set[str] = state.setdefault("seen_error_lines", set())
    last_alert_at: float = state.setdefault("last_alert_at", 0.0)
    last_step: int | None = state.get("last_step")
    training_done: bool = state.setdefault("training_done", False)
    last_heartbeat_step: int = state.setdefault("last_heartbeat_step", 0)

    if train_log is not None and train_log.is_file():
        for alert in scan_train_log(train_log, seen_error_lines):
            if TRAIN_DONE_MARKER in alert:
                training_done = True
                state["training_done"] = True
            last_alert_at = maybe_alert(f"⚠️ 训练 watchdog\n{alert}", last_alert_at, cooldown)

    if training_done:
        print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] training marked done, skip stale checks")
        return

    if not loss_log.is_file():
        last_alert_at = maybe_alert(
            f"⚠️ 训练 watchdog\nloss.log 不存在: {loss_log}",
            last_alert_at,
            cooldown,
        )
        state["last_alert_at"] = last_alert_at
        return

    mtime = loss_log.stat().st_mtime
    stale_secs = time.time() - mtime
    snapshot = parse_last_loss_line(read_tail_text(loss_log))

    if snapshot.step is not None and snapshot.step != last_step:
        print(
            f"[{datetime.now():%Y-%m-%d %H:%M:%S}] ok step={snapshot.step} "
            f"ts={snapshot.timestamp} stale={stale_secs:.0f}s"
        )
        state["last_step"] = snapshot.step

        if (
            heartbeat_steps > 0
            and snapshot.step >= last_heartbeat_step + heartbeat_steps
        ):
            notify(
                "✅ 训练 watchdog 心跳\n"
                f"step={snapshot.step}\n"
                f"timestamp={snapshot.timestamp}\n"
                f"log={loss_log}"
            )
            state["last_heartbeat_step"] = snapshot.step

    if stale_secs > stale_threshold:
        detail = snapshot.line or "(empty loss.log)"
        last_alert_at = maybe_alert(
            "⚠️ 训练可能已停止\n"
            f"loss.log 已 {stale_secs:.0f}s 未更新 (阈值 {stale_threshold:.0f}s)\n"
            f"path={loss_log}\n"
            f"最后一条: {detail}",
            last_alert_at,
            cooldown,
        )

    state["last_alert_at"] = last_alert_at


def main() -> None:
    args = parse_args()
    loss_log, train_log = resolve_paths(args)
    state: dict = {}

    print(
        f"watching loss.log={loss_log}\n"
        f"train.log={train_log if train_log and train_log.is_file() else '(disabled or missing)'}\n"
        f"check_interval={args.check_interval}s stale_threshold={args.stale_threshold}s "
        f"cooldown={args.cooldown}s heartbeat_steps={args.heartbeat_steps}"
    )

    while True:
        check_once(
            loss_log=loss_log,
            train_log=train_log,
            stale_threshold=args.stale_threshold,
            cooldown=args.cooldown,
            heartbeat_steps=args.heartbeat_steps,
            state=state,
        )
        if args.once:
            break
        time.sleep(args.check_interval)


if __name__ == "__main__":
    main()
