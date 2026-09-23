#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Concurrent dynamic-batch WebSocket server for TBot-SA1 policies."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import socket
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

from serve_lerobot_policy import (
    MsgpackPacker,
    ServeArgs,
    TBotSA1PolicyService,
    _health_check,
    msgpack_unpack,
    parse_args as parse_production_args,
)


@dataclass(frozen=True)
class BatchOptions:
    max_batch_size: int = 8
    batch_wait_ms: float = 10.0
    queue_size: int = 64


@dataclass
class BatchServeArgs:
    serve: ServeArgs
    batch: BatchOptions


@dataclass
class PendingRequest:
    obs: dict[str, Any]
    future: asyncio.Future[dict[str, Any]]
    enqueued_at: float
    connection_id: str


def parse_args() -> BatchServeArgs:
    batch_parser = argparse.ArgumentParser(add_help=False)
    batch_parser.add_argument("--max_batch_size", type=int, default=8)
    batch_parser.add_argument("--batch_wait_ms", type=float, default=10.0)
    batch_parser.add_argument("--queue_size", type=int, default=64)
    batch_args, production_argv = batch_parser.parse_known_args()
    if batch_args.max_batch_size < 1:
        batch_parser.error("--max_batch_size 必须 >= 1")
    if batch_args.batch_wait_ms < 0:
        batch_parser.error("--batch_wait_ms 必须 >= 0")
    if batch_args.queue_size < batch_args.max_batch_size:
        batch_parser.error("--queue_size 必须 >= --max_batch_size")

    original_argv = sys.argv
    try:
        sys.argv = [original_argv[0], *production_argv]
        serve_args = parse_production_args()
    finally:
        sys.argv = original_argv
    return BatchServeArgs(
        serve=serve_args,
        batch=BatchOptions(
            max_batch_size=batch_args.max_batch_size,
            batch_wait_ms=batch_args.batch_wait_ms,
            queue_size=batch_args.queue_size,
        ),
    )


def _collate_nested(values: list[Any]) -> Any:
    if not values:
        raise ValueError("不能合并空 batch。")
    first = values[0]
    if isinstance(first, dict):
        keys = tuple(first)
        if any(tuple(value) != keys for value in values):
            raise ValueError("batch 中的嵌套输入 key 不一致。")
        return {key: _collate_nested([value[key] for value in values]) for key in keys}
    if isinstance(first, torch.Tensor):
        if any(not isinstance(value, torch.Tensor) for value in values):
            raise TypeError("batch 中同一字段的类型不一致。")
        return torch.cat(values, dim=0)
    raise TypeError(f"不支持合批的输入类型: {type(first).__name__}")


class BatchedTBotSA1PolicyService(TBotSA1PolicyService):
    """Adds request-level batch inference without mutating the production service."""

    def infer_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            return []
        prepared = [self._obs_to_inputs(obs) for obs in observations]
        inputs = _collate_nested([item[0] for item in prepared])
        states = [item[1] for item in prepared]

        with torch.inference_mode():
            action_pred, _ = self.policy.predict_action_chunk(inputs, decode_image=False)
        if action_pred.ndim != 3 or action_pred.shape[0] != len(observations):
            raise RuntimeError(
                f"策略输出 shape 异常: {tuple(action_pred.shape)}，"
                f"期望 ({len(observations)}, T, A)"
            )

        results: list[dict[str, Any]] = []
        for batch_idx, state in enumerate(states):
            model_action = action_pred[
                batch_idx, : self.infer_horizon, : self.target_action_dim
            ]
            action = self.unnormalize_action_fn({"action": model_action})["action"]
            model_action_np = model_action.detach().cpu().numpy().astype(np.float32)
            action_np = action.detach().cpu().numpy().astype(np.float32)
            if self.action_mode == "delta" and self.delta_mask is not None:
                state_pad = np.zeros_like(self.delta_mask, dtype=np.float32)
                usable_dims = min(len(state_pad), len(state))
                state_pad[:usable_dims] = state[:usable_dims]
                action_dims = min(action_np.shape[-1], len(self.delta_mask))
                action_np[:, :action_dims] += (
                    state_pad[None, :action_dims] * self.delta_mask[None, :action_dims]
                )
            results.append(
                {
                    "actions": action_np,
                    "action": action_np[0],
                    "model_actions": model_action_np,
                    "model_action": model_action_np[0],
                }
            )
        return results


class DynamicBatchWebsocketPolicyServer:
    """Many async clients feeding one single-owner dynamic batch GPU worker."""

    def __init__(
        self,
        policy: BatchedTBotSA1PolicyService,
        *,
        host: str,
        port: int,
        metadata: dict[str, Any],
        batch_options: BatchOptions,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._options = batch_options
        self._queue: asyncio.Queue[PendingRequest | None] = asyncio.Queue(
            maxsize=batch_options.queue_size
        )
        self._metadata = {
            **metadata,
            "dynamic_batching": True,
            "max_batch_size": batch_options.max_batch_size,
            "batch_wait_ms": batch_options.batch_wait_ms,
            "queue_size": batch_options.queue_size,
        }
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        worker = asyncio.create_task(self._batch_worker(), name="dynamic-batch-worker")
        try:
            async with websocket_server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            await self._queue.put(None)
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        connection_id = str(websocket.remote_address)
        logging.info("客户端已连接: %s", connection_id)
        packer = MsgpackPacker()
        await websocket.send(packer.pack(self._metadata))
        while True:
            future: asyncio.Future[dict[str, Any]] | None = None
            try:
                started_at = time.monotonic()
                obs = msgpack_unpack(await websocket.recv())
                if not isinstance(obs, dict):
                    raise TypeError("请求 payload 必须是字典。")
                future = asyncio.get_running_loop().create_future()
                request = PendingRequest(obs, future, time.monotonic(), connection_id)
                try:
                    self._queue.put_nowait(request)
                except asyncio.QueueFull as exc:
                    raise RuntimeError("推理队列已满，请稍后重试。") from exc
                action = await future
                timing = action.setdefault("server_timing", {})
                timing["total_ms"] = (time.monotonic() - started_at) * 1000.0
                await websocket.send(packer.pack(action))
            except websockets.ConnectionClosed:
                if future is not None and not future.done():
                    future.cancel()
                logging.info("客户端断开: %s", connection_id)
                break
            except asyncio.CancelledError:
                if future is not None and not future.done():
                    future.cancel()
                raise
            except Exception as exc:
                logging.exception("请求处理失败 | client=%s", connection_id)
                if future is not None and not future.done():
                    future.cancel()
                try:
                    await websocket.send(str(exc))
                    await websocket.close(
                        code=websockets.frames.CloseCode.INTERNAL_ERROR,
                        reason="Internal server error",
                    )
                except websockets.ConnectionClosed:
                    pass
                break

    async def _batch_worker(self) -> None:
        wait_seconds = self._options.batch_wait_ms / 1000.0
        while True:
            first = await self._queue.get()
            if first is None:
                self._queue.task_done()
                break
            batch = [first]
            deadline = time.monotonic() + wait_seconds
            while len(batch) < self._options.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                if item is None:
                    self._queue.task_done()
                    await self._queue.put(None)
                    break
                batch.append(item)

            active = [item for item in batch if not item.future.cancelled()]
            if active:
                infer_started = time.monotonic()
                try:
                    # 唯一 worker 串行持有模型；在线程中执行以免 GPU 推理阻塞 WebSocket 事件循环。
                    results = await asyncio.to_thread(
                        self._policy.infer_batch,
                        [item.obs for item in active],
                    )
                    infer_ms = (time.monotonic() - infer_started) * 1000.0
                    for item, result in zip(active, results, strict=True):
                        if item.future.cancelled():
                            continue
                        result["server_timing"] = {
                            "queue_ms": (infer_started - item.enqueued_at) * 1000.0,
                            "infer_ms": infer_ms,
                            "batch_size": len(active),
                        }
                        item.future.set_result(result)
                    logging.info(
                        "动态批推理完成 | batch_size=%d | infer_ms=%.1f | queue=%d",
                        len(active),
                        infer_ms,
                        self._queue.qsize(),
                    )
                except Exception as exc:
                    logging.exception("动态批推理失败 | batch_size=%d", len(active))
                    for item in active:
                        if not item.future.done():
                            item.future.set_exception(exc)
            for _ in batch:
                self._queue.task_done()


def run_batch_server(
    policy: BatchedTBotSA1PolicyService,
    args: ServeArgs,
    batch_options: BatchOptions,
) -> None:
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError:
        local_ip = "unknown"
    logging.info(
        "动态批 WebSocket 服务 | host=%s ip=%s port=%d | max_batch=%d | wait_ms=%.1f",
        args.host,
        local_ip,
        args.port,
        batch_options.max_batch_size,
        batch_options.batch_wait_ms,
    )
    DynamicBatchWebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
        batch_options=batch_options,
    ).serve_forever()


def main(args: BatchServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))
    policy = BatchedTBotSA1PolicyService(args.serve)
    run_batch_server(policy, args.serve, args.batch)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)
    main(parse_args())
