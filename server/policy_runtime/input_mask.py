"""按路径屏蔽观测或组装后的模型输入。

路径用点号，最后一维可用 ``[i,j]``。含点的 key（如 ``observation.images.image0``）
按最长前缀匹配。整段 ``behavior_prompt.action`` 会同时把 ``action_is_pad`` 置为 True。
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

OBS_IMAGES = "observation.images"
OBS_STATE = "observation.state"
CAMERA_ALIASES = {
    f"{OBS_IMAGES}.image0": ("cam_high", "head", "image0", "observation.images.cam_high"),
    f"{OBS_IMAGES}.image1": ("cam_left_wrist", "left_wrist", "left", "image1", "observation.images.cam_left_wrist"),
    f"{OBS_IMAGES}.image2": (
        "cam_right_wrist",
        "right_wrist",
        "right",
        "image2",
        "observation.images.cam_right_wrist",
    ),
}

_SPEC_RE = re.compile(r"^(?P<path>[^\[\]]+?)(?:\[(?P<indices>[^\]]*)\])?\s*$")
_IMAGE_HINTS = ("image", "cam", "camera", "pixel")

CLIENT_IMAGE_ALIASES: dict[str, tuple[str, ...]] = {}
for _canonical, _aliases in CAMERA_ALIASES.items():
    names = (_canonical, *_aliases)
    for _name in names:
        CLIENT_IMAGE_ALIASES[_name] = names

CLIENT_STATE_ALIASES = (OBS_STATE, "state", "qpos")
CLIENT_PROMPT_ALIASES = ("prompt", "task")


@dataclass(frozen=True)
class MaskRule:
    spec: str
    path: tuple[str, ...]
    indices: tuple[int, ...] | None
    stage: str  # "client" | "model"

    @property
    def joined(self) -> str:
        return ".".join(self.path)


def parse_mask_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        parts = [str(item).strip() for item in value]
    else:
        parts = []
        buf: list[str] = []
        depth = 0
        for char in str(value):
            if char == "[":
                depth += 1
            elif char == "]" and depth:
                depth -= 1
            if char == "," and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue
            buf.append(char)
        parts.append("".join(buf).strip())
    return [item for item in parts if item]


def parse_mask_spec(spec: str) -> MaskRule:
    text = str(spec).strip()
    matched = _SPEC_RE.match(text)
    if not matched:
        raise ValueError(f"无法解析 INPUT_MASKS 项: {spec!r}")
    path_text = matched.group("path").strip()
    if not path_text:
        raise ValueError(f"INPUT_MASKS 路径为空: {spec!r}")
    path = tuple(part for part in path_text.split(".") if part)
    if not path:
        raise ValueError(f"INPUT_MASKS 路径为空: {spec!r}")
    raw_indices = matched.group("indices")
    indices = None
    if raw_indices is not None:
        items = [item.strip() for item in raw_indices.split(",") if item.strip()]
        if not items:
            raise ValueError(f"INPUT_MASKS 索引为空: {spec!r}")
        try:
            indices = tuple(int(item) for item in items)
        except ValueError as exc:
            raise ValueError(f"INPUT_MASKS 索引必须是整数: {spec!r}") from exc
    stage = "model" if path[0] == "behavior_prompt" else "client"
    return MaskRule(spec=text, path=path, indices=indices, stage=stage)


def parse_mask_rules(value: Any) -> list[MaskRule]:
    return [parse_mask_spec(item) for item in parse_mask_list(value)]


def apply_client_masks(obs: dict[str, Any], rules: list[MaskRule]) -> dict[str, Any]:
    if not rules:
        return obs
    masked = copy.copy(obs)
    if isinstance(masked.get("images"), dict):
        masked["images"] = dict(masked["images"])
    for rule in rules:
        if rule.stage != "client":
            continue
        _apply_client_rule(masked, rule)
    return masked


def apply_model_masks(inputs: dict[str, Any], rules: list[MaskRule]) -> dict[str, Any]:
    if not rules:
        return inputs
    for rule in rules:
        if rule.stage != "model":
            continue
        _apply_model_rule(inputs, rule)
    return inputs


def _apply_client_rule(obs: dict[str, Any], rule: MaskRule) -> None:
    joined = rule.joined
    if joined in CLIENT_PROMPT_ALIASES or joined == "instruction":
        if rule.indices is not None:
            raise ValueError(f"{rule.spec!r} 不能对文本字段使用索引")
        obs["prompt"] = ""
        obs["task"] = ""
        return
    if joined in CLIENT_STATE_ALIASES:
        key = next((alias for alias in CLIENT_STATE_ALIASES if alias in obs), None)
        if key is None:
            raise KeyError(f"{rule.spec!r} 未找到当前 state（observation.state / state / qpos）")
        obs[key] = _mask_array(obs[key], rule.indices, spec=rule.spec)
        return
    image_aliases = _client_image_aliases(joined)
    if image_aliases is not None:
        if rule.indices is not None:
            raise ValueError(f"{rule.spec!r} 屏蔽相机时不要写索引，直接丢掉该路图像")
        removed = _drop_client_images(obs, image_aliases)
        if not removed:
            raise KeyError(f"{rule.spec!r} 未找到对应相机。可用键: {sorted(image_aliases)}")
        return
    parent, key = _resolve_parent(obs, rule.path, spec=rule.spec)
    parent[key] = _mask_value(parent[key], rule)


def _apply_model_rule(inputs: dict[str, Any], rule: MaskRule) -> None:
    if rule.path == ("behavior_prompt",) and rule.indices is None:
        prompt = inputs.get("behavior_prompt")
        if not isinstance(prompt, dict):
            raise KeyError(f"{rule.spec!r} 需要已组装的 behavior_prompt")
        _mask_entire_behavior_prompt(prompt)
        return
    parent, key = _resolve_parent(inputs, rule.path, spec=rule.spec)
    if key == "action" and rule.indices is None and _is_behavior_prompt_parent(inputs, parent):
        parent["action"] = _mask_array(parent["action"], None, spec=rule.spec)
        if "action_is_pad" in parent:
            parent["action_is_pad"] = _filled_bool(parent["action_is_pad"], True)
        return
    if key == "state" and rule.indices is None and _is_behavior_prompt_parent(inputs, parent):
        parent["state"] = _mask_array(parent["state"], None, spec=rule.spec)
        parent["state_is_available"] = _unavailable_like(parent["state"], parent.get("state_is_available"))
        return
    if key == "mask" and rule.indices is None:
        parent[key] = _filled_bool(parent[key], False)
        return
    if key == "action_is_pad" and rule.indices is None:
        parent[key] = _filled_bool(parent[key], True)
        return
    parent[key] = _mask_value(parent[key], rule)
    prompt = inputs.get("behavior_prompt")
    if not isinstance(prompt, dict) or rule.indices is not None:
        return
    pixels = prompt.get("bp_pixel_values")
    if not isinstance(pixels, dict):
        return
    if key == "bp_pixel_values" and parent is prompt:
        _invalidate_bp_images(prompt, list(pixels))
        return
    if parent is pixels and key in pixels:
        _invalidate_bp_images(prompt, [key])


def _unavailable_like(state: Any, existing: Any) -> Any:
    if existing is not None:
        return _filled_bool(existing, False)
    shape = tuple(state.shape[:-1])
    if isinstance(state, torch.Tensor):
        return torch.zeros(shape, dtype=torch.bool, device=state.device)
    return np.zeros(shape, dtype=bool)


def _invalidate_bp_images(prompt: dict[str, Any], cameras: list[str]) -> None:
    """把示教相机标成无效。没有 image_masks 时按像素的 chunk 形状补上。"""
    pixels = prompt.get("bp_pixel_values")
    if not isinstance(pixels, dict):
        return
    existing = prompt.get("image_masks")
    if not isinstance(existing, dict):
        existing = {}
    updated = dict(existing)
    for camera in cameras:
        if camera in existing:
            updated[camera] = _filled_bool(existing[camera], False)
            continue
        pixel = pixels.get(camera)
        if pixel is None:
            continue
        if isinstance(pixel, torch.Tensor):
            shape = pixel.shape[:2] if pixel.ndim >= 4 else pixel.shape[:1]
            updated[camera] = torch.zeros(shape, dtype=torch.bool, device=pixel.device)
        else:
            array = np.asarray(pixel)
            shape = array.shape[:2] if array.ndim >= 4 else array.shape[:1]
            updated[camera] = np.zeros(shape, dtype=bool)
    prompt["image_masks"] = updated


def _mask_entire_behavior_prompt(prompt: dict[str, Any]) -> None:
    for key, value in list(prompt.items()):
        if key == "mask":
            prompt[key] = _filled_bool(value, False)
        elif key == "action_is_pad":
            prompt[key] = _filled_bool(value, True)
        elif key == "state_is_available":
            prompt[key] = _filled_bool(value, False)
        elif key == "image_masks" and isinstance(value, dict):
            prompt[key] = {camera: _filled_bool(flag, False) for camera, flag in value.items()}
        elif key == "bp_image_grid_thw":
            # 网格是 patch 布局，不是内容。image_masks 全 False 时模型仍要用它分配占位 visual token。
            continue
        else:
            prompt[key] = _zero_tree(value)


def _mask_value(value: Any, rule: MaskRule) -> Any:
    if isinstance(value, dict):
        if rule.indices is not None:
            raise ValueError(f"{rule.spec!r} 不能对字典使用索引，请写到具体子字段")
        return {key: _zero_tree(item) for key, item in value.items()}
    if _is_image_name(rule.path[-1]) and rule.indices is None:
        return _zero_tree(value)
    return _mask_array(value, rule.indices, spec=rule.spec)


def _zero_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _zero_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool:
            return torch.zeros_like(value)
        return torch.zeros_like(value)
    if isinstance(value, np.ndarray):
        return np.zeros_like(value)
    if isinstance(value, (str, bytes)):
        return type(value)()
    return value


def _mask_array(value: Any, indices: tuple[int, ...] | None, *, spec: str) -> Any:
    array = _as_writable(value)
    if indices is None:
        if isinstance(array, torch.Tensor):
            return torch.zeros_like(array)
        return np.zeros_like(array)
    last_dim = int(array.shape[-1])
    resolved = []
    for index in indices:
        actual = index if index >= 0 else last_dim + index
        if actual < 0 or actual >= last_dim:
            raise IndexError(f"{spec!r} 索引 {index} 超出最后一维 {last_dim}")
        resolved.append(actual)
    if isinstance(array, torch.Tensor):
        array[..., resolved] = 0
        return array
    array[..., resolved] = 0
    return array


def _as_writable(value: Any) -> np.ndarray | torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    array = np.asarray(value)
    if array.dtype == object:
        raise TypeError(f"无法 mask 的类型: {type(value).__name__}")
    return np.array(array, copy=True)


def _filled_bool(value: Any, fill: bool) -> Any:
    if isinstance(value, torch.Tensor):
        return torch.full_like(value, fill, dtype=torch.bool)
    array = np.asarray(value)
    return np.full(array.shape, fill, dtype=bool)


def _is_behavior_prompt_parent(inputs: dict[str, Any], parent: dict[str, Any]) -> bool:
    return parent is inputs.get("behavior_prompt")


def _is_image_name(name: str) -> bool:
    lowered = name.lower()
    return any(hint in lowered for hint in _IMAGE_HINTS)


def _client_image_aliases(joined: str) -> tuple[str, ...] | None:
    if joined in CLIENT_IMAGE_ALIASES:
        return CLIENT_IMAGE_ALIASES[joined]
    if joined.startswith("images."):
        return _client_image_aliases(joined[len("images.") :])
    return None


def _drop_client_images(obs: dict[str, Any], aliases: tuple[str, ...]) -> list[str]:
    removed: list[str] = []
    images = obs.get("images")
    if isinstance(images, dict):
        for alias in aliases:
            if alias in images:
                images.pop(alias)
                removed.append(alias)
    for alias in aliases:
        if alias in obs:
            obs.pop(alias)
            removed.append(alias)
    return removed


def _resolve_parent(root: dict[str, Any], path: tuple[str, ...], *, spec: str) -> tuple[dict[str, Any], str]:
    if not isinstance(root, dict):
        raise KeyError(f"{spec!r} 根节点不是字典")
    node: Any = root
    index = 0
    parents: list[tuple[dict[str, Any], str]] = []
    while index < len(path):
        if not isinstance(node, dict):
            raise KeyError(f"{spec!r} 在 {'.'.join(path[:index])} 处不是字典")
        matched = None
        for end in range(len(path), index, -1):
            key = ".".join(path[index:end])
            if key in node:
                matched = (end, key)
                break
        if matched is None:
            available = ", ".join(sorted(map(str, node.keys()))) or "<empty>"
            raise KeyError(f"{spec!r} 找不到 {'.'.join(path[index:])}；当前层键: {available}")
        end, key = matched
        parents.append((node, key))
        node = node[key]
        index = end
    return parents[-1]


def log_mask_rules(rules: list[MaskRule]) -> None:
    if not rules:
        logging.info("INPUT_MASKS 为空，推理使用原始观测和完整 BP。")
        return
    logging.info("INPUT_MASKS 已启用: %s", [rule.spec for rule in rules])
