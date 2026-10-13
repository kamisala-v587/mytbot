"""把训练配置里的 bp_action_use / bp_state_use 应用到缓存示教上。

缓存里保存的是完整 action 和 state。推理前按训练同一条规则裁维度，
避免把训练时关掉的示教动作或关节送进模型。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import prepare_import_path


def load_bp_dim_spec(path: str | Path) -> tuple[Any, Any]:
    """从 json / jsonc 读取示教维度开关。

    优先用 dataset.bp_action_use / dataset.bp_state_use，也接受写在根上的同名字段。
    缺一个字段时另一个保持训练默认：全部保留。
    """
    prepare_import_path()
    from lerobot.utils.jsonc import load_jsonc

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"BP 维度配置不存在: {config_path}")
    payload = load_jsonc(config_path)
    dataset = payload.get("dataset")
    source = dataset if isinstance(dataset, dict) else payload
    if not isinstance(source, dict):
        raise ValueError(f"BP 维度配置必须是对象: {config_path}")
    if "bp_action_use" not in source and "bp_state_use" not in source:
        raise ValueError(f"{config_path} 没有 bp_action_use 或 bp_state_use")
    return source.get("bp_action_use", True), source.get("bp_state_use", True)


def resolve_bp_dim_use(
    explicit_path: str | None,
    train_cfg: Any,
) -> tuple[Any, Any, str]:
    """显式 json 优先；否则用 checkpoint 的 train_config.json。都没有则全部保留。"""
    path = str(explicit_path or "").strip()
    if path:
        action_use, state_use = load_bp_dim_spec(path)
        return action_use, state_use, str(Path(path).expanduser())
    dataset = getattr(train_cfg, "dataset", None) if train_cfg is not None else None
    if dataset is None:
        return True, True, "default"
    return (
        getattr(dataset, "bp_action_use", True),
        getattr(dataset, "bp_state_use", True),
        "checkpoint train_config.json",
    )


def apply_bp_dim_use(prompt: dict[str, Any], bp_action_use: Any, bp_state_use: Any) -> dict[str, Any]:
    """按训练的 BPVAv3SelectPromptDimsFn 裁一份示教。不改传入的字典。"""
    prepare_import_path()
    from lerobot.policies.BPVAv3.transform_bpvav3 import BPVAv3SelectPromptDimsFn
    from lerobot.transforms.core_bp import BP_PREFIX

    wrapped = {BP_PREFIX: dict(prompt)}
    selected = BPVAv3SelectPromptDimsFn(bp_action_use=bp_action_use, bp_state_use=bp_state_use)(wrapped)
    return selected[BP_PREFIX]
