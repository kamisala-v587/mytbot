"""按 checkpoint type 或 POLICY 常量选择适配器。"""

from __future__ import annotations

import importlib
import logging
import pkgutil

from .common import ServerConfig, prepare_import_path, read_ckpt_type

prepare_import_path()

_SKIP_MODULES = frozenset({"common", "base", "registry"})


def iter_adapters():
    import policy_runtime

    adapters = []
    for module_info in pkgutil.iter_modules(policy_runtime.__path__):
        if module_info.name in _SKIP_MODULES or module_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{policy_runtime.__name__}.{module_info.name}")
        adapter = getattr(module, "ADAPTER", None)
        if adapter is not None:
            adapters.append(adapter)
    adapters.sort(key=lambda item: item.name)
    if not adapters:
        raise RuntimeError("policy_runtime 下没有导出 ADAPTER 的策略模块。")
    return adapters


def create_policy(config: ServerConfig):
    ckpt_type = read_ckpt_type(config.ckpt_path)
    adapters = iter_adapters()
    matched = [adapter for adapter in adapters if adapter.matches(config.policy, ckpt_type)]
    known = ", ".join(f"{adapter.name}({', '.join(adapter.aliases)})" for adapter in adapters)
    if not matched:
        raise ValueError(
            f"没有适配器能处理 POLICY={config.policy!r}、checkpoint type={ckpt_type!r}。已注册: {known}。"
            "新增策略时，在 server/policy_runtime/ 下增加模块并导出 ADAPTER。"
        )
    if len(matched) > 1:
        names = ", ".join(adapter.name for adapter in matched)
        raise ValueError(f"多个适配器同时匹配: {names}")
    logging.info("选择适配器 %s | checkpoint type=%s", matched[0].name, ckpt_type)
    return matched[0](config)
