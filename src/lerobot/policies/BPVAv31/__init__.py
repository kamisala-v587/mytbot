from __future__ import annotations

import importlib
from typing import Any

_SYMBOL_IMPORTS = {
    "BPVAv31Config": "lerobot.policies.BPVAv31.configuration_bpvav31",
    "BPVAv31DatasetConfig": "lerobot.policies.BPVAv31.configuration_bpvav31",
    "BPVAv31Model": "lerobot.policies.BPVAv31.modeling_bpvav31",
    "BPVAv31Policy": "lerobot.policies.BPVAv31.modeling_bpvav31",
}
__all__ = list(_SYMBOL_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_SYMBOL_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
