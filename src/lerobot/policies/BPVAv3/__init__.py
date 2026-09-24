from __future__ import annotations

import importlib
from typing import Any

_SYMBOL_IMPORTS = {
    "BPVAv3Config": "lerobot.policies.BPVAv3.configuration_bpvav3",
    "BPVAv3DatasetConfig": "lerobot.policies.BPVAv3.configuration_bpvav3",
    "BPVAv3Model": "lerobot.policies.BPVAv3.modeling_bpvav3",
    "BPVAv3Policy": "lerobot.policies.BPVAv3.modeling_bpvav3",
}
__all__ = list(_SYMBOL_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_SYMBOL_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
