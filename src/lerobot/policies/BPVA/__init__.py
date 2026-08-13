from __future__ import annotations

import importlib
from typing import Any

_SYMBOL_IMPORTS = {
    "BPVAConfig": "lerobot.policies.BPVA.configuration_bpva",
    "BPVADatasetConfig": "lerobot.policies.BPVA.configuration_bpva",
    "BPTransformerObsEncoder": "lerobot.policies.BPVA.bp_transformer_obs_encoder",
    "BPTransformerObsEncoderOutput": "lerobot.policies.BPVA.bp_transformer_obs_encoder",
    "BPObsEncoder": "lerobot.policies.BPVA.bp_transformer_obs_encoder",
    "BPObsEncoderOutput": "lerobot.policies.BPVA.bp_transformer_obs_encoder",
    "BPVAModel": "lerobot.policies.BPVA.modeling_bpva",
    "BPVAPolicy": "lerobot.policies.BPVA.modeling_bpva",
}

__all__ = list(_SYMBOL_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_SYMBOL_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
