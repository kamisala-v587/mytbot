from __future__ import annotations

import importlib
from typing import Any

_SYMBOL_IMPORTS = {
    "BPVAv2Config": "lerobot.policies.BPVAv2.configuration_bpva",
    "BPVAv2DatasetConfig": "lerobot.policies.BPVAv2.configuration_bpva",
    "BPTransformerObsEncoder": "lerobot.policies.BPVAv2.bp_transformer_obs_encoder",
    "BPTransformerObsEncoderOutput": "lerobot.policies.BPVAv2.bp_transformer_obs_encoder",
    "BPObsEncoder": "lerobot.policies.BPVAv2.bp_transformer_obs_encoder",
    "BPObsEncoderOutput": "lerobot.policies.BPVAv2.bp_transformer_obs_encoder",
    "BPVAv2Model": "lerobot.policies.BPVAv2.modeling_bpva",
    "BPVAv2Policy": "lerobot.policies.BPVAv2.modeling_bpva",
}
__all__ = list(_SYMBOL_IMPORTS)

def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_SYMBOL_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
