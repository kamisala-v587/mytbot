from __future__ import annotations

import importlib
from typing import Any

_SYMBOL_IMPORTS = {
    "BPTBotV2Config": "lerobot.policies.BP_TBot_v2.configuration_bp_tbot",
    "BPTBotV2DatasetConfig": "lerobot.policies.BP_TBot_v2.configuration_bp_tbot",
    "BPTransformerObsEncoder": "lerobot.policies.BP_TBot_v2.bp_transformer_obs_encoder",
    "BPTransformerObsEncoderOutput": "lerobot.policies.BP_TBot_v2.bp_transformer_obs_encoder",
    "BPObsEncoder": "lerobot.policies.BP_TBot_v2.bp_transformer_obs_encoder",
    "BPObsEncoderOutput": "lerobot.policies.BP_TBot_v2.bp_transformer_obs_encoder",
    "TBotBPModel": "lerobot.policies.BP_TBot_v2.modeling_bp_tbot",
    "TBotBPPolicy": "lerobot.policies.BP_TBot_v2.modeling_bp_tbot",
    "BPTBotModel": "lerobot.policies.BP_TBot_v2.modeling_bp_tbot",
    "BPTBotPolicy": "lerobot.policies.BP_TBot_v2.modeling_bp_tbot",
}

__all__ = list(_SYMBOL_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _SYMBOL_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_SYMBOL_IMPORTS[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
