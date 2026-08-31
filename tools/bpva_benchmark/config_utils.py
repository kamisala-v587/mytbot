"""Shared config-loading helpers for the BPVA benchmark tools.

`TrainPipelineConfig.from_pretrained` uses draccus `ChoiceRegistry` lookups for
`dataset.type` / `policy.type`. Those choices are only populated once the
corresponding config module has actually been imported (the
`@DatasetConfig.register_subclass(...)` / `@PreTrainedConfig.register_subclass(...)`
decorators run at import time). The production training entrypoint
(`lerobot.scripts.lerobot_train`) never imports `configuration_bpva` directly;
it only calls `register_third_party_plugins()`, which discovers third-party
`lerobot_*` packages on `sys.path` and relies on some other import already
having pulled in the built-in BPVA config as a side effect (for example via
`lerobot.policies.factory`). A standalone benchmark script has no such
side effect, so parsing a `type: "bpva"` JSONC config fails with
`DecodingError: Couldn't find a choice class for 'bpva'` unless we import
`configuration_bpva` explicitly before calling `TrainPipelineConfig.from_pretrained`.
"""

from __future__ import annotations


def register_bpva_configs() -> None:
    """Import BPVA config modules so their choice-registry decorators run.

    Also runs `register_third_party_plugins()` so behavior matches the
    production `lerobot_train.py` entrypoint for any third-party plugins.
    Safe to call multiple times; imports are cached by Python.
    """
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()

    # Import side effect: registers "bpva" (and legacy aliases) into both
    # DatasetConfig and PreTrainedConfig choice registries.
    import lerobot.policies.BPVA.configuration_bpva  # noqa: F401
