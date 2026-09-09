"""Register benchmark configuration choices used by standalone entrypoints.

`TrainPipelineConfig.from_pretrained` resolves both `dataset.type` and
`policy.type` through draccus choice registries. The decorators that populate
those registries only run after each configuration module is imported, so the
standalone benchmark entrypoints must explicitly import the built-in choices
before parsing a config.
"""

from __future__ import annotations


def register_benchmark_configs() -> None:
    """Register third-party plugins plus built-in BPVA and TBot choices.

    Imports are cached, so this is safe to call repeatedly.
    """
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()

    # Import side effects register dataset and policy choices in both registries.
    import lerobot.policies.BPVA.configuration_bpva  # noqa: F401
    import lerobot.policies.BPVAv2.configuration_bpva  # noqa: F401
    import lerobot.policies.TBot_SA1.configuration_tbot_sa1  # noqa: F401


def register_bpva_configs() -> None:
    """Backward-compatible alias for existing benchmark callers."""
    register_benchmark_configs()
