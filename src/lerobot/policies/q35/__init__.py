from lerobot.policies.q35.configuration_q35 import Q35Config, Q35DatasetConfig

__all__ = [
    "Q35Config",
    "Q35DatasetConfig",
    "Q35Policy",
]


def __getattr__(name: str):
    if name == "Q35Policy":
        from lerobot.policies.q35.modeling_q35 import Q35Policy

        return Q35Policy
    raise AttributeError(name)
