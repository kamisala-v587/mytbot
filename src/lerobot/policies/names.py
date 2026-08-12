"""Canonical policy names."""

TBOT_SA1 = "TBot_SA1"
TBOT_SA1_WAN = "TBot_SA1_Wan"
BP_TBOT = "BP_TBot"
TBOT_BP = "tbot_bp"
BP_TBOT_V2 = TBOT_BP

TBOT_SA1_LEGACY_ALIASES = frozenset()
TBOT_SA1_WAN_LEGACY_ALIASES = frozenset()

TBOT_SA1_ALIASES = frozenset({TBOT_SA1, "tbot_sa1"})
TBOT_SA1_WAN_ALIASES = frozenset({TBOT_SA1_WAN, "tbot_sa1_wan"})
BP_TBOT_ALIASES = frozenset({BP_TBOT, "bp_tbot"})
BP_TBOT_V2_ALIASES = frozenset({TBOT_BP, "BP_TBot_v2", "bp_tbot_v2", "tbot_bp"})

_LOGGED_LEGACY_POLICY_TYPES: set[str] = set()


def is_tbot_sa1(policy_type: str | None) -> bool:
    return policy_type in TBOT_SA1_ALIASES


def is_tbot_sa1_wan(policy_type: str | None) -> bool:
    return policy_type in TBOT_SA1_WAN_ALIASES


def is_bp_tbot(policy_type: str | None) -> bool:
    return policy_type in BP_TBOT_ALIASES


def is_bp_tbot_v2(policy_type: str | None) -> bool:
    return policy_type in BP_TBOT_V2_ALIASES


def legacy_policy_target(policy_type: str | None) -> str | None:
    return None


def log_legacy_policy_name(policy_type: str | None) -> None:
    return


def canonical_policy_type(policy_type: str | None) -> str | None:
    if is_tbot_sa1(policy_type):
        log_legacy_policy_name(policy_type)
        return TBOT_SA1
    if is_tbot_sa1_wan(policy_type):
        log_legacy_policy_name(policy_type)
        return TBOT_SA1_WAN
    if is_bp_tbot(policy_type):
        return BP_TBOT
    if is_bp_tbot_v2(policy_type):
        return BP_TBOT_V2
    return policy_type
