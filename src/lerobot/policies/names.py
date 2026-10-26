"""Canonical policy names."""

TBOT_SA1 = "TBot_SA1"
TBOT_SA1_WAN = "TBot_SA1_Wan"
BPVA = "bpva"
BPVAV2 = "bpvav2"
BPVAV3 = "bpvav3"
BPVAV31 = "bpvav31"
BPV4_BPEN = "bpv4_bpen"
BPV4_XDEMO = "bpv4_xdemo"
BPV4CDX_AE = "bpv4cdx_ae"
BPV4CDX_B = "bpv4cdx_b"

TBOT_SA1_LEGACY_ALIASES = frozenset()
TBOT_SA1_WAN_LEGACY_ALIASES = frozenset()

TBOT_SA1_ALIASES = frozenset({TBOT_SA1, "tbot_sa1"})
TBOT_SA1_WAN_ALIASES = frozenset({TBOT_SA1_WAN, "tbot_sa1_wan"})
BPVA_ALIASES = frozenset({BPVA, "BPVA", "tbot_bp", "BP_TBot_v2", "bp_tbot_v2"})
BPVAV2_ALIASES = frozenset({BPVAV2})
BPVAV3_ALIASES = frozenset({BPVAV3})
BPVAV31_ALIASES = frozenset({BPVAV31, "bpvav3.1", "bpv3.1"})
BPV4_BPEN_ALIASES = frozenset({BPV4_BPEN})
BPV4_XDEMO_ALIASES = frozenset({BPV4_XDEMO})
BPV4CDX_AE_ALIASES = frozenset({BPV4CDX_AE})
BPV4CDX_B_ALIASES = frozenset({BPV4CDX_B})

_LOGGED_LEGACY_POLICY_TYPES: set[str] = set()


def is_tbot_sa1(policy_type: str | None) -> bool:
    return policy_type in TBOT_SA1_ALIASES


def is_tbot_sa1_wan(policy_type: str | None) -> bool:
    return policy_type in TBOT_SA1_WAN_ALIASES


def is_bpva(policy_type: str | None) -> bool:
    return policy_type in BPVA_ALIASES


def is_bpvav2(policy_type: str | None) -> bool:
    return policy_type in BPVAV2_ALIASES


def is_bpvav3(policy_type: str | None) -> bool:
    return policy_type in BPVAV3_ALIASES


def is_bpvav31(policy_type: str | None) -> bool:
    return policy_type in BPVAV31_ALIASES


def is_bpv4_bpen(policy_type: str | None) -> bool:
    return policy_type in BPV4_BPEN_ALIASES


def is_bpv4_xdemo(policy_type: str | None) -> bool:
    return policy_type in BPV4_XDEMO_ALIASES


def is_bpv4cdx_ae(policy_type: str | None) -> bool:
    return policy_type in BPV4CDX_AE_ALIASES


def is_bpv4cdx_b(policy_type: str | None) -> bool:
    return policy_type in BPV4CDX_B_ALIASES


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
    if is_bpva(policy_type):
        return BPVA
    if is_bpvav2(policy_type):
        return BPVAV2
    if is_bpvav3(policy_type):
        return BPVAV3
    if is_bpvav31(policy_type):
        return BPVAV31
    if is_bpv4_bpen(policy_type):
        return BPV4_BPEN
    if is_bpv4_xdemo(policy_type):
        return BPV4_XDEMO
    if is_bpv4cdx_ae(policy_type):
        return BPV4CDX_AE
    if is_bpv4cdx_b(policy_type):
        return BPV4CDX_B
    return policy_type
