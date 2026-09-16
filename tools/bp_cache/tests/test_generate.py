from __future__ import annotations

import pytest


def test_generate_reuses_public_entrypoint() -> None:
    from tools.bp_cache import generate
    from tools import run_bp_cache

    assert generate.main is run_bp_cache.main
    assert generate.parse_args is run_bp_cache.parse_args


def test_generate_help_does_not_run_pipeline() -> None:
    from tools.bp_cache.generate import main

    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
