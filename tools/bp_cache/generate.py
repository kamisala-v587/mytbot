"""Module entry point for BP cache generation."""
from __future__ import annotations

from tools.run_bp_cache import main, parse_args

__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    raise SystemExit(main())
