#!/usr/bin/env python3
"""Parse JSONC and emit shell-safe KEY=value lines for bash eval."""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path
from typing import Any


def strip_jsonc_comments(text: str) -> str:
    """Remove // and /* */ comments without touching string literals."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if escape:
            result.append(char)
            escape = False
            index += 1
            continue
        if char == "\\" and in_string:
            result.append(char)
            escape = True
            index += 1
            continue
        if char == '"':
            in_string = not in_string
            result.append(char)
            index += 1
            continue
        if not in_string and char == "/" and index + 1 < len(text) and text[index + 1] == "/":
            while index < len(text) and text[index] != "\n":
                index += 1
            continue
        result.append(char)
        index += 1
    return "".join(result)


def flatten_config(obj: Any, flat: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flatten nested JSON objects; leaf keys map to UPPER_SNAKE env names."""
    if flat is None:
        flat = {}
    if not isinstance(obj, dict):
        return flat
    for key, value in obj.items():
        if isinstance(value, dict):
            flatten_config(value, flat)
        else:
            flat[str(key).upper()] = value
    return flat


def to_shell_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def load_jsonc(path: Path | str) -> dict[str, Any]:
    """Load a JSONC file into a dict."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(strip_jsonc_comments(text))
    if not isinstance(payload, dict):
        raise ValueError(f"JSONC root must be an object: {config_path}")
    return payload


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: load_jsonc_config.py <config.jsonc> [PRESET_KEY1,PRESET_KEY2,...]", file=sys.stderr)
        return 1

    config_path = Path(sys.argv[1])
    preset_keys = set()
    if len(sys.argv) > 2 and sys.argv[2]:
        preset_keys = {item.strip() for item in sys.argv[2].split(",") if item.strip()}

    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(strip_jsonc_comments(text))
    flat = flatten_config(payload)

    for key, value in sorted(flat.items()):
        if key in preset_keys:
            continue
        print(f"{key}={shlex.quote(to_shell_value(value))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
