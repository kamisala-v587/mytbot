# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""JSONC helpers for train/eval config files with // and block comments."""

from __future__ import annotations

import json
import os
import re
import tempfile
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


def load_jsonc(path: Path | str) -> dict[str, Any]:
    """Load a JSONC file into a dict."""
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    payload = json.loads(strip_jsonc_comments(text))
    if not isinstance(payload, dict):
        raise ValueError(f"JSONC root must be an object: {config_path}")
    return payload


def is_jsonc_path(path: Path | str) -> bool:
    return Path(path).suffix.lower() == ".jsonc"


def materialize_config_file(path: Path | str) -> Path:
    """Return a plain JSON path for draccus; materialize .jsonc to a temp file."""
    config_path = Path(path)
    if not is_jsonc_path(config_path):
        return config_path

    raw = load_jsonc(config_path)
    fd, tmp_name = tempfile.mkstemp(suffix=".json", prefix=f"{config_path.stem}_")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=4)
    return Path(tmp_name)
