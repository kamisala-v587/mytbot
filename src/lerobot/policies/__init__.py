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

from __future__ import annotations

import importlib
from typing import Any

_CONFIG_IMPORTS = {
    "QwenA1Config": ".InternVLA_A1_3B.configuration_internvla_a1",
    "InternA1Config": ".InternVLA_A1_2B.configuration_internvla_a1",
    "QwenActionConfig": ".qwenaction.configuration_qwenaction",
    "TBotSA1Config": ".TBot_SA1.configuration_tbot_sa1",
    "BPTBotConfig": ".BP_TBot.configuration_bp_tbot",
    "BPTBotV2Config": ".BP_TBot_v2.configuration_bp_tbot",
    "FastWAMConfig": ".fastwam.configuration_fastwam",
    "TBotSA1WanConfig": ".TBot_SA1_Wan.configuration_tbot_sa1_wan",
    "PI0Config": ".pi0.configuration_pi0",
    "PI05Config": ".pi05.configuration_pi05",
}

__all__ = list(_CONFIG_IMPORTS)


def __getattr__(name: str) -> Any:
    if name not in _CONFIG_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(_CONFIG_IMPORTS[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
