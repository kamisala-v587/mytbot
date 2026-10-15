#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${BPV3_PROBE_PYTHON:-python}" "$HERE/probe.py" --config "${BPV3_PROBE_CONFIG:-$HERE/config.yaml}" "$@"
