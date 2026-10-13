#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${EX_PLUS_PYTHON:-python}" "$HERE/run.py" "${1:-$HERE/experiment.yml}" "${@:2}"
