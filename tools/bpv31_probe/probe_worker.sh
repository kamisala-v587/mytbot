#!/usr/bin/env bash
# Usage: probe_worker.sh <config.yaml> <checkpoint name> <gpu>
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUT=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$1")
mkdir -p "$OUT/logs"
CUDA_VISIBLE_DEVICES="$3" /vla/workspace/shiliangchao/env/q35va/bin/python3.10 "$HERE/probe_v31.py" \
  --config "$1" --worker "$2" > "$OUT/logs/$2.log" 2>&1
