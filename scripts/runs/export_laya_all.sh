#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
for m in fl_server_central fl_e1_head fl_e1_full; do
  uv run --locked python scripts/export_laya_all.py --model "$m" "$@" 2>&1 | tee "logs/export_$m.log"
  touch "logs/export_$m.done"
done
