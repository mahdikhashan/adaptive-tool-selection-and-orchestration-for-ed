#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_laya.py --model base --format int8 "$@" 2>&1 | tee logs/q_laya_base_int8.log
touch logs/q_laya_base_int8.done
