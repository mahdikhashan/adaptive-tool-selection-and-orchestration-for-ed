#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_laya.py --model base "$@" 2>&1 | tee logs/q4_laya_base.log
uv run --locked python scripts/vision_summarize.py q4_laya_base > /dev/null
touch logs/q4_laya_base.done
