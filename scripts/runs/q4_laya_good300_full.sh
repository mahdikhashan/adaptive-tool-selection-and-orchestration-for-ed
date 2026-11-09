#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_laya.py --model good300_full "$@" 2>&1 | tee logs/q4_laya_good300_full.log
uv run --locked python scripts/vision_summarize.py q4_laya_good300_full > /dev/null
touch logs/q4_laya_good300_full.done
