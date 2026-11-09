#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_laya.py --model good300_full --format fp16 "$@" 2>&1 | tee logs/q_laya_good300_full_fp16.log
uv run --locked python scripts/vision_summarize.py q_laya_good300_full_fp16 > /dev/null
touch logs/q_laya_good300_full_fp16.done
