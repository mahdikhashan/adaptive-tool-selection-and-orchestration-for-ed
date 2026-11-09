#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_laya.py --model base --format nf4 "$@" 2>&1 | tee logs/q_laya_base_nf4.log
uv run --locked python scripts/vision_summarize.py q_laya_base_nf4 > /dev/null
touch logs/q_laya_base_nf4.done
