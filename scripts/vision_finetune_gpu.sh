#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run --locked python scripts/vision_finetune.py "$@" 2>&1 | tee -a logs/vision_finetune.log
