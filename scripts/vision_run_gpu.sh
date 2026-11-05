#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run --locked python scripts/vision_run.py > logs/vision_zeroshot.log 2>&1
tail -1 logs/vision_zeroshot.log
