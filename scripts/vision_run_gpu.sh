#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run --locked python scripts/vision_run.py 2>&1 | tee -a logs/vision_zeroshot.log
uv run --locked python scripts/vision_run.py --device cpu --latency-only --latency-frames 100 2>&1 | tee -a logs/vision_latency_cpu.log
ls -la results/vision results/vision/zeroshot
