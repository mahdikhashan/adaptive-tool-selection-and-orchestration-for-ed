#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
uv run --locked python scripts/vision_run.py > logs/vision_zeroshot.log 2>&1
uv run --locked python scripts/vision_run.py --device cpu --latency-only --latency-frames 100 > logs/vision_latency_cpu.log 2>&1
grep -h "latency\|zeroshot:" logs/vision_zeroshot.log logs/vision_latency_cpu.log
