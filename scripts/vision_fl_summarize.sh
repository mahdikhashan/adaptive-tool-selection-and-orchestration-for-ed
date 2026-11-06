#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for d in results/vision/fl_*/; do
    [ -f "$d/run.json" ] || continue
    uv run --locked python scripts/vision_summarize.py "$(basename "$d")" > /dev/null
    echo "summarized $(basename "$d")"
done
for d in results/vision/fl_*_d0/; do
    name=$(basename "$d" _d0)
    uv run --locked python scripts/vision_fl_local_summary.py "$name" > /dev/null
    echo "combined $name"
done
