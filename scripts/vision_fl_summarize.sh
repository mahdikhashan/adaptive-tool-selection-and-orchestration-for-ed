#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for d in results/vision/fl_*/; do
    uv run --locked python scripts/vision_summarize.py "$(basename "$d")" > /dev/null
    echo "summarized $(basename "$d")"
done
