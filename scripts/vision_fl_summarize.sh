#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for d in results/vision/fl_*/; do
    [ -f "$d/run.json" ] || continue
    uv run --locked python scripts/vision_summarize.py "$(basename "$d")" > /dev/null
    echo "summarized $(basename "$d")"
done
uv run --locked python scripts/vision_fl_local_summary.py fl_local_k6
