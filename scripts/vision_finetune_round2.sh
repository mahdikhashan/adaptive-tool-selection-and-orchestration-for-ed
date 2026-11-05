#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
for freeze in last_n full; do
    name="good300_${freeze/_/}"
    if [ -f "results/vision/${name}/run.json" ]; then
        echo "done    ${name}"
        continue
    fi
    uv run --locked python scripts/vision_finetune.py --name "$name" --good-per-component 300 --calib-images 300 \
        --freeze "$freeze" 2>&1 | tee -a "logs/vision_${name}.log"
done
ls results/vision
