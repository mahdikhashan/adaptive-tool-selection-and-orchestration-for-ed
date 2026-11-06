#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_b_full_prox --mode fedavg --drones 6 --share full --prox 0.01 "$@" 2>&1 | tee logs/vision_fl_b_full_prox.log
touch logs/vision_fl_b_full_prox.done
