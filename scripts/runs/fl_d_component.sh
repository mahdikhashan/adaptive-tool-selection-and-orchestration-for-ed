#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_d_component --mode fedavg --drones 6 --partition component --share last_n --n-last 4 "$@" 2>&1 | tee logs/vision_fl_d_component.log
touch logs/vision_fl_d_component.done
