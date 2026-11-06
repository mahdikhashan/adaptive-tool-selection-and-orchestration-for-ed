#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_c_base_n25 --mode fedavg --drones 5 --towers fleet --natural --budget 25 --local-passes 1 --share last_n --n-last 4 --val-images-held 1500 "$@" 2>&1 | tee logs/vision_fl_c_base_n25.log
touch logs/vision_fl_c_base_n25.done
