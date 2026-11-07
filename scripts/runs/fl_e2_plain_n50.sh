#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_e2_plain_n50 --mode fedavg --drones 5 --natural --local-passes 1 --share last_n --n-last 4 --budget 50 --start fl_server_central --towers fleet --quantize int8 --val-images-held 1500 "$@" 2>&1 | tee logs/vision_fl_e2_plain_n50.log
touch logs/vision_fl_e2_plain_n50.done
