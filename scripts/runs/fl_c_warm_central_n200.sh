#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_c_warm_central_n200 --mode central --drones 5 --towers fleet --natural --budget 200 --local-passes 1 --share last_n --n-last 4 --val-images-held 1500 --start fl_server_central "$@" 2>&1 | tee logs/vision_fl_c_warm_central_n200.log
touch logs/vision_fl_c_warm_central_n200.done
