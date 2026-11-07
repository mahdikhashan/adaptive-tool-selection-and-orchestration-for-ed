#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_e3_iid --mode fedavg --drones 6 --share last_n --n-last 4 --partition iid --start fl_server_central --towers fleet --quantize int8 --val-images-held 1500 "$@" 2>&1 | tee logs/vision_fl_e3_iid.log
touch logs/vision_fl_e3_iid.done
