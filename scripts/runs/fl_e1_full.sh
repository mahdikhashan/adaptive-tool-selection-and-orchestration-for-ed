#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_e1_full --mode fedavg --drones 5 --share full --start fl_server_central --towers fleet --quantize int8 --val-images-held 1500 "$@" 2>&1 | tee logs/vision_fl_e1_full.log
touch logs/vision_fl_e1_full.done
