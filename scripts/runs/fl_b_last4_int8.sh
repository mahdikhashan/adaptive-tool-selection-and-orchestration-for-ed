#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_b_last4_int8 --mode fedavg --drones 6 --share last_n --n-last 4 --quantize int8 "$@" 2>&1 | tee logs/vision_fl_b_last4_int8.log
touch logs/vision_fl_b_last4_int8.done
