#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_b_last4_r30 --mode fedavg --drones 6 --share last_n --n-last 4 --rounds 30 --passes 7.5 "$@" 2>&1 | tee logs/vision_fl_b_last4_r30.log
touch logs/vision_fl_b_last4_r30.done
