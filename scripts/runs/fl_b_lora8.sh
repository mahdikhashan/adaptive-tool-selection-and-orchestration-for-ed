#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_b_lora8 --mode fedavg --drones 6 --share lora --lora-rank 8 "$@" 2>&1 | tee logs/vision_fl_b_lora8.log
touch logs/vision_fl_b_lora8.done
