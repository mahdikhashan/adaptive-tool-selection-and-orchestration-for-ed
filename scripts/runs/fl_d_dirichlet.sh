#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/vision_fl.py --name fl_d_dirichlet --mode fedavg --drones 6 --partition dirichlet --alpha 0.3 --share last_n --n-last 4 "$@" 2>&1 | tee logs/vision_fl_d_dirichlet.log
touch logs/vision_fl_d_dirichlet.done
