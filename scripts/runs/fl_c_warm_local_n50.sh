#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu=${1:?usage: bash scripts/runs/fl_c_warm_local_n50.sh <gpu number from nvidia-smi>}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
export TMPDIR="${RUN_TMP:-/dev/shm}/${USER}-fedorch-gpu$gpu"
mkdir -p "$TMPDIR" logs
rm -f logs/vision_fl_c_warm_local_n50.done
uv run --locked python scripts/vision_fl.py --name fl_c_warm_local_n50 --mode local --drones 5 --towers fleet --natural --budget 50 --local-passes 1 --share last_n --n-last 4 --val-images-held 1500 --start fl_server_central "${@:2}" 2>&1 | tee logs/vision_fl_c_warm_local_n50.log
touch logs/vision_fl_c_warm_local_n50.done
