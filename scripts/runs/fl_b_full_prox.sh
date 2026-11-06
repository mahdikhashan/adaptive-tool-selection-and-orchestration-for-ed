#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu=${1:?usage: bash scripts/runs/fl_b_full_prox.sh <gpu number from nvidia-smi>}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
export TMPDIR="${RUN_TMP:-/dev/shm}/${USER}-fedorch-gpu$gpu"
mkdir -p "$TMPDIR" logs
rm -f logs/vision_fl_b_full_prox.done
uv run --locked python scripts/vision_fl.py --name fl_b_full_prox --mode fedavg --drones 6 --share full --prox 0.01 "${@:2}" 2>&1 | tee logs/vision_fl_b_full_prox.log
touch logs/vision_fl_b_full_prox.done
