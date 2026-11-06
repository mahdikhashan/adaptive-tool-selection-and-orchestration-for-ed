#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
gpu=${1:?usage: bash scripts/runs/fl_b_head.sh <gpu number from nvidia-smi>}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
export TMPDIR="${RUN_TMP:-/dev/shm}/${USER}-fedorch-gpu$gpu"
mkdir -p "$TMPDIR" logs
rm -f logs/vision_fl_b_head.done
uv run --locked python scripts/vision_fl.py --name fl_b_head --mode fedavg --drones 6 --share head "${@:2}" 2>&1 | tee logs/vision_fl_b_head.log
touch logs/vision_fl_b_head.done
