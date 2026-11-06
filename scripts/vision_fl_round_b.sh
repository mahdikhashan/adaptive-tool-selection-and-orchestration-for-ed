#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
export TMPDIR="/dev/shm/${USER}-fedorch"
mkdir -p "$TMPDIR"

run() {
    local name=$1; shift
    if [ -f "results/vision/${name}/run.json" ]; then
        echo "done    ${name}"
        return
    fi
    echo "running ${name}"
    uv run --locked python scripts/vision_fl.py --name "$name" --mode fedavg --drones 6 "$@" 2>&1 | tee -a "logs/vision_${name}.log"
}

run fl_b_head --share head
run fl_b_lora8 --share lora --lora-rank 8
run fl_b_last4 --share last_n --n-last 4
run fl_b_last4_int8 --share last_n --n-last 4 --quantize int8
run fl_b_last4_prox --share last_n --n-last 4 --prox 0.01
run fl_b_layers --share layers
run fl_b_full_prox --share full --prox 0.01
run fl_b_last4_r30 --share last_n --n-last 4 --rounds 30 --passes 7.5
ls results/vision | grep fl_b_
