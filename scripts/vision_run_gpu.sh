#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/vision logs

run() {
    local name=$1; shift
    if [ -f "results/vision/${name}/train.json" ]; then
        echo "done    ${name}"
        return
    fi
    echo "running ${name}"
    uv run --locked python scripts/vision_run.py "$@" > "logs/vision_${name}.log" 2>&1
    echo "done    ${name}"
}

run zeroshot_s0 --mode zeroshot
run central_s0 --mode central
for k in 3 6 18; do
    run "fedavg_k${k}_s0" --mode fedavg --drones "$k"
done
run local_k6_s0 --mode local --drones 6

ls results/vision
