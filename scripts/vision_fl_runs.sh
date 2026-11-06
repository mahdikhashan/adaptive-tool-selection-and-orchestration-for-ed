#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

run() {
    local name=$1; shift
    if [ -f "results/vision/${name}/run.json" ] || [ -f "results/vision/${name}_d0/run.json" ]; then
        echo "done    ${name}"
        return
    fi
    echo "running ${name}"
    uv run --locked python scripts/vision_fl.py --name "$name" "$@" 2>&1 | tee -a "logs/vision_${name}.log"
}

run fl_central_k1 --mode central
run fl_fedavg_k6 --mode fedavg --drones 6
run fl_fedavg_k18 --mode fedavg --drones 18
run fl_fedavg_k3 --mode fedavg --drones 3
run fl_server_central --mode central --towers server
run fl_fleet_base_k5 --mode fedavg --drones 5 --towers fleet
run fl_fleet_warm_k5 --mode fedavg --drones 5 --towers fleet --start fl_server_central
run fl_local_k6 --mode local --drones 6
ls results/vision
