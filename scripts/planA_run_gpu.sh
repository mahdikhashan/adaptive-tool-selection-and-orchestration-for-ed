#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/planA logs

run() {
    local name=$1; shift
    if [ -f "results/planA/p_${name}.jsonl" ]; then
        echo "done    ${name}"
        return
    fi
    echo "running ${name}"
    uv run --locked python "$@" --temperature 1.0 --oversample 5 --step planA > "logs/planA_${name}.log" 2>&1
    echo "done    ${name}"
}

for f in 0.01 0.05 0.2 1.0; do
    for s in 0 1 2; do
        [ "$f" = "1.0" ] && [ "$s" = "0" ] && continue
        run "nanojev_T1.0_O5_f${f}_s${s}" scripts/step4b_train_nanojev.py --fraction "$f" --seed "$s"
    done
done

for f in 0.05 0.2; do
    run "laya_T1.0_O5_f${f}_s0" scripts/step4b_train_laya.py --fraction "$f" --seed 0
done

ls results/planA
