#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
for f in scripts/runs/fl_*.sh; do
    name=$(basename "$f" .sh)
    if pgrep -f -- "--name $name( |$)" >/dev/null; then state="running"
    elif [ -f results/vision/"$name"/run.json ] || [ -f results/vision/"$name"_d0/run.json ]; then state="done"
    else state="not started"
    fi
    printf '%-24s %s\n' "$name" "$state"
done
for f in scripts/runs/q_*.sh; do
    name=$(basename "$f" .sh)
    if pgrep -f -- "logs/$name.log" >/dev/null; then state="running: $(grep -h -E 'images, [0-9.]+ ms each|items, [0-9.]+ ms|Error' "logs/$name.log" | tail -1)"
    elif [ -f results/vision/"$name"/run.json ] || [ -f results/quant/"$name"/summary.json ]; then state="done"
    else state="not started"
    fi
    printf '%-24s %s\n' "$name" "$state"
done
