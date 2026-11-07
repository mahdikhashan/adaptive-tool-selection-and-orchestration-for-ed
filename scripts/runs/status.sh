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
