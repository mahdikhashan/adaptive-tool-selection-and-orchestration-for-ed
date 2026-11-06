#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
for f in scripts/runs/fl_*.sh; do
    name=$(basename "$f" .sh)
    if [ -f "logs/vision_$name.done" ]; then state="done"
    elif [ -f "logs/vision_$name.log" ]; then state="started: $(grep -h -E '^\{"round"|images, [0-9.]+ ms each|Error' "logs/vision_$name.log" | tail -1)"
    else state="not started"
    fi
    printf '%-24s %s\n' "$name" "$state"
done
