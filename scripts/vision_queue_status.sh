#!/usr/bin/env bash
cd "$(dirname "$0")/.."
queue=results/vision/.queue
grep -v '^#' scripts/vision_jobs.txt | cut -d'|' -f1 | while read -r name; do
    if [ -f "$queue/$name.done" ]; then state="done"
    elif [ -f "$queue/$name.claim/failed" ]; then state="FAILED"
    elif [ -d "$queue/$name.claim" ]; then state="running: $(grep -h -E '^\{"round"|images, [0-9.]+ ms each' "logs/vision_$name.log" 2>/dev/null | tail -1)"
    else state="waiting"
    fi
    printf '%-24s %s\n' "$name" "$state"
done
