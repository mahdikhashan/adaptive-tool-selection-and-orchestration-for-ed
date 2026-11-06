#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
gpu=$1
jobs=${2:-scripts/vision_jobs.txt}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu
export TMPDIR="${QUEUE_TMP:-/dev/shm}/${USER}-fedorch-gpu$gpu"
queue=results/vision/.queue
mkdir -p "$TMPDIR" logs "$queue"
while IFS='|' read -r name args; do
    case "$name" in ""|\#*) continue ;; esac
    [ -f "$queue/$name.done" ] && continue
    mkdir "$queue/$name.claim" 2>/dev/null || continue
    echo "$(date +%H:%M) gpu $gpu: start $name"
    if uv run --locked python scripts/vision_fl.py --name "$name" $args < /dev/null > "logs/vision_$name.log" 2>&1; then
        touch "$queue/$name.done"
        echo "$(date +%H:%M) gpu $gpu: done  $name"
    else
        echo "gpu $gpu" > "$queue/$name.claim/failed"
        echo "$(date +%H:%M) gpu $gpu: FAILED $name, see logs/vision_$name.log"
    fi
done < "$jobs"
echo "$(date +%H:%M) gpu $gpu: no more jobs"
