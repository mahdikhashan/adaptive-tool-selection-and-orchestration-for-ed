#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/server_env.sh

model="${1:?usage: bash scripts/runs/tools_v3_selector.sh Qwen/Qwen3-1.7B | Qwen/Qwen3-4B}"
export TV3_SELECTOR="$model"
tag=$(basename "$model" | tr '[:upper:]' '[:lower:]')
case "$tag" in
  qwen3-1.7b) sft="--batch-size 2 --grad-accum 8"; grpo="--batch-size 8 --steps 400"; eval="--batch-size 12" ;;
  qwen3-4b)   sft="--batch-size 1 --grad-accum 16"; grpo="--batch-size 8 --steps 300"; eval="--batch-size 8" ;;
  *)          sft=""; grpo=""; eval="" ;;
esac
logs=logs/tools_v3/$tag
mkdir -p "$logs"
progress="$logs/progress.txt"
say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$progress"; }
python_run() { uv run --locked --extra gpu-tools --extra select python "$@"; }

for need in results/tools_v3/images.jsonl results/tools_v3/select/sft_qwen.jsonl results/tools_v3/select/grpo_qwen.jsonl; do
  [ -f "$need" ] || { say "missing $need: run scripts/runs/tools_v3_all.sh first"; exit 1; }
done

say "selector $model (tag $tag), commit $(git rev-parse --short HEAD)"
for entry in "sft|scripts/tv3_sft_qwen.py $sft" "grpo|scripts/tv3_grpo_qwen.py $grpo" "eval|scripts/tv3_select_eval.py base sft grpo $eval"; do
  name="${entry%%|*}"
  cmd="${entry#*|}"
  if [ -f "$logs/$name.done" ]; then
    say "$name: already done, skipped"
    continue
  fi
  say "$name: started (log: $logs/$name.log)"
  start=$(date +%s)
  if python_run $cmd > "$logs/$name.log" 2>&1; then
    touch "$logs/$name.done"
    say "$name: done in $(( ($(date +%s) - start) / 60 )) min"
  else
    say "$name: FAILED after $(( ($(date +%s) - start) / 60 )) min. Last lines:"
    tail -n 15 "$logs/$name.log" | tee -a "$progress"
  fi
done
say "finished. Results: results/tools_v3/select/eval.json (methods '$tag base', '$tag sft', '$tag grpo')"
