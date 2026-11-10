#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
source scripts/server_env.sh

logs=logs/tools_v3
mkdir -p "$logs" results/tools_v3
progress="$logs/progress.txt"
only="${ONLY:-}"
python_run() { uv run --locked --extra gpu-tools --extra select python "$@"; }

stages=(
  "recut|scripts/tv3_recut.py"
  "train|scripts/tv3_train.py"
  "features|scripts/tv3_features.py"
  "yolo|scripts/tv3_yolo.py"
  "vlm|scripts/tv3_vlm.py"
  "assemble|scripts/tv3_assemble.py"
  "select_data|scripts/tv3_select_data.py"
  "sft_qwen|scripts/tv3_sft_qwen.py"
  "grpo_qwen|scripts/tv3_grpo_qwen.py"
  "eval|scripts/tv3_select_eval.py"
)

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$progress"; }

say "tools_v3 run started on $(hostname), commit $(git rev-parse --short HEAD), GPU ${CUDA_VISIBLE_DEVICES:-not set}"
python_run -c "import torch, trl, sklearn, ultralytics; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', 'trl', trl.__version__)" 2>&1 | tee -a "$progress"

total=${#stages[@]}
n=0
failed=()
for entry in "${stages[@]}"; do
  n=$((n + 1))
  name="${entry%%|*}"
  script="${entry#*|}"
  if [ -n "$only" ] && [[ " $only " != *" $name "* ]]; then
    continue
  fi
  if [ -f "$logs/$name.done" ] && [ -z "$only" ]; then
    say "stage $n/$total $name: already done, skipped"
    continue
  fi
  say "stage $n/$total $name: started (log: $logs/$name.log)"
  start=$(date +%s)
  if python_run "$script" > "$logs/$name.log" 2>&1; then
    touch "$logs/$name.done"
    say "stage $n/$total $name: done in $(( ($(date +%s) - start) / 60 )) min"
  else
    failed+=("$name")
    say "stage $n/$total $name: FAILED after $(( ($(date +%s) - start) / 60 )) min. Last lines:"
    tail -n 15 "$logs/$name.log" | tee -a "$progress"
  fi
done

say "run finished. Failed stages: ${failed[*]:-none}"
if [ -f results/tools_v3/value_report.json ]; then
  say "value report: results/tools_v3/value_report.json"
fi
if [ -f results/tools_v3/select/eval.json ]; then
  python3 - <<'EOF' | tee -a "$progress"
import json
r = json.load(open("results/tools_v3/select/eval.json"))
print("risk per 1,000 / mean time (s) / deferred share, test items:", r["items"])
for name, res in r["methods"].items():
    if "error" in res:
        print(f"  {name:14s} ERROR {res['error'][:120]}")
    else:
        print(f"  {name:14s} " + " | ".join(f"{b}s: {v['risk_per_1000']:6.1f} {v['mean_time_s']:.2f} {v['deferred']:.3f}" for b, v in res.items()))
EOF
fi
say "push the results: git add results/tools_v3 logs/tools_v3 && git commit -m 'tools_v3 results' && git push"
