#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server_env.sh
export OMP_NUM_THREADS=1

jobs="${JOBS:-8}"
seeds="${SEEDS:-0 1 2 3 4}"
rounds="${ROUNDS:-60}"
out=results/route/mission_warm
mkdir -p "$out" logs/mission_warm
uv run --locked --with scikit-learn python -c "import sklearn, torch; print('ok', torch.__version__)"

common="--budgets .1,.15,.2,.3 --rounds $rounds --eval-every 10 --segments 256 --gamma .98 --gae .9 --lr 1e-4 --warm --warm-segments 2000 --fed"
run_one() {
  local name=$1; shift
  if uv run --locked --with scikit-learn python scripts/route_mission.py $common "$@" > "logs/mission_warm/$name.log" 2>&1; then
    echo "done: $name"
  else
    echo "FAILED: $name (see logs/mission_warm/$name.log)"
  fi
}

for cap in .05 .025; do
  for variant in "warm:0" "anchor:.5"; do
    for seed in $seeds; do
      name="${variant%%:*}_cap${cap#.}_s$seed"
      while [ "$(jobs -rp | wc -l)" -ge "$jobs" ]; do sleep 5; done
      run_one "$name" --cap "$cap" --anchor "${variant#*:}" --seeds "$seed" --tag "_warm/$name" &
    done
  done
done
wait

git rev-parse HEAD > "$out/commit.txt"
python3 scripts/show_mission.py mission_warm | tee "$out/summary.txt"
