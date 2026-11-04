#!/usr/bin/env bash
# One-time (idempotent) setup on the JKU AI-lab student servers.
#     cd /system/user/studentwork/$USER/fedorch && bash scripts/setup_server.sh
# Safe to re-run: every step skips work that is already done.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh

case "$PWD" in
    /system/user/studentwork/*) ;;
    *) echo "WARNING: repo is at $PWD, not under /system/user/studentwork (home is limited to 20 GB)." ;;
esac

# 1. uv (into studentwork/bin, not home)
if ! command -v uv >/dev/null 2>&1; then
    echo "== installing uv into $UV_INSTALL_DIR"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
uv --version

# 2. Python environment from the lockfile; the venv lives in the repo (on studentwork)
echo "== uv sync --locked"
uv sync --locked
uv run --locked python - <<'EOF'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(),
      "| built for:", torch.cuda.get_arch_list() if torch.cuda.is_available() else "-")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  visible GPU {i}: {p.name}, {p.total_memory/1e9:.0f} GB, compute capability {p.major}.{p.minor}")
EOF

# 3. Point the repo's data folders at studentwork
mkdir -p data
for link in raw:"$INSPLAD_DIR" external:"$FEDORCH_DATA/external"; do
    name="${link%%:*}"; target="${link#*:}"
    mkdir -p "$target"
    if [ ! -e "data/$name" ]; then ln -s "$target" "data/$name"; fi
done
ls -l data/

# 4. InsPLAD crops (6.4 GB download, only the two crop folders are kept)
if [ ! -d "$INSPLAD_DIR/supervised_fault_classification" ] || [ ! -d "$INSPLAD_DIR/unsupervised_anomaly_detection" ]; then
    echo "== fetching InsPLAD"
    uv run --locked python scripts/get_insplad.py --dest "$INSPLAD_DIR"
else
    echo "== InsPLAD crops already present in $INSPLAD_DIR"
fi

echo "== disk usage"
du -sh "$FEDORCH_WORK" 2>/dev/null || true
df -h "$FEDORCH_WORK"
echo "Setup done. Next: export CUDA_VISIBLE_DEVICES=<free booked GPU>; then run step 1 (see README)."
