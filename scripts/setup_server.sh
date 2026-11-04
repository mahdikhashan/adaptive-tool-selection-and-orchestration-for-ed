#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/server_env.sh

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
git lfs install --local || echo "git-lfs is not available. Do not commit result files larger than 50 MB."
uv sync --locked
uv run --locked python scripts/get_insplad.py
df -h "$FEDORCH_WORK"
