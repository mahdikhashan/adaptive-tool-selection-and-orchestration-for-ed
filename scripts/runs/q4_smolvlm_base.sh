#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_smolvlm.py --model base "$@" 2>&1 | tee logs/q4_smolvlm_base.log
touch logs/q4_smolvlm_base.done
