#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_smolvlm.py --model base --format nf4 "$@" 2>&1 | tee logs/q_smolvlm_base_nf4.log
touch logs/q_smolvlm_base_nf4.done
