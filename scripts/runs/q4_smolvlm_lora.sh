#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_smolvlm.py --model lora "$@" 2>&1 | tee logs/q4_smolvlm_lora.log
touch logs/q4_smolvlm_lora.done
