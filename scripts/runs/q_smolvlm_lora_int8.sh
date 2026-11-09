#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_smolvlm.py --model lora --format int8 "$@" 2>&1 | tee logs/q_smolvlm_lora_int8.log
touch logs/q_smolvlm_lora_int8.done
