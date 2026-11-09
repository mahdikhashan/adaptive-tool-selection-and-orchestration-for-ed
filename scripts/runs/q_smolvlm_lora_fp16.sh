#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
uv run --locked python scripts/quant_smolvlm.py --model lora --format fp16 "$@" 2>&1 | tee logs/q_smolvlm_lora_fp16.log
touch logs/q_smolvlm_lora_fp16.done
