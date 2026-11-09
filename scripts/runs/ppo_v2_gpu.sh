#!/usr/bin/env bash
# User-invoked GPU pipeline. This file does not start jobs on another machine.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server_env.sh

stage="${1:-help}"
if [ "$#" -gt 0 ]; then shift; fi
task_root="${PPO_V2_ROOT:-$FEDORCH_WORK/runs/ppo_v2}"
full_bundle="$task_root/full"
small_bundle="$task_root/small"
yolo_output="$task_root/yolo"
laya_path="${PPO_V2_LAYA:-$FEDORCH_WORK/models/ppo_v2/laya_base}"
smol_path="${PPO_V2_SMOL_BASE:-$FEDORCH_WORK/models/ppo_v2/smol_base}"
smol_adapter="${PPO_V2_SMOL_ADAPTER:-$FEDORCH_WORK/runs/step2/smolvlm_lora_full}"
run_python() { uv run --locked --extra gpu-tools python "$@"; }
require_gpu() {
  run_python -c 'import torch; assert torch.cuda.is_available(), "A CUDA GPU is required; refusing CPU training"; print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
}
cache_args=(--bundle "$small_bundle" --yolo-checkpoint "$yolo_output/frozen_yolo.pt"
  --laya-checkpoint "$laya_path" --smol-base "$smol_path" --smol-adapter "$smol_adapter"
  --device cuda --smol-max-tokens "${PPO_V2_SMOL_TOKENS:-96}"
  --tools YOLO_CLS,SMOL_EXPLAIN,ROI_PROPOSE,ZOOM_ROI,IMAGE_QUALITY)

case "$stage" in
  prepare)
    mkdir -p "$task_root"
    run_python scripts/ppo_v2_prepare.py --output "$full_bundle" --fraction 1 --seed 0
    run_python scripts/ppo_v2_prepare.py --output "$small_bundle" --fraction 0.1 --seed 0
    ;;
  yolo)
    require_gpu
    run_python scripts/ppo_v2_yolo.py --bundle "$full_bundle" --output "$yolo_output" \
      --base "${PPO_V2_YOLO_BASE:-$FEDORCH_WORK/models/ppo_v2/yolo26n-cls.pt}" \
      --epochs 20 --batch 64 --imgsz 224 --device 0 "$@"
    ;;
  profile)
    require_gpu
    run_python scripts/ppo_v2_cache.py "${cache_args[@]}" --splits train,val --profile-limit 20 "$@"
    ;;
  cache)
    require_gpu
    run_python scripts/ppo_v2_cache.py "${cache_args[@]}" --splits train,val "$@"
    ;;
  cache-test)
    require_gpu
    run_python scripts/ppo_v2_cache.py "${cache_args[@]}" --splits test "$@"
    ;;
  pilot)
    require_gpu
    run_python scripts/ppo_v2_compare.py --bundle "$small_bundle" \
      --output "${PPO_V2_OUTPUT:-$task_root/pilot_seed0}" --device cuda \
      --feedback audit --seed 0 --rounds 3 --episodes-per-client 64 \
      --eval-split val --budgets 2,6,15 "$@"
    ;;
  *)
    echo "Usage: bash scripts/runs/ppo_v2_gpu.sh prepare|yolo|profile|cache|pilot|cache-test [extra CLI flags]"
    echo "Set CUDA_VISIBLE_DEVICES to your booked 2080 Ti. Defaults: one YOLO, seed0, 10% train, 5% audits."
    echo "Existing local checkpoints required. See docs/ppo_v2_gpu.md for downloads and commands."
    echo "Cloud is excluded from this edge-only pilot; add its endpoint explicitly for a fresh cloud-enabled bundle."
    ;;
esac
