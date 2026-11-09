#!/usr/bin/env python3
"""Export measured frozen-model observations; --help never loads models."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedorch.ppo_v2_tools import DEFAULT_TOOLS, export_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--yolo-checkpoint', required=True,
                        help='Pretrained1000-class .pt as-is; optional separately trained canonical six-class .pt')
    parser.add_argument('--laya-checkpoint', required=True, help='Local frozen Laya-Vision checkpoint')
    parser.add_argument('--smol-base', help='Local SmolVLM-256M base directory')
    parser.add_argument('--smol-adapter', help='Local existing fine-tuned LoRA directory')
    parser.add_argument('--tools', default=','.join(DEFAULT_TOOLS), help='Comma-separated optional tool names')
    parser.add_argument('--enhance', action='store_true', help='Separate enhancement ablation views5..7, off by default')
    parser.add_argument('--device', default='cuda', help='cuda (FP32, compatible with 2080Ti) or cpu')
    parser.add_argument('--imgsz', type=int, default=224, help='Must match frozen YOLO training size')
    parser.add_argument('--smol-max-tokens', type=int, default=96)
    parser.add_argument('--splits', default='train,val,test',
                        help='Schedule train,val for early results; add test later with the same cache settings')
    cloud = parser.add_mutually_exclusive_group()
    cloud.add_argument('--cloud-local-checkpoint', help='Local Qwen3-VL-8B directory on this GPU VM')
    cloud.add_argument('--cloud-url', help='Optional exact OpenAI-compatible /v1/chat/completions endpoint')
    parser.add_argument('--cloud-precision', choices=['nf4', 'fp16'], default='nf4',
                        help='Local Qwen runtime only: NF4 with FP16 compute on2080Ti, FP16 requires >=22GiB GPU')
    parser.add_argument('--cloud-max-tokens', type=int, default=64)
    parser.add_argument('--cloud-views', choices=['original', 'all'], default='original',
                        help='original saves Qwen compute; absent cropped-view cloud actions remain masked')
    parser.add_argument('--cloud-model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--cloud-revision', default=None, help='Version/hash served remotely; required for cloud cache')
    parser.add_argument('--cloud-timeout', type=float, default=60.)
    parser.add_argument('--profile-limit', '--limit', dest='profile_limit', type=int, default=0,
                        help='Unique source images to profile; >0 marks cache incomplete and profile-only')
    args = parser.parse_args()
    if args.profile_limit < 0 or args.imgsz <= 0 or args.cloud_max_tokens < 1:
        parser.error('profile-limit must be nonnegative and imgsz positive')
    if 'CLOUD_QWEN8B' in args.tools and not args.cloud_local_checkpoint and not args.cloud_revision:
        parser.error('Cloud caching requires explicit --cloud-revision for the served checkpoint')
    print(json.dumps(export_cache(args), indent=2))


if __name__ == '__main__':
    main()
