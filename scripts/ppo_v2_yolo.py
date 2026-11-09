#!/usr/bin/env python3
"""Train ONE six-class classifier before freezing it for all controller methods.

This full-label tool pretraining is separate from five-percent controller audits.
No augmentation, random crops, test fitting, TensorRT, or quantization.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fedorch.ppo_v2_tools import train_yolo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, help='Full training-items bundle, not 10%% controller subsample')
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', default='yolo26n-cls.pt', help='Existing local ImageNet initialization .pt')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--imgsz', type=int, default=224)
    parser.add_argument('--device', default='0')
    parser.add_argument('--workers', type=int, default=0, help='0 permits custom deterministic dataset classes without spawn pickling')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--prepare-only', action='store_true', help='Prepare split-safe original-image dataset without loading models')
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.imgsz) < 1 or args.workers != 0:
        parser.error('epochs/batch/imgsz must be positive; this custom dataset requires --workers 0')
    print(json.dumps(train_yolo(args), indent=2))


if __name__ == '__main__':
    main()
