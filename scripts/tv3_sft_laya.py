import argparse
import json
import math
import time

import torch
from laya.vlm_train import collect_logits, fit_temperatures_from, load_jsonl_examples, train

from fedorch import vision
from fedorch.common import WORK, results_dir
from fedorch.tools_v3 import laya_checkpoint, models_dir


def main():
    parser = argparse.ArgumentParser(description="Supervised fine-tuning of laya-vision as tool selector with the laya library trainer.")
    parser.add_argument("--start", default=str(laya_checkpoint()))
    parser.add_argument("--max-examples", type=int, default=12000)
    parser.add_argument("--passes", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze", default="last_n")
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=str(models_dir() / "laya_select"))
    args = parser.parse_args()
    root = str(WORK / "data" / "tools_v3_laya")
    examples = load_jsonl_examples(root, "select", "train", limit=args.max_examples)
    calib = load_jsonl_examples(root, "select", "calib", limit=1500)
    vision.fp32_on_old_gpus()
    run_on = "cuda" if torch.cuda.is_available() else "cpu"
    agent = vision.load(run_on, args.start) if args.start else vision.load(run_on)
    steps = max(1, math.ceil(args.passes * len(examples) / args.batch_size))
    print(f"[laya sft] {len(examples)} examples, {steps} steps, start {args.start}", flush=True)
    start, stats = time.time(), {}
    losses = train(agent.model, agent.processor, examples, steps=steps, batch_size=args.batch_size, freeze=args.freeze,
                   lr_head=args.lr_head, lr_backbone=args.lr_backbone, warmup=25, device=str(agent.device), seed=0,
                   log_every=50, stats=stats, num_workers=args.workers)
    agent.temperature = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib, batch_size=16))
    agent.save(args.out)
    (results_dir("tools_v3") / "select" / "laya_sft_train.json").write_text(json.dumps(
        {"examples": len(examples), "steps": steps, "minutes": round((time.time() - start) / 60, 1), "temperature": agent.temperature,
         "loss_every_50": [round(sum(losses[i:i + 50]) / len(losses[i:i + 50]), 4) for i in range(0, len(losses), 50)]}, indent=1))
    print(f"[laya sft] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
