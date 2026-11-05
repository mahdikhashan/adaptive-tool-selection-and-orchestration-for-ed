import argparse
import json
import random
import time

import torch

from fedorch import vision
from fedorch.common import results_dir, write_env, write_json
from fedorch.items import load_items


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def peak_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latency-frames", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()
    out = results_dir("vision/zeroshot")
    write_env("vision/zeroshot")
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    agent = vision.load(args.device)
    test = [i for i in items if i["split"] == "test"] or items
    timing = vision.latency(agent, random.Random(0).sample(test, min(len(test), args.latency_frames + 10)))
    print("latency", json.dumps(timing), flush=True)
    if args.latency_only:
        write_json(results_dir("vision") / f"latency_{args.device}.json",
                   dict(device=args.device, gpu=gpu_name(), precision="fp32", latency=timing))
        return
    start = time.time()
    probs = vision.predict(agent, items, args.batch_size, args.workers)
    seconds = time.time() - start
    with (out / "scores.jsonl").open("w") as file:
        for item, p in zip(items, probs):
            file.write(json.dumps({"item_id": item["item_id"], "probs": p}) + "\n")
    write_json(out / "run.json", dict(vars(args), checkpoint=vision.CHECKPOINT, revision=vision.REVISION,
                                      items=len(items), seconds=round(seconds, 1),
                                      batch_ms_per_item=round(1000 * seconds / len(items), 2),
                                      device=str(agent.device), gpu=gpu_name(), precision="fp32",
                                      peak_gpu_mb=peak_mb(), latency=timing))
    print(f"zeroshot: {len(items)} items in {seconds:.0f} s, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
