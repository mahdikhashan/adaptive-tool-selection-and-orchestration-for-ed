import argparse
import json
import time

import torch

from fedorch import vision
from fedorch.common import results_dir, write_env, write_json
from fedorch.items import load_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = results_dir("vision/zeroshot")
    write_env("vision/zeroshot")
    items = load_items()
    if args.limit:
        items = items[: args.limit]
    agent = vision.load("cuda" if torch.cuda.is_available() else "cpu")
    start = time.time()
    probs = vision.predict(agent, items, args.batch_size, args.workers)
    seconds = time.time() - start
    with (out / "scores.jsonl").open("w") as file:
        for item, p in zip(items, probs):
            file.write(json.dumps({"item_id": item["item_id"], "probs": p}) + "\n")
    write_json(out / "run.json", dict(vars(args), checkpoint=vision.CHECKPOINT, revision=vision.REVISION,
                                      items=len(items), seconds=round(seconds, 1),
                                      ms_per_item=round(1000 * seconds / len(items), 2)))
    print(f"zeroshot: {len(items)} items in {seconds:.0f} s, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
