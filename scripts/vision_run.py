import argparse
import json
import random
import time

import torch

from fedorch import vision
from fedorch.common import results_dir, write_env, write_json
from fedorch.items import load_items

CONDITION_ONLY = {"condition": vision.QUESTIONS["condition"]}


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def peak_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0


def timing(agent, rows, frames):
    sample = random.Random(0).sample(rows, min(len(rows), frames + 10))
    return {"all_questions_ms": vision.latency(agent, sample, vision.QUESTIONS),
            "condition_only_ms": vision.latency(agent, sample, CONDITION_ONLY)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latency-frames", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()
    rows = vision.images(load_items())
    if args.limit:
        rows = rows[: args.limit]
    test = [r for r in rows if r["split"] == "test"] or rows
    agent = vision.load(args.device)
    if args.latency_only:
        write_json(results_dir("vision") / f"latency_{args.device}.json",
                   dict(device=args.device, gpu=gpu_name(), precision="fp32", **timing(agent, test, args.latency_frames)))
        return
    out = results_dir("vision/zeroshot")
    write_env("vision/zeroshot")
    path = out / "predictions.jsonl"
    done = {json.loads(line)["image"] for line in path.open()} if path.exists() else set()
    todo = [r for r in rows if r["image"] not in done]
    print(f"{len(rows)} images, {len(done)} done, {len(todo)} to do", flush=True)
    for row in rows[:10]:
        vision.predict(agent, row["image"])
    start = time.time()
    with path.open("a") as file:
        for n, row in enumerate(todo, 1):
            answers, ms, truncated = vision.predict(agent, row["image"])
            file.write(json.dumps({"image": row["image"], "answers": answers, "ms": round(ms, 2),
                                   "truncated": truncated}, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                file.flush()
                print(f"{n}/{len(todo)} images, {(time.time() - start) / n * 1000:.1f} ms each", flush=True)
    write_json(out / "run.json", dict(vars(args), checkpoint=vision.CHECKPOINT, revision=vision.REVISION,
                                      images=len(rows), questions=vision.QUESTIONS, device=str(agent.device),
                                      gpu=gpu_name(), precision="fp32", peak_gpu_mb=peak_mb(),
                                      **timing(agent, test, args.latency_frames)))
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
