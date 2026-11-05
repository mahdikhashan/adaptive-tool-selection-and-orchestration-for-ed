import argparse

import torch

from fedorch import vision
from fedorch.common import results_dir, write_env, write_json
from fedorch.items import load_items


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
                   dict(device=args.device, gpu=vision.gpu_name(), precision="fp32", **vision.timing(agent, test, args.latency_frames)))
        return
    out = results_dir("vision/zeroshot")
    write_env("vision/zeroshot")
    vision.predict_all(agent, rows, out / "predictions.jsonl")
    write_json(out / "run.json", dict(vars(args), checkpoint=vision.CHECKPOINT, revision=vision.REVISION,
                                      images=len(rows), questions=vision.QUESTIONS, device=str(agent.device),
                                      gpu=vision.gpu_name(), precision="fp32", peak_gpu_mb=vision.peak_mb(),
                                      **vision.timing(agent, test, args.latency_frames)))
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
