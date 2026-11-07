import argparse

import torch

from fedorch import vision
from fedorch.common import WORK, results_dir, write_env, write_json
from fedorch.items import load_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    vision.fp32_on_old_gpus()
    rows = vision.images(load_items())
    if args.limit:
        rows = rows[: args.limit]
    path = WORK / "models" / "vision" / args.model
    agent = vision.load(args.device, path)
    out = results_dir(f"vision_export/{args.model}")
    write_env(f"vision_export/{args.model}")
    vision.predict_all(agent, rows, out / "predictions.jsonl")
    write_json(out / "run.json", dict(vars(args), model_dir=str(path), images=len(rows), questions=vision.QUESTIONS,
                                      temperature=agent.temperature, gpu=vision.gpu_name()))
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
