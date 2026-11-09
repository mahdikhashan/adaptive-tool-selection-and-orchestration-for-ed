import argparse
import random
import time

import torch
from laya.vlm import load_vlm
from transformers import AutoModel

from fedorch import quant, vision
from fedorch.common import WORK, results_dir, write_env, write_json
from fedorch.items import load_items


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--format", required=True, choices=quant.FORMATS)
    parser.add_argument("--val-images", type=int, default=1500)
    parser.add_argument("--latency-frames", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def build(model, fmt, out):
    agent = vision.load("cuda", None if model == "base" else WORK / "models" / "vision" / model)
    quant.convert(agent.model.encoder, AutoModel, fmt, out / "backbone_quantized")
    agent.save(str(out), include_backbone=False)
    del agent
    torch.cuda.empty_cache()


def load(out, fmt):
    agent = load_vlm(str(out), device="cuda", dtype="fp32")
    agent.model.encoder = quant.load(AutoModel, out / "backbone_quantized", fmt, attn_implementation="sdpa")
    if fmt == "fp16":
        agent.model.half()
        agent._torch_dtype = lambda: torch.float16
    torch.cuda.empty_cache()
    return agent


def main():
    args = arguments()
    name = f"q_laya_{args.model}_{args.format}"
    out = quant.STORE / f"laya_{args.model}_{args.format}"
    vision.fp32_on_old_gpus()
    if not (out / "backbone_quantized").exists():
        build(args.model, args.format, out)
    agent = load(out, args.format)
    torch.cuda.reset_peak_memory_stats()
    rows = vision.images(load_items(("val", "test")))
    val = [r for r in rows if r["split"] == "val"]
    held = random.Random(0).sample(val, min(args.val_images, len(val))) + [r for r in rows if r["split"] == "test"]
    if args.limit:
        held = held[: args.limit // 2] + held[-(args.limit // 2):]
    print(f"{name}: {quant.quantized_layers(agent.model)} bitsandbytes layers, weights {quant.weight_mb(agent.model)} MB, "
          f"stored {quant.folder_mb(out)} MB at {out}", flush=True)
    res = results_dir(f"vision/{name}")
    write_env(f"vision/{name}")
    start = time.time()
    vision.predict_all(agent, held, res / "predictions.jsonl")
    write_json(res / "run.json", {"run": name, "model": args.model, "format": args.format, "stored_at": str(out),
                                  "stored_mb": quant.folder_mb(out), "weight_mb": quant.weight_mb(agent.model),
                                  "bitsandbytes_layers": quant.quantized_layers(agent.model), "images": len(held),
                                  "seconds": round(time.time() - start, 1), "device": "cuda", "gpu": vision.gpu_name(),
                                  "precision": args.format, "peak_gpu_mb": vision.peak_mb(),
                                  **vision.timing(agent, [r for r in held if r["split"] == "test"] or held, args.latency_frames)})
    print(f"{name}: done, wrote {res}", flush=True)


if __name__ == "__main__":
    main()
