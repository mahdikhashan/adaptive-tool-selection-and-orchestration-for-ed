import argparse
import json
import math
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
    parser.add_argument("--model", required=True, choices=["base", "good300_full"])
    parser.add_argument("--val-images", type=int, default=1500)
    parser.add_argument("--latency-frames", type=int, default=100)
    return parser.parse_args()


def build(model, out):
    agent = vision.load("cuda", None if model == "base" else WORK / "models" / "vision" / model)
    quant.convert(agent.model.encoder, AutoModel, out / "backbone_quantized")
    agent.save(str(out), include_backbone=False)
    del agent
    torch.cuda.empty_cache()


def load(out):
    agent = load_vlm(str(out), device="cuda", dtype="fp32")
    agent.model.encoder = quant.load(AutoModel, out / "backbone_quantized", attn_implementation="sdpa")
    agent._torch_dtype = lambda: torch.float16
    torch.cuda.empty_cache()
    return agent


def head_dtypes(agent):
    return sorted({str(t.dtype).replace("torch.", "") for n, t in list(agent.model.named_parameters()) + list(agent.model.named_buffers())
                   if not n.startswith("encoder.") and t.is_floating_point()})


def non_finite(path):
    rows = [json.loads(line) for line in path.open()]
    return sum(not math.isfinite(r["answers"]["defect"]) or any(not math.isfinite(v) for v in r["answers"]["condition"].values())
               for r in rows)


def main():
    args = arguments()
    name = f"q4_laya_{args.model}"
    out = quant.STORE / f"laya_{args.model}_nf4_fp16"
    vision.fp32_on_old_gpus()
    if not (out / "backbone_quantized").exists():
        build(args.model, out)
    agent = load(out)
    loaded = quant.allocated_mb()
    torch.cuda.reset_peak_memory_stats()
    rows = vision.images(load_items(("val", "test")))
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    held = random.Random(0).sample(val, min(args.val_images, len(val))) + test
    print(f"{name}: {quant.quantized_layers(agent.model)} 4-bit layers, backbone {quant.float_dtypes(agent.model.encoder)}, "
          f"head {head_dtypes(agent)}, "
          f"weights {quant.weight_mb(agent.model)} MB, stored {quant.folder_mb(out)} MB, GPU after load {loaded} MB", flush=True)
    res = results_dir(f"vision/{name}")
    write_env(f"vision/{name}")
    start = time.time()
    vision.predict_all(agent, held, res / "predictions.jsonl")
    write_json(res / "run.json", {"run": name, "family": "laya-vision", "model": args.model, "format": "nf4 weights, fp16 compute",
                                  "stored_at": str(out), "stored_mb": quant.folder_mb(out), "weight_mb": quant.weight_mb(agent.model),
                                  "gpu_mb_after_load": loaded, "four_bit_layers": quant.quantized_layers(agent.model),
                                  "backbone_float_dtypes": quant.float_dtypes(agent.model.encoder),
                                  "backbone_weight_mb": quant.weight_mb(agent.model.encoder), "head_float_dtypes": head_dtypes(agent),
                                  "head_note": "laya casts the decision head input to fp32 in _readout", "images": len(held),
                                  "non_finite": non_finite(res / "predictions.jsonl"), "seconds": round(time.time() - start, 1),
                                  "device": "cuda", "gpu": vision.gpu_name(), "precision": "nf4/fp16",
                                  **vision.timing(agent, test, args.latency_frames), "peak_gpu_mb": vision.peak_mb()})
    print(f"{name}: done, wrote {res}", flush=True)


if __name__ == "__main__":
    main()
