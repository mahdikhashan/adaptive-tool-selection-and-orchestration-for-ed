import argparse
import json
import math
import random
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from fedorch import quant, vision
from fedorch.common import results_dir, runs_dir, write_env, write_json
from fedorch.items import image_path, load_items
from fedorch.vlm import MAX_SIDE, LetterScorer, load_image

SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
ADAPTER = runs_dir("step2") / "smolvlm_lora_full"
MISS, FALSE_ALARM = 10, 1


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["base", "lora"])
    parser.add_argument("--format", required=True, choices=quant.FORMATS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sample", type=int, default=150)
    return parser.parse_args()


def build(model, fmt, out):
    fp32 = AutoModelForImageTextToText.from_pretrained(SMOLVLM, dtype=torch.float32)
    if model == "lora":
        from peft import PeftModel
        if not ADAPTER.exists():
            raise SystemExit(f"missing {ADAPTER}: run 'uv run --locked python scripts/step2_train_smolvlm.py full' first")
        fp32 = PeftModel.from_pretrained(fp32, str(ADAPTER)).merge_and_unload()
    processor = AutoProcessor.from_pretrained(SMOLVLM)
    processor.image_processor.size = {"longest_edge": MAX_SIDE}
    quant.convert(fp32, AutoModelForImageTextToText, fmt, out, extra=[processor])


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


def evaluate(scorer, items, path):
    done = {json.loads(line)["item_id"] for line in path.open()} if path.exists() else set()
    with path.open("a") as file, torch.no_grad():
        for n, item in enumerate([i for i in items if i["item_id"] not in done], 1):
            letters = list(item["options"])
            logits, ms = timed(lambda: scorer.letter_logits(load_image(image_path(item)), item["prompt"], letters))
            probs = torch.softmax(logits, -1).tolist()
            finite = all(math.isfinite(p) for p in probs)
            prediction = item["options"][letters[int(np.argmax(probs))]] if finite else "invalid"
            file.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "label": item["label"],
                                   "prediction": prediction, "finite": finite, "ms": round(ms, 2)}) + "\n")
            if n % 500 == 0:
                file.flush()
                print(f"{n} items, {ms:.1f} ms last", flush=True)
    return [json.loads(line) for line in path.open()]


def summary(rows, warmup):
    def scores(part):
        correct = [r["prediction"] == r["label"] for r in part]
        risk = sum(FALSE_ALARM * (r["prediction"] != "good") if r["label"] == "good" else MISS * (r["prediction"] == "good")
                   for r in part)
        return {"items": len(part), "accuracy": round(float(np.mean(correct)), 4), "risk_per_1000": round(1000 * risk / len(part), 2)}
    return {"test_items": scores(rows), **{f"task_{t}": scores([r for r in rows if r["task"] == t]) for t in ("T2", "T3")},
            "non_finite": sum(not r["finite"] for r in rows), "latency_ms": vision.stats([r["ms"] for r in rows[warmup:]])}


def main():
    args = arguments()
    name = f"q_smolvlm_{args.model}_{args.format}"
    out = quant.STORE / f"smolvlm_{args.model}_{args.format}"
    if not (out / "config.json").exists():
        build(args.model, args.format, out)
    model = quant.load(AutoModelForImageTextToText, out, args.format)
    torch.cuda.reset_peak_memory_stats()
    scorer = LetterScorer(str(out), dtype=torch.float16 if args.format == "fp16" else torch.float32, device="cuda", model=model)
    test = load_items(("test",))
    rng = random.Random(0)
    items = sum((rng.sample([i for i in test if (i["label"] == "good") == g], args.sample // 2) for g in (True, False)), [])
    print(f"{name}: {quant.quantized_layers(model)} bitsandbytes layers, weights {quant.weight_mb(model)} MB, "
          f"stored {quant.folder_mb(out)} MB at {out}, {len(items)} test items", flush=True)
    res = results_dir(f"quant/{name}")
    write_env(f"quant/{name}")
    rows = evaluate(scorer, items, res / "predictions.jsonl")
    result = {"run": name, "model": args.model, "format": args.format, "stored_at": str(out), "stored_mb": quant.folder_mb(out),
              "weight_mb": quant.weight_mb(model), "bitsandbytes_layers": quant.quantized_layers(model), "gpu": vision.gpu_name(),
              "peak_gpu_mb": vision.peak_mb(), **summary(rows, args.warmup)}
    write_json(res / "summary.json", result)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
