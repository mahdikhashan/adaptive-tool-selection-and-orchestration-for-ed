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
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def build(model, out):
    weights = AutoModelForImageTextToText.from_pretrained(SMOLVLM, dtype=torch.float16)
    if model == "lora":
        from peft import PeftModel
        if not ADAPTER.exists():
            raise SystemExit(f"missing {ADAPTER}: run 'uv run --locked python scripts/step2_train_smolvlm.py full' first")
        weights = PeftModel.from_pretrained(weights, str(ADAPTER)).merge_and_unload()
    processor = AutoProcessor.from_pretrained(SMOLVLM)
    processor.image_processor.size = {"longest_edge": MAX_SIDE}
    quant.convert(weights, AutoModelForImageTextToText, out, extra=[processor])


def timed(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, 1000 * (time.perf_counter() - start)


def evaluate(scorer, items, path):
    done = {json.loads(line)["item_id"] for line in path.open()} if path.exists() else set()
    todo = [i for i in items if i["item_id"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to do", flush=True)
    with path.open("a") as file, torch.no_grad():
        for n, item in enumerate(todo, 1):
            letters = list(item["options"])
            logits, ms = timed(lambda: scorer.letter_logits(load_image(image_path(item)), item["prompt"], letters))
            probs = torch.softmax(logits, -1).tolist()
            finite = all(math.isfinite(p) for p in probs)
            prediction = item["options"][letters[int(np.argmax(probs))]] if finite else "invalid"
            file.write(json.dumps({"item_id": item["item_id"], "task": item["task"], "label": item["label"], "prediction": prediction,
                                   "probs": {item["options"][k]: round(p, 5) for k, p in zip(letters, probs)},
                                   "finite": finite, "ms": round(ms, 2)}) + "\n")
            if n % 500 == 0:
                file.flush()
                print(f"{n}/{len(todo)} items, {ms:.1f} ms last", flush=True)
    return [json.loads(line) for line in path.open()]


def scores(rows):
    correct = np.array([r["prediction"] == r["label"] for r in rows])
    risk = sum(FALSE_ALARM * (r["prediction"] != "good") if r["label"] == "good" else MISS * (r["prediction"] == "good") for r in rows)
    rng = np.random.default_rng(0)
    boot = [correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]
    return {"items": len(rows), "accuracy": round(float(correct.mean()), 4),
            "accuracy_95ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "risk_per_1000": round(1000 * risk / len(rows), 2)}


def main():
    args = arguments()
    name = f"q4_smolvlm_{args.model}"
    out = quant.STORE / f"smolvlm_{args.model}_nf4_fp16"
    if not (out / "config.json").exists():
        build(args.model, out)
    model = quant.load(AutoModelForImageTextToText, out)
    loaded = quant.allocated_mb()
    torch.cuda.reset_peak_memory_stats()
    scorer = LetterScorer(str(out), dtype=torch.float16, device="cuda", model=model)
    items = load_items(("test",))
    print(f"{name}: {quant.quantized_layers(model)} 4-bit layers, dtypes {quant.float_dtypes(model)}, weights {quant.weight_mb(model)} MB, "
          f"stored {quant.folder_mb(out)} MB, GPU after load {loaded} MB", flush=True)
    res = results_dir(f"quant/{name}")
    write_env(f"quant/{name}")
    for item in random.Random(0).sample(items, args.warmup):
        with torch.no_grad():
            scorer.letter_logits(load_image(image_path(item)), item["prompt"], list(item["options"]))
    start = time.time()
    rows = evaluate(scorer, items, res / "predictions.jsonl")
    result = {"run": name, "family": "SmolVLM-256M", "model": args.model, "format": "nf4 weights, fp16 compute",
              "stored_at": str(out), "stored_mb": quant.folder_mb(out), "weight_mb": quant.weight_mb(model), "gpu_mb_after_load": loaded,
              "four_bit_layers": quant.quantized_layers(model), "float_dtypes": quant.float_dtypes(model), "gpu": vision.gpu_name(),
              "peak_gpu_mb": vision.peak_mb(), "seconds": round(time.time() - start, 1), "test_items": scores(rows),
              **{f"task_{t}": scores([r for r in rows if r["task"] == t]) for t in ("T2", "T3")},
              "non_finite": sum(not r["finite"] for r in rows), "latency_ms": vision.stats([r["ms"] for r in rows]),
              "latency_note": "one letter-scoring forward pass for each item, image already loaded"}
    write_json(res / "summary.json", result)
    print(json.dumps(result, indent=1), flush=True)


if __name__ == "__main__":
    main()
