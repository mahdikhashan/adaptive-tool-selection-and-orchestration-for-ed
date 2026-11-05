import argparse
import json
import math
import time

import torch

from fedorch.classifier import load_classifier, option_probabilities, to_tensor
from fedorch.common import device, gpu_dtype, results_dir, runs_dir, write_env
from fedorch.items import image_path, load_items
from fedorch.vlm import Int8Split, LetterScorer, jpeg_bytes, load_image

SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
CLOUD = "Qwen/Qwen3-VL-8B-Instruct"
TOOLS = ["classifier", "smolvlm_base", "smolvlm_local", "smolvlm_split", "cloud"]


def timed(run_on, fn):
    if run_on == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = fn()
    if run_on == "cuda":
        torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1000


def classifier_tool(run_on):
    model = load_classifier(runs_dir("step2") / "classifier.pt", run_on)

    def run(item):
        pixels = to_tensor(image_path(item))[None].to(run_on)
        with torch.no_grad(), torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
            logits, ms = timed(run_on, lambda: model(pixel_values=pixels).logits[0].float())
        return option_probabilities(torch.softmax(logits, -1).tolist(), item), ms, 0
    return run


def vlm_tool(run_on, model_id, dtype, adapter=None, split=False, cloud=False):
    scorer = LetterScorer(model_id, dtype=dtype, device=run_on, adapter=adapter)
    int8 = Int8Split(scorer.model) if split else None

    def run(item):
        image = load_image(image_path(item))
        letters = list(item["options"])
        with torch.no_grad():
            logits, ms = timed(run_on, lambda: scorer.letter_logits(image, item["prompt"], letters))
        probs = torch.softmax(logits, -1).tolist()
        payload = jpeg_bytes(image) if cloud else int8.payload_bytes if split else 0
        return {item["options"][letter]: p for letter, p in zip(letters, probs)}, ms, payload
    return run


def build(tool, run_on, cloud_model):
    adapter = runs_dir("step2") / "smolvlm_lora"
    return {
        "classifier": lambda: classifier_tool(run_on),
        "smolvlm_base": lambda: vlm_tool(run_on, SMOLVLM, torch.float32),
        "smolvlm_local": lambda: vlm_tool(run_on, SMOLVLM, torch.float32, adapter=adapter),
        "smolvlm_split": lambda: vlm_tool(run_on, SMOLVLM, torch.float32, adapter=adapter, split=True),
        "cloud": lambda: vlm_tool(run_on, cloud_model, gpu_dtype(), cloud=True),
    }[tool]()


def row(item, tool, probs, ms, payload):
    finite = all(math.isfinite(p) for p in probs.values())
    prediction = max(probs, key=probs.get) if finite else "invalid"
    return {"item_id": item["item_id"], "tool": tool, "prediction": prediction, "correct": prediction == item["label"],
            "abstained": prediction == "cannot-judge", "confidence": probs.get(prediction, 0.0) if finite else 0.0,
            "probs": probs, "compute_ms": round(ms, 2), "payload_bytes": payload, "finite": finite}


def main():
    parser = argparse.ArgumentParser(description="Run one tool on every tower14 item and append one JSON line per item.")
    parser.add_argument("tool", help=f"one of: {', '.join(TOOLS)}")
    parser.add_argument("--cloud-model", default=CLOUD)
    parser.add_argument("--limit", type=int, default=None, help="run only the first N items (for a quick test)")
    args = parser.parse_args()
    if args.tool not in TOOLS:
        parser.error(f"unknown tool: {args.tool}")

    out = results_dir("step2") / f"tool_{args.tool}.jsonl"
    done = {json.loads(line)["item_id"] for line in out.open()} if out.exists() else set()
    todo = [item for item in load_items()[:args.limit] if item["item_id"] not in done]
    print(f"{args.tool}: {len(done)} items done, {len(todo)} to run")

    run_on = device()
    run = build(args.tool, run_on, args.cloud_model)
    with out.open("a") as file:
        for count, item in enumerate(todo, 1):
            file.write(json.dumps(row(item, args.tool, *run(item))) + "\n")
            if count % 500 == 0:
                file.flush()
                print(f"{count}/{len(todo)}")
    write_env("step2")


if __name__ == "__main__":
    main()
