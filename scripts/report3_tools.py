"""Ask every tool the three laya report questions (component, defect, condition) on val/test images.

One JSON line per (image, tool). Resumable. Letter scoring for the VLMs (one forward pass per question).
"""
import argparse
import json
import random
import time

import torch

from fedorch import vision
from fedorch.common import WORK, results_dir, runs_dir
from fedorch.items import INSPLAD, CLASSES, load_items
from fedorch.vlm import LetterScorer, load_image

SMOLVLM = "HuggingFaceTB/SmolVLM-256M-Instruct"
COMPONENTS = list(dict.fromkeys(vision.COMPONENT.values()))
CONDITIONS = list(vision.CONDITION.values())
LETTERS = "ABCDEFGHIJK"


def prompt(question, options):
    lines = "\n".join(f"{LETTERS[n]}. {o}" for n, o in enumerate(options))
    return f"This image shows a component on a power line. {question}\nOptions:\n{lines}\nAnswer with one letter."


QUESTIONS = {
    "component": ("Which power line component is in this image?", COMPONENTS),
    "defect": ("Does this power line component have a defect?", ["good, no defect", "defect"]),
    "condition": ("Which condition does this power line component have?", CONDITIONS),
}


def vlm_tool(scorer):
    scorer.letter_ids.update({l: scorer.token_ids(l) for l in LETTERS})

    def run(row):
        image = load_image(INSPLAD / row["image"])
        out = {}
        for name, (question, options) in QUESTIONS.items():
            with torch.no_grad():
                logits = scorer.letter_logits(image, prompt(question, options), list(LETTERS[:len(options)]))
            out[name] = dict(zip(options, torch.softmax(logits, -1).tolist()))
        out["defect"] = out["defect"]["defect"]
        return out
    return run


def classifier_tool(device):
    from fedorch.classifier import load_classifier, to_tensor
    model = load_classifier(runs_dir("step2") / "classifier_full.pt", device)
    text = [vision.CONDITION[c] for c in CLASSES]

    def run(row):
        pixels = to_tensor(INSPLAD / row["image"])[None].to(device)
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=pixels).logits[0].float(), -1).tolist()
        return {"component": None, "defect": 1 - probs[0], "condition": dict(zip(text, probs))}
    return run


def laya_tool():
    from laya.vlm import load_vlm
    from transformers import AutoModel
    from fedorch import quant
    folder = quant.STORE / "laya_good300_full_nf4_fp16"
    agent = load_vlm(str(folder), device="cuda", dtype="fp32")
    agent.model.encoder = quant.load(AutoModel, folder / "backbone_quantized", attn_implementation="sdpa")
    agent._torch_dtype = lambda: torch.float16
    questions = {k: vision.QUESTIONS[k] for k in ("component", "defect", "condition")}

    def run(row):
        answers, _, truncated = vision.predict(agent, row["image"], questions)
        if truncated:
            raise ValueError("laya input truncated")
        return answers
    return run


def build(tool):
    if tool == "classifier":
        return classifier_tool("cuda")
    if tool == "laya":
        return laya_tool()
    if tool == "smolvlm":
        return vlm_tool(LetterScorer(SMOLVLM, dtype=torch.float32, device="cuda", adapter=runs_dir("step2") / "smolvlm_lora_full"))
    from transformers import AutoModelForImageTextToText
    from fedorch import quant
    model = AutoModelForImageTextToText.from_pretrained(str(WORK / "models/ppo_v2/qwen8b"), quantization_config=quant.NF4,
                                                        dtype=torch.float16, device_map={"": 0}).eval()
    return vlm_tool(LetterScorer(str(WORK / "models/ppo_v2/qwen8b"), dtype=torch.float16, device="cuda", model=model))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", default="classifier,laya,smolvlm,qwen")
    parser.add_argument("--per-split", type=int, default=2000, help="images per split (val, test), 0 = all")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rows = [r for r in vision.images(load_items()) if r["split"] in ("val", "test")]
    if args.per_split:
        rng = random.Random(args.seed)
        rows = [r for s in ("val", "test") for r in rng.sample([x for x in rows if x["split"] == s], min(args.per_split, sum(x["split"] == s for x in rows)))]
    path = results_dir("report3") / "answers.jsonl"
    done = {(d["image"], d["tool"]) for d in map(json.loads, path.open())} if path.exists() else set()
    for tool in args.tools.split(","):
        todo = [r for r in rows if (r["image"], tool) not in done]
        if not todo:
            continue
        run = build(tool)
        start = time.time()
        with path.open("a") as out:
            for n, row in enumerate(todo, 1):
                t0 = time.perf_counter()
                answers = run(row)
                torch.cuda.synchronize()
                out.write(json.dumps({"image": row["image"], "split": row["split"], "tool": tool, "answers": answers,
                                      "ms": round(1000 * (time.perf_counter() - t0), 1)}) + "\n")
                if n % 200 == 0 or n == len(todo):
                    out.flush()
                    print(f"{tool}: {n}/{len(todo)} images, {(time.time() - start) / n:.3f} s each", flush=True)
        del run
        torch.cuda.empty_cache()
    print(f"done, wrote {path}", flush=True)


if __name__ == "__main__":
    main()
