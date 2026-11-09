"""Cache for the base-laya agent: views, ResNet and Qwen tool outputs, and frozen base-laya
option vectors (scorer inputs) for every reachable (view, ResNet-shown, Qwen-shown) state.

Base laya (4-bit, not fine-tuned) answers one call with four questions: the next action and the
three report questions. The forward hook stores the vector each option marker gives to laya's
scorer, so PPO can train the scorer later without running laya again.
"""
import argparse
import json
import random
import time

import numpy as np
import torch
from PIL import Image

from fedorch import quant, vision
from fedorch.common import WORK, results_dir, runs_dir
from fedorch.items import CLASSES, INSPLAD, load_items

VIEWS = ["full image", "top-left", "top-right", "bottom-left", "bottom-right", "centre"]
ACTIONS = ["write the inspection report now", "zoom into the top-left part", "zoom into the top-right part",
           "zoom into the bottom-left part", "zoom into the bottom-right part", "zoom into the centre",
           "run the onboard defect classifier on this view", "ask the cloud vision-language model about the full image"]
ACTION_Q = {"type": "choice", "criteria": ACTIONS,
            "instructions": "What should the drone do next to complete the inspection report for this power line component?"}
QUESTIONS = {"action": ACTION_Q, **{k: vision.QUESTIONS[k] for k in ("component", "defect", "condition")}}
COMPONENTS = vision.QUESTIONS["component"]["criteria"]
CONDITIONS = vision.QUESTIONS["condition"]["criteria"]
SIZES = [len(ACTIONS), len(COMPONENTS), 2, len(CONDITIONS)]


def views(image):
    w, h = image.size
    boxes = [None, (0, 0, 2 * w // 3, 2 * h // 3), (w // 3, 0, w, 2 * h // 3), (0, h // 3, 2 * w // 3, h),
             (w // 3, h // 3, w, h), (w // 6, h // 6, 5 * w // 6, 5 * h // 6)]
    out, times = [], []
    for box in boxes:
        start = time.perf_counter()
        out.append(image if box is None else image.crop(box))
        times.append(time.perf_counter() - start)
    return out, times


def resnet(device):
    from fedorch.classifier import MEAN, SIZE, STD, load_classifier
    model = load_classifier(runs_dir("step2") / "classifier_full.pt", device)
    text = [vision.CONDITION[c] for c in CLASSES]

    def run(image):
        pixels = (np.asarray(image.resize((SIZE, SIZE)), dtype=np.float32) / 255 - MEAN) / STD
        x = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].to(device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=x).logits[0].float(), -1).tolist()
        torch.cuda.synchronize()
        return dict(zip(text, probs)), time.perf_counter() - start
    return run


def base_laya():
    from laya.vlm import load_vlm
    from transformers import AutoModel
    folder = quant.STORE / "laya_base_nf4_fp16"
    agent = load_vlm(str(folder), device="cuda", dtype="fp32")
    agent.model.encoder = quant.load(AutoModel, folder / "backbone_quantized", attn_implementation="sdpa")
    agent._torch_dtype = lambda: torch.float16
    captured = []
    agent.model.scorer.register_forward_hook(lambda module, inputs, output: captured.append(inputs[0].detach().float().cpu()))

    def run(image, evidence):
        captured.clear()
        state = {"image": image}
        if evidence:
            state["evidence"] = evidence
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            agent.predict(state, QUESTIONS, prefix_cache=False)
        torch.cuda.synchronize()
        rows = torch.cat(captured, 0)
        if rows.shape[0] != len(QUESTIONS):
            raise RuntimeError("expected one scorer row per question")
        return np.concatenate([rows[r, :k].numpy() for r, k in enumerate(SIZES)]).astype(np.float16), time.perf_counter() - start
    return agent, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=int, default=800)
    parser.add_argument("--test", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qwen-tokens", type=int, default=32)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    rows = [r for r in vision.images(load_items()) if r["split"] in ("val", "test")]
    rows = [r for s, n in (("val", args.val), ("test", args.test)) for r in rng.sample([x for x in rows if x["split"] == s], n)]
    out = runs_dir("agent_cache")
    meta_path = out / "meta.jsonl"
    done = {json.loads(l)["image"] for l in meta_path.open()} if meta_path.exists() else set()
    rows = [r for r in rows if r["image"] not in done]
    print(f"{len(rows)} images to cache", flush=True)
    if not rows:
        return
    tools = {}
    clf = resnet("cuda")
    for row in rows:
        image = Image.open(INSPLAD / row["image"]).convert("RGB")
        vs, crop_s = views(image)
        tools[row["image"]] = {"crop_s": crop_s, "resnet": [clf(v) for v in vs]}
    del clf
    torch.cuda.empty_cache()
    from fedorch.ppo_v2_tools import LocalQwenAdapter
    qwen = LocalQwenAdapter(WORK / "models/ppo_v2/qwen8b", "nf4", "cuda", max_new_tokens=args.qwen_tokens)
    for n, row in enumerate(rows, 1):
        start = time.perf_counter()
        result = qwen(Image.open(INSPLAD / row["image"]).convert("RGB"))
        tools[row["image"]]["qwen"] = (result["explanation"], time.perf_counter() - start, result["bytes_up"] + result["bytes_down"])
        if n % 100 == 0:
            print(f"qwen {n}/{len(rows)}", flush=True)
    del qwen
    torch.cuda.empty_cache()
    agent, run = base_laya()
    torch.save(agent.model.scorer.state_dict(), out / "base_scorer.pt")
    start = time.time()
    with meta_path.open("a") as meta:
        for n, row in enumerate(rows, 1):
            image = Image.open(INSPLAD / row["image"]).convert("RGB")
            vs, _ = views(image)
            t = tools[row["image"]]
            text, qwen_s, qwen_bytes = t["qwen"]
            vectors = np.zeros((len(VIEWS), 2, 2, sum(SIZES), 576), np.float16)
            laya_s = np.zeros((len(VIEWS), 2, 2))
            for v, view in enumerate(vs):
                probs = t["resnet"][v][0]
                top = sorted(probs.items(), key=lambda kv: -kv[1])[:2]
                for r in (0, 1):
                    for c in (0, 1):
                        evidence = {}
                        if r:
                            evidence["onboard defect classifier on this view (prediction, not truth)"] = ", ".join(f"{k} {p:.2f}" for k, p in top)
                        if c:
                            evidence["cloud vision-language model on the full image (prediction, not truth)"] = text
                        vectors[v, r, c], laya_s[v, r, c] = run(view, evidence)
            np.save(out / f"{n + len(done):05d}.npy", vectors)
            meta.write(json.dumps({"file": f"{n + len(done):05d}.npy", "image": row["image"], "split": row["split"],
                                   "tower": row["tower"], "component": COMPONENTS.index(row["component"]),
                                   "defect": int(row["defect"]), "condition": CONDITIONS.index(row["condition"]),
                                   "laya_s": laya_s.round(4).tolist(), "crop_s": [round(x, 5) for x in t["crop_s"]],
                                   "resnet_s": [round(x[1], 4) for x in t["resnet"]], "resnet": [x[0] for x in t["resnet"]],
                                   "qwen_s": round(qwen_s, 3), "qwen_bytes": qwen_bytes, "qwen_text": text}) + "\n")
            meta.flush()
            if n % 25 == 0 or n == len(rows):
                print(f"laya {n}/{len(rows)} images, {(time.time() - start) / n:.2f} s each", flush=True)
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
