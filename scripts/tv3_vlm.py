import argparse
import io
import math
import os
import statistics
import time

import torch
from PIL import Image

from fedorch import vision
from fedorch.common import WORK
from fedorch.tools_v3 import (DEFECT_TYPES, Progress, append, done_images, held_images, image_file, laya_checkpoint,
                              recut_file, record_timing)

PROMPTS = {"qwen_any": "Look at this {c}. Does it have any visible defect, such as rust, corrosion, a missing cap, a bird nest or damage? Answer Yes or No.",
           **{f"qwen_{t}": "Look at this {c}. Is there " + DEFECT_TYPES[t] + " on it? Answer Yes or No." for t in DEFECT_TYPES},
           "qwen_other": "Look at this {c}. Is there peeling paint or a torn part on it? Answer Yes or No."}
KIND = {"qwen_any": "any", "qwen_other": "other-defect", "qwen_any_recut": "any", **{f"qwen_{t}": t for t in DEFECT_TYPES}}


def laya_recut():
    rows = [r for r in held_images() if recut_file(r).exists() and r["image"] not in done_images("laya_recut")]
    if not rows:
        print("laya_recut: no re-cut images", flush=True)
        return
    agent = vision.load("cuda", laya_checkpoint())
    progress, times = Progress("laya_recut", len(rows), 500), []
    for r in rows:
        start = time.perf_counter()
        answers = agent.predict({"image": str(recut_file(r))}, vision.QUESTIONS)["answers"]
        times.append(1000 * (time.perf_counter() - start))
        append("laya_recut", [{"image": r["image"], "answers": {q: vision.compact(a) for q, a in answers.items()}}])
        progress.tick()
    record_timing("laya_recut", statistics.median(times))


def prepare(path, side=512):
    image = Image.open(path).convert("RGB")
    image.thumbnail((side, side), Image.Resampling.BILINEAR)
    size = tuple(max(32, math.ceil(s / 32) * 32) for s in image.size)
    padded = Image.new("RGB", size, (114, 114, 114))
    padded.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return padded


def load_qwen():
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    checkpoint = os.environ.get("TV3_QWEN", str(WORK / "models" / "ppo_v2" / "qwen8b"))
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    processor = AutoProcessor.from_pretrained(checkpoint, min_pixels=32 * 32, max_pixels=512 ** 2)
    processor.tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, dtype=torch.float16, device_map={"": 0},
                                                        quantization_config=quant, attn_implementation="sdpa").eval()
    yes = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
    no = processor.tokenizer.encode("No", add_special_tokens=False)[0]
    return model, processor, yes, no


def ask(qwen_parts, image, texts):
    model, processor, yes, no = qwen_parts
    messages = [[{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": t}]}] for t in texts]
    inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                           return_tensors="pt", padding=True)
    inputs = {k: v.to(0, torch.float16) if v.is_floating_point() else v.to(0) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, [yes, no]].float()
    return torch.softmax(logits, -1)[:, 0].cpu().tolist()


def run_prompts(qwen_parts, tools, path, label):
    done = {t: done_images(t) for t in tools}
    rows = [r for r in held_images() if path(r).exists() and any(r["image"] not in done[t] for t in tools)]
    progress, times = Progress(label, len(rows), 200), []
    for r in rows:
        image = prepare(path(r))
        torch.cuda.synchronize()
        start = time.perf_counter()
        probs = ask(qwen_parts, image, [PROMPTS[t.replace("_recut", "")].format(c=r["component"]) for t in tools])
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start) / len(tools))
        for t, v in zip(tools, probs):
            if r["image"] not in done[t]:
                append(t, [{"image": r["image"], "defect": {KIND[t]: float(v)}}])
        progress.tick()
    if rows:
        buffer = io.BytesIO()
        prepare(path(rows[0])).save(buffer, format="JPEG", quality=90)
        for t in tools:
            record_timing(t, statistics.median(times[5:] or times), buffer.tell())


def qwen():
    parts = load_qwen()
    run_prompts(parts, list(PROMPTS), image_file, "qwen prompts")
    run_prompts(parts, ["qwen_any_recut"], recut_file, "qwen re-cut")


def main():
    parser = argparse.ArgumentParser(description="laya-vision on re-cut images and Qwen3-VL-8B yes/no prompts for each defect type.")
    parser.add_argument("parts", nargs="*", default=["laya_recut", "qwen"])
    args = parser.parse_args()
    failed = []
    for part in args.parts:
        start = time.time()
        try:
            {"laya_recut": laya_recut, "qwen": qwen}[part]()
            print(f"[{part}] done in {(time.time() - start) / 60:.1f} min", flush=True)
        except Exception as error:
            failed.append(part)
            print(f"[{part}] FAILED: {error!r}", flush=True)
        torch.cuda.empty_cache()
    print(f"[vlm] finished, failed parts: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
