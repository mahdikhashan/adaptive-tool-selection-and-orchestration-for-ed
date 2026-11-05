import argparse
import collections
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from fedorch.common import device, results_dir, runs_dir, write_env, write_json
from fedorch.folds import MODELS, fit_and_cache
from fedorch.items import image_path, load_items
from fedorch.vlm import LORA_TARGETS, LetterScorer, load_image

MODEL = "HuggingFaceTB/SmolVLM-256M-Instruct"


def loss_of(scorer, item):
    letters = list(item["options"])
    target = letters.index(next(k for k, v in item["options"].items() if v == item["label"]))
    logits = scorer.letter_logits(load_image(image_path(item)), item["prompt"], letters)
    return torch.nn.functional.cross_entropy(logits[None], torch.tensor([target], device=logits.device)), int(logits.argmax()) == target


def balanced_sampler(items, seed):
    counts = collections.Counter(item["label"] for item in items)
    weights = [counts[item["label"]] ** -0.5 for item in items]
    generator = random.Random(seed)
    while True:
        yield generator.choices(items, weights)[0]


def train_one(model_name, all_items, args):
    fit, cache = fit_and_cache(all_items, model_name)
    torch.manual_seed(args.seed)
    scorer = LetterScorer(MODEL, dtype=torch.float32, device=device())
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.15, target_modules=LORA_TARGETS)
    scorer.model = get_peft_model(scorer.model, config)
    optimizer = torch.optim.AdamW([p for p in scorer.model.parameters() if p.requires_grad], lr=args.lr)
    schedule = get_cosine_schedule_with_warmup(optimizer, 50, args.steps)
    samples = balanced_sampler(fit, args.seed)
    history = []

    scorer.model.train()
    for step in range(args.steps):
        losses = []
        for _ in range(args.accumulation):
            loss, _ = loss_of(scorer, next(samples))
            (loss / args.accumulation).backward()
            losses.append(loss.item())
        optimizer.step()
        schedule.step()
        optimizer.zero_grad()
        history.append(float(np.mean(losses)))
        if step % 50 == 0:
            print(f"{model_name} step {step}: loss {np.mean(history[-50:]):.3f}")

    scorer.model.eval()
    held_out = random.Random(args.seed).sample(cache, min(args.val_items, len(cache)))
    with torch.no_grad():
        correct = [loss_of(scorer, item)[1] for item in held_out]
    scorer.model.save_pretrained(runs_dir("step2") / f"smolvlm_lora_{model_name}")
    write_json(results_dir("step2") / f"smolvlm_train_{model_name}.json", {
        "model": model_name, "fit_towers": sorted({i["tower"] for i in fit}), "fit_items": len(fit),
        "steps": args.steps, "accumulation": args.accumulation, "lr": args.lr, "lora": config.to_dict(),
        "loss_every_50_steps": [float(np.mean(history[i:i + 50])) for i in range(0, len(history), 50)],
        "held_out_items": len(held_out), "held_out_accuracy": float(np.mean(correct)),
    })
    print(f"{model_name} held-out accuracy {np.mean(correct):.3f}")


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM-256M (EVLM recipe) once per cross-fitting fold, plus once on all train and val towers for the test towers.")
    parser.add_argument("models", nargs="*", help=f"any of: {', '.join(MODELS)} (default: all)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model: {', '.join(unknown)}")

    all_items = load_items()
    for model_name in args.models or MODELS:
        train_one(model_name, all_items, args)
    write_env("step2")


if __name__ == "__main__":
    main()
