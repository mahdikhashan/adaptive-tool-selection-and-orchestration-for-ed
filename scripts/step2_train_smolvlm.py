import argparse
import collections
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_cosine_schedule_with_warmup

from fedorch.common import device, results_dir, runs_dir, write_env, write_json
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


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tune SmolVLM-256M on the tower14 train towers (EVLM recipe).")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--val-items", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    scorer = LetterScorer(MODEL, dtype=torch.float32, device=device())
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.15, target_modules=LORA_TARGETS)
    scorer.model = get_peft_model(scorer.model, config)
    optimizer = torch.optim.AdamW([p for p in scorer.model.parameters() if p.requires_grad], lr=args.lr)
    schedule = get_cosine_schedule_with_warmup(optimizer, 50, args.steps)
    samples = balanced_sampler(load_items(["train"]), args.seed)
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
            print(f"step {step}: loss {np.mean(history[-50:]):.3f}")

    scorer.model.eval()
    val = random.Random(args.seed).sample(load_items(["val"]), args.val_items)
    with torch.no_grad():
        correct = [loss_of(scorer, item)[1] for item in val]
    scorer.model.save_pretrained(runs_dir("step2") / "smolvlm_lora")
    write_json(results_dir("step2") / "smolvlm_train.json", {
        "steps": args.steps, "accumulation": args.accumulation, "lr": args.lr, "lora": config.to_dict(),
        "loss_every_50_steps": [float(np.mean(history[i:i + 50])) for i in range(0, len(history), 50)],
        "val_items": args.val_items, "val_accuracy": float(np.mean(correct)),
    })
    write_env("step2")
    print(f"val accuracy {np.mean(correct):.3f}")


if __name__ == "__main__":
    main()
