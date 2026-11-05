import argparse
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fedorch.cache import load_cache
from fedorch.common import device, results_dir, runs_dir, write_json
from fedorch.decision import ACTION_TEXT, ACTIONS, QUESTION, config_name, evidence_text, load_evidence, soft_target, training_items, write_p
from fedorch.items import load_items

MODEL, REVISION = "sdmlai/nano-jev", "v1.0"


def examples(items, cache, evidence, temperature):
    return [(i["item_id"], evidence_text(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]]),
             [soft_target(i, cache[i["item_id"]], temperature)[a] for a in ACTIONS]) for i in items]


def option_logits(model, tokenizer, batch, run_on):
    firsts = [f"question: {QUESTION} option: {ACTION_TEXT[a]}" for _ in batch for a in ACTIONS]
    seconds = [b[1] for b in batch for _ in ACTIONS]
    encoded = tokenizer(firsts, seconds, truncation="only_second", max_length=320, padding=True, return_tensors="pt").to(run_on)
    with torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
        return model(**encoded).logits.float().view(len(batch), len(ACTIONS))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune nano-jev (LoRA) on the action choice with risk-based soft targets.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--temperature", type=float, default=1.0, help="soft-target temperature")
    parser.add_argument("--oversample", type=int, default=1, help="copies of each informative training frame")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items per split (for a quick test)")
    args = parser.parse_args()

    torch.manual_seed(0)
    run_on, items, cache, evidence = device(), load_items(), load_cache(), load_evidence()
    split = {s: [i for i in items if i["split"] == s][:args.limit] for s in ["train", "val", "test"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "key", "value"], modules_to_save=["classifier"])
    model = get_peft_model(model, config).to(run_on)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=run_on == "cuda")
    name = config_name("nanojev", args.temperature, args.oversample)
    train, history = examples(training_items(split["train"], cache, args.oversample), cache, evidence, args.temperature), []

    for epoch in range(args.epochs):
        model.train()
        random.Random(epoch).shuffle(train)
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            target = torch.tensor([b[2] for b in batch], device=run_on)
            loss = -(target * torch.log_softmax(option_logits(model, tokenizer, batch, run_on), -1)).sum(-1).mean()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            history.append(loss.item())
        print(f"epoch {epoch}: loss {sum(history[-200:]) / len(history[-200:]):.4f}")

    model.eval()
    p = {}
    with torch.no_grad():
        for s in ["val", "test"]:
            rows = examples(split[s], cache, evidence, args.temperature)
            for start in range(0, len(rows), 128):
                batch = rows[start:start + 128]
                for b, probs in zip(batch, torch.softmax(option_logits(model, tokenizer, batch, run_on), -1).tolist()):
                    p[b[0]] = dict(zip(ACTIONS, probs))
    write_p(name, p)
    model.save_pretrained(runs_dir("step4b") / f"{name}_lora")
    write_json(results_dir("step4b") / f"{name}_train.json", {"epochs": args.epochs, "lr": args.lr, "examples": len(train),
                                                               "loss_every_200_steps": history[::200], "lora": config.to_dict()})


if __name__ == "__main__":
    main()
