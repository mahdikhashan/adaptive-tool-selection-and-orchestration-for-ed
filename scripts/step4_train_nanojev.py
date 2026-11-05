import argparse
import random

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fedorch.cache import load_cache
from fedorch.common import device, results_dir, runs_dir, write_json
from fedorch.features import QUESTION, TOOL_TEXT, TOOLS, evidence, write_q
from fedorch.items import load_items

MODEL, REVISION = "sdmlai/nano-jev", "v1.0"


def pairs(items, cache):
    return [(f"question: {QUESTION} option: {TOOL_TEXT[tool]}", evidence(item, cache[item["item_id"]]["classifier"]),
             float(cache[item["item_id"]][tool]["correct"]), item["item_id"], tool) for item in items for tool in TOOLS]


def logits(model, tokenizer, batch, run_on):
    encoded = tokenizer([b[0] for b in batch], [b[1] for b in batch], truncation="only_second", max_length=256,
                        padding=True, return_tensors="pt").to(run_on)
    with torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
        return model(**encoded).logits.squeeze(-1).float()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune nano-jev (LoRA) to predict whether each tool is correct.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items per split (for a quick test)")
    args = parser.parse_args()

    torch.manual_seed(0)
    run_on, items, cache = device(), load_items(), load_cache()
    split = {name: [i for i in items if i["split"] == name][:args.limit] for name in ["train", "val", "test"]}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION)
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, target_modules=["query", "key", "value"], modules_to_save=["classifier"])
    model = get_peft_model(model, config).to(run_on)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=run_on == "cuda")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    train, history = pairs(split["train"], cache), []

    for epoch in range(args.epochs):
        model.train()
        random.Random(epoch).shuffle(train)
        for start in range(0, len(train), args.batch_size):
            batch = train[start:start + args.batch_size]
            loss = loss_fn(logits(model, tokenizer, batch, run_on), torch.tensor([b[2] for b in batch], device=run_on))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            history.append(loss.item())
        print(f"epoch {epoch}: loss {sum(history[-200:]) / len(history[-200:]):.4f}")

    model.eval()
    q = {}
    with torch.no_grad():
        for name in ["val", "test"]:
            rows = pairs(split[name], cache)
            for start in range(0, len(rows), 256):
                batch = rows[start:start + 256]
                for b, p in zip(batch, torch.sigmoid(logits(model, tokenizer, batch, run_on)).tolist()):
                    q.setdefault(b[3], {})[b[4]] = p
    write_q("nanojev", q)
    model.save_pretrained(runs_dir("step4") / "nanojev_lora")
    write_json(results_dir("step4") / "nanojev_train.json", {"epochs": args.epochs, "lr": args.lr, "pairs": len(train),
                                                              "loss_every_200_steps": history[::200], "lora": config.to_dict()})


if __name__ == "__main__":
    main()
