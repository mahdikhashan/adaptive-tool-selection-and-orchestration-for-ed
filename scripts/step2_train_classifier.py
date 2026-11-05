import argparse
import collections

import numpy as np
import torch

from fedorch.classifier import new_classifier, to_tensor
from fedorch.common import device, results_dir, runs_dir, write_env, write_json
from fedorch.folds import MODELS, fit_and_cache, select_split
from fedorch.items import CLASSES, image_path, load_items


class Crops(torch.utils.data.Dataset):
    def __init__(self, items, train):
        self.items = items
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return to_tensor(image_path(item), self.train), CLASSES.index(item["label"])


def unique_t3(items, limit):
    by_image = {item["image"]: item for item in items if item["task"] == "T3"}
    return list(by_image.values())[:limit]


def class_weights(items):
    counts = collections.Counter(item["label"] for item in items)
    weights = torch.tensor([counts.get(c, 1) ** -0.5 for c in CLASSES])
    return weights / weights.mean()


def evaluate(model, loader, run_on):
    model.eval()
    predicted, actual = [], []
    with torch.no_grad(), torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
        for pixels, labels in loader:
            predicted += model(pixel_values=pixels.to(run_on)).logits.argmax(-1).tolist()
            actual += labels.tolist()
    predicted, actual = np.array(predicted), np.array(actual)
    recall = {c: float((predicted[actual == i] == i).mean()) for i, c in enumerate(CLASSES) if (actual == i).any()}
    return {"accuracy": float((predicted == actual).mean()), "macro_recall": float(np.mean(list(recall.values()))), "recall": recall}


def train_one(model_name, all_items, args, run_on):
    fit, _ = fit_and_cache(all_items, model_name)
    train_items, select_items = select_split(fit)
    train, val = unique_t3(train_items, args.limit), unique_t3(select_items, args.limit)
    loaders = {name: torch.utils.data.DataLoader(Crops(items, name == "train"), batch_size=args.batch_size,
                                                 shuffle=name == "train", num_workers=args.workers)
               for name, items in [("train", train), ("val", val)]}
    model = new_classifier().to(run_on)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scaler = torch.amp.GradScaler(enabled=run_on == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights(train).to(run_on))
    history, best = [], -1.0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for pixels, labels in loaders["train"]:
            with torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
                loss = loss_fn(model(pixel_values=pixels.to(run_on)).logits, labels.to(run_on))
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
        metrics = {"epoch": epoch, "train_loss": float(np.mean(losses)), **evaluate(model, loaders["val"], run_on)}
        history.append(metrics)
        print(f"{model_name} epoch {epoch}: loss {metrics['train_loss']:.3f}, val accuracy {metrics['accuracy']:.3f}, val macro recall {metrics['macro_recall']:.3f}")
        if metrics["macro_recall"] > best:
            best = metrics["macro_recall"]
            torch.save(model.state_dict(), runs_dir("step2") / f"classifier_{model_name}.pt")

    write_json(results_dir("step2") / f"classifier_train_{model_name}.json", {
        "model": model_name, "train_towers": sorted({i["tower"] for i in train_items}),
        "select_towers": sorted({i["tower"] for i in select_items}),
        "train_images": len(train), "val_images": len(val), "best_val_macro_recall": best, "history": history})


def main():
    parser = argparse.ArgumentParser(description="Train one ResNet-50 defect classifier (tier-1 tool) per cross-fitting fold, plus one on all train and val towers for the test towers.")
    parser.add_argument("models", nargs="*", help=f"any of: {', '.join(MODELS)} (default: all)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N images per set (for a quick test)")
    args = parser.parse_args()
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"unknown model: {', '.join(unknown)}")

    run_on = device()
    all_items = load_items()
    for model_name in args.models or MODELS:
        train_one(model_name, all_items, args, run_on)
    write_env("step2")


if __name__ == "__main__":
    main()
