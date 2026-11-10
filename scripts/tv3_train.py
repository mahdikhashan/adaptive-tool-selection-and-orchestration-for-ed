import argparse
import math
import statistics
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

from fedorch.items import CLASSES
from fedorch.tools_v3 import (GROUPS, Progress, append, condition_label, done_images, group_of, held_images, image_file,
                              models_dir, record_timing, train_images)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODELS = {"convnext": ("facebook/convnext-tiny-224", 224), "efficientnet": ("google/efficientnet-b0", 224),
          **{f"spec_{g}": ("microsoft/resnet-50", 384) for g in GROUPS}}


def pixels(path, size, flip=False):
    image = Image.open(path).convert("RGB").resize((size, size))
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    array = (np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, size, train):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        flip = self.train and np.random.rand() < 0.5
        return {"pixel_values": pixels(image_file(row), self.size, flip), "labels": CLASSES.index(condition_label(row))}


def balanced(rows):
    counts = Counter(condition_label(r) for r in rows)
    top = max(counts.values())
    out = []
    for r in rows:
        out += [r] * max(1, round(math.sqrt(top / counts[condition_label(r)])))
    return out


def split_by_tower(rows):
    towers = sorted({r["tower"] for r in rows})
    held = set(towers[::6])
    return [r for r in rows if r["tower"] not in held], [r for r in rows if r["tower"] in held]


def macro_recall(prediction):
    predicted, actual = prediction.predictions.argmax(-1), prediction.label_ids
    recalls = [float((predicted[actual == k] == k).mean()) for k in range(len(CLASSES)) if (actual == k).any()]
    return {"macro_recall": float(np.mean(recalls))}


def train(name, args):
    checkpoint, size = MODELS[name]
    rows = train_images()
    if name.startswith("spec_"):
        rows = [r for r in rows if group_of(r["asset"]) == name[5:]]
    fit, select = split_by_tower(rows)
    print(f"[{name}] training images {len(fit)}, selection images {len(select)}", flush=True)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=len(CLASSES), ignore_mismatched_sizes=True)
    out = models_dir() / name
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=args.epochs, learning_rate=args.lr,
                              per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                              eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                              metric_for_best_model="macro_recall", greater_is_better=True, fp16=torch.cuda.is_available(),
                              dataloader_num_workers=args.workers, remove_unused_columns=False, report_to="none",
                              logging_steps=50, disable_tqdm=True, weight_decay=0.05)
    trainer = Trainer(model=model, args=targs, train_dataset=Images(balanced(fit), size, True),
                      eval_dataset=Images(select or fit[:200], size, False), compute_metrics=macro_recall)
    trainer.train()
    trainer.save_model(str(out / "best"))
    return trainer.model.eval(), size


def predict(name, model, size, args):
    rows = held_images()
    if name.startswith("spec_"):
        rows = [r for r in rows if group_of(r["asset"]) == name[5:]]
    done = done_images(name)
    rows = [r for r in rows if r["image"] not in done]
    run_on = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(run_on)
    times = []
    for r in rows[:30]:
        x = pixels(image_file(r), size).unsqueeze(0).to(run_on)
        if run_on == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(pixel_values=x)
        if run_on == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    if times:
        record_timing(name, statistics.median(times[5:] or times))
    progress = Progress(f"{name} predict", len(rows), 1000)
    for k in range(0, len(rows), args.batch_size):
        batch = rows[k:k + args.batch_size]
        x = torch.stack([pixels(image_file(r), size) for r in batch]).to(run_on)
        with torch.no_grad(), torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
            probs = torch.softmax(model(pixel_values=x).logits.float(), -1).cpu().numpy()
        append(name, [{"image": r["image"], "probs": dict(zip(CLASSES, map(float, p.round(5))))} for r, p in zip(batch, probs)])
        progress.tick(len(batch))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune image-classifier tools with the Hugging Face Trainer and cache their answers on val/test images.")
    parser.add_argument("models", nargs="*", default=list(MODELS))
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    for name in args.models:
        start = time.time()
        best = models_dir() / name / "best"
        if best.exists():
            model, size = AutoModelForImageClassification.from_pretrained(str(best)).eval(), MODELS[name][1]
            print(f"[{name}] using trained model {best}", flush=True)
        else:
            model, size = train(name, args)
        predict(name, model, size, args)
        print(f"[{name}] done in {(time.time() - start) / 60:.1f} min", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
