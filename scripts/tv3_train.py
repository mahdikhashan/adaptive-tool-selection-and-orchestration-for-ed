import argparse
import math
import shutil
import statistics
import time
from collections import Counter

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification, Trainer, TrainingArguments

from fedorch import vision
from fedorch.items import CLASSES
from fedorch.tools_v3 import (GROUPS, Progress, append, condition_label, done_images, group_of, held_images, image_file,
                              models_dir, recut_file, record_timing, specialist_group, train_images)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MODELS = {"convnext": ("facebook/convnext-tiny-224", 224), "efficientnet": ("google/efficientnet-b0", 224),
          "resnet18": ("microsoft/resnet-18", 224), "resnet101": ("microsoft/resnet-101", 224),
          "vit_base": ("google/vit-base-patch16-224", 224), "deit_small": ("facebook/deit-small-patch16-224", 224),
          "swin_tiny": ("microsoft/swin-tiny-patch4-window7-224", 224), "mobilenet_v2": ("google/mobilenet_v2_1.0_224", 224),
          "efficientnet_b3": ("google/efficientnet-b3", 300), "convnext_small": ("facebook/convnext-small-224", 224),
          **{f"spec_{g}": ("microsoft/resnet-50", 384) for g in GROUPS},
          **{f"cspec_{g}": ("facebook/convnext-tiny-224", 224) for g in GROUPS}}
COMPONENT_MODELS = {"component_resnet18": ("microsoft/resnet-18", 224)}
MODELS.update(COMPONENT_MODELS)
COMPONENTS = sorted(set(vision.COMPONENT.values()))
TTA = {"convnext": "convnext_tta", "efficientnet": "efficientnet_tta"}
RECUT = {"convnext": "convnext_recut"}


def classes_of(name):
    return COMPONENTS if name in COMPONENT_MODELS else CLASSES


def label_of(name, row):
    return row["component"] if name in COMPONENT_MODELS else condition_label(row)


def pixels(path, size, flip=False):
    image = Image.open(path).convert("RGB").resize((size, size))
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    array = (np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, size, train, name):
        self.rows, self.size, self.train, self.name = rows, size, train, name

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        flip = self.train and np.random.rand() < 0.5
        return {"pixel_values": pixels(image_file(row), self.size, flip), "labels": classes_of(self.name).index(label_of(self.name, row))}


def balanced(rows, name):
    counts = Counter(label_of(name, r) for r in rows)
    top = max(counts.values())
    out = []
    for r in rows:
        out += [r] * max(1, round(math.sqrt(top / counts[label_of(name, r)])))
    return out


def split_by_tower(rows):
    towers = sorted({r["tower"] for r in rows})
    held = set(towers[::6])
    return [r for r in rows if r["tower"] not in held], [r for r in rows if r["tower"] in held]


def macro_recall(prediction):
    predicted, actual = prediction.predictions.argmax(-1), prediction.label_ids
    recalls = [float((predicted[actual == k] == k).mean()) for k in np.unique(actual)]
    return {"macro_recall": float(np.mean(recalls))}


def train(name, args):
    checkpoint, size = MODELS[name]
    rows = train_images()
    if specialist_group(name):
        rows = [r for r in rows if group_of(r["asset"]) == specialist_group(name)]
    fit, select = split_by_tower(rows)
    print(f"[{name}] training images {len(fit)}, selection images {len(select)}", flush=True)
    model = AutoModelForImageClassification.from_pretrained(checkpoint, num_labels=len(classes_of(name)), ignore_mismatched_sizes=True)
    out = models_dir() / name
    targs = TrainingArguments(output_dir=str(out / "trainer"), num_train_epochs=args.epochs, learning_rate=args.lr,
                              per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
                              eval_strategy="epoch", save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
                              metric_for_best_model="macro_recall", greater_is_better=True, fp16=torch.cuda.is_available(),
                              dataloader_num_workers=args.workers, remove_unused_columns=False, report_to="none",
                              logging_steps=50, disable_tqdm=True, weight_decay=0.05)
    trainer = Trainer(model=model, args=targs, train_dataset=Images(balanced(fit, name), size, True, name),
                      eval_dataset=Images(select or fit[:200], size, False, name), compute_metrics=macro_recall)
    trainer.train()
    trainer.save_model(str(out / "best"))
    shutil.rmtree(out / "trainer", ignore_errors=True)
    return trainer.model.eval(), size


def predict(name, model, size, args, tool=None, path=image_file, flip=False):
    tool = tool or name
    rows = held_images()
    if specialist_group(name):
        rows = [r for r in rows if group_of(r["asset"]) == specialist_group(name)]
    done = done_images(tool)
    rows = [r for r in rows if r["image"] not in done and path(r).exists()]
    run_on = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(run_on)
    times = []
    for r in rows[:30]:
        x = pixels(path(r), size).unsqueeze(0).to(run_on)
        if run_on == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            model(pixel_values=x)
        if run_on == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start) * (2 if flip else 1))
    if times:
        record_timing(tool, statistics.median(times[5:] or times))
    progress = Progress(f"{tool} predict", len(rows), 1000)
    for k in range(0, len(rows), args.batch_size):
        batch = rows[k:k + args.batch_size]
        views = [False, True] if flip else [False]
        probs = 0
        for mirror in views:
            x = torch.stack([pixels(path(r), size, mirror) for r in batch]).to(run_on)
            with torch.no_grad(), torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
                probs = probs + torch.softmax(model(pixel_values=x).logits.float(), -1).cpu().numpy() / len(views)
        key = "component" if name in COMPONENT_MODELS else "probs"
        append(tool, [{"image": r["image"], key: dict(zip(classes_of(name), map(float, p.round(5))))} for r, p in zip(batch, probs)])
        progress.tick(len(batch))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune image-classifier tools with the Hugging Face Trainer and cache their answers on val/test images.")
    parser.add_argument("models", nargs="*", default=[m for m in MODELS if m not in COMPONENT_MODELS])
    parser.add_argument("--epochs", type=float, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    failed = []
    for n, name in enumerate(args.models, 1):
        start = time.time()
        print(f"[train] model {n}/{len(args.models)}: {name}", flush=True)
        try:
            best = models_dir() / name / "best"
            if best.exists():
                model, size = AutoModelForImageClassification.from_pretrained(str(best)).eval(), MODELS[name][1]
                print(f"[{name}] using trained model {best}", flush=True)
            else:
                model, size = train(name, args)
            predict(name, model, size, args)
            if name in TTA:
                predict(name, model, size, args, tool=TTA[name], flip=True)
            if name in RECUT:
                predict(name, model, size, args, tool=RECUT[name], path=recut_file)
            print(f"[{name}] done in {(time.time() - start) / 60:.1f} min", flush=True)
            del model
        except Exception as error:
            failed.append(name)
            print(f"[{name}] FAILED: {error!r}", flush=True)
        torch.cuda.empty_cache()
    print(f"[train] finished, failed models: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
