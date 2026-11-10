import argparse
import statistics
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from fedorch.classifier import load_classifier, to_tensor
from fedorch.items import CLASSES
from fedorch.tools_v3 import (DEFECT_TYPES, Progress, append, classifier_checkpoint, condition_label, done_images,
                              held_images, image_file, models_dir, recut_file, record_timing, train_images)

CLIP_TEXT = {"good": "a photo of a {} in good condition with no defect", "rust": "a photo of a rusty {}",
             "corrosion": "a photo of a corroded {}", "missing-cap": "a photo of a {} with a missing cap",
             "nest": "a photo of a {} with a bird nest", "other-defect": "a photo of a damaged {} with peeling paint or a torn part"}
RUN_ON = "cuda" if torch.cuda.is_available() else "cpu"


def timed(fn, n=30):
    times = []
    for _ in range(n):
        if RUN_ON == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if RUN_ON == "cuda":
            torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times[5:])


def dinov2_features(rows, batch_size):
    path = models_dir() / "dinov2_features.npz"
    cache = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    known = dict(zip(cache.get("images", []), cache.get("features", [])))
    todo = [r for r in rows if r["image"] not in known]
    if todo:
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        model = AutoModel.from_pretrained("facebook/dinov2-small").to(RUN_ON).eval()
        progress = Progress("dinov2 features", len(todo), 2000)
        for k in range(0, len(todo), batch_size):
            batch = todo[k:k + batch_size]
            inputs = processor(images=[Image.open(image_file(r)).convert("RGB") for r in batch], return_tensors="pt").to(RUN_ON)
            with torch.no_grad():
                out = model(**inputs).last_hidden_state[:, 0].float().cpu().numpy()
            known.update({r["image"]: f for r, f in zip(batch, out)})
            progress.tick(len(batch))
        one = processor(images=[Image.open(image_file(rows[0])).convert("RGB")], return_tensors="pt").to(RUN_ON)
        ms = timed(lambda: model(**one))
        record_timing("dinov2_probe", ms)
        for t in DEFECT_TYPES:
            record_timing(f"head_{t}", ms)
        np.savez(path, images=np.array(list(known)), features=np.stack(list(known.values())))
    return known


def probes(batch_size):
    train, held = train_images(), held_images()
    features = dinov2_features(train + held, batch_size)
    x = np.stack([features[r["image"]] for r in train])
    y = np.array([CLASSES.index(condition_label(r)) for r in train])
    xh = np.stack([features[r["image"]] for r in held])
    if not done_images("dinov2_probe"):
        model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(x, y)
        p = model.predict_proba(xh)
        append("dinov2_probe", [{"image": r["image"], "probs": {CLASSES[c]: float(p[i, k]) for k, c in enumerate(model.classes_)}}
                                for i, r in enumerate(held)])
        print("dinov2_probe cached", flush=True)
    for t in DEFECT_TYPES:
        tool = f"head_{t}"
        if done_images(tool):
            continue
        yt = (y == CLASSES.index(t)).astype(int)
        model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(x, yt)
        p = model.predict_proba(xh)[:, 1]
        append(tool, [{"image": r["image"], "defect": {t: float(v)}} for r, v in zip(held, p)])
        print(f"{tool} cached", flush=True)


def clip_zeroshot(batch_size):
    rows = [r for r in held_images() if r["image"] not in done_images("clip_zeroshot")]
    if not rows:
        return
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(RUN_ON).eval()
    progress = Progress("clip", len(rows), 2000)
    for k in range(0, len(rows), batch_size):
        batch = rows[k:k + batch_size]
        texts = [CLIP_TEXT[c].format(r["component"]) for r in batch for c in CLASSES]
        inputs = processor(text=texts, images=[Image.open(image_file(r)).convert("RGB") for r in batch], return_tensors="pt", padding=True).to(RUN_ON)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image.float()
        own = torch.stack([logits[k, k * len(CLASSES):(k + 1) * len(CLASSES)] for k in range(len(batch))])
        probs = torch.softmax(own, -1).cpu().numpy()
        append("clip_zeroshot", [{"image": r["image"], "probs": dict(zip(CLASSES, map(float, p)))} for r, p in zip(batch, probs)])
        progress.tick(len(batch))
    one = processor(text=[CLIP_TEXT[c].format("shackle") for c in CLASSES], images=[Image.open(image_file(rows[0])).convert("RGB")], return_tensors="pt", padding=True).to(RUN_ON)
    record_timing("clip_zeroshot", timed(lambda: model(**one)))


def quality():
    rows = [r for r in held_images() if r["image"] not in done_images("quality")]
    progress, out = Progress("quality", len(rows), 2000), []
    for r in rows:
        gray = cv2.imread(str(image_file(r)), cv2.IMREAD_GRAYSCALE)
        out.append({"image": r["image"], "score": float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())), "contrast": float(gray.std())})
        progress.tick()
        if len(out) == 500:
            append("quality", out)
            out = []
    append("quality", out)
    if rows:
        gray = cv2.imread(str(image_file(rows[0])), cv2.IMREAD_GRAYSCALE)
        record_timing("quality", timed(lambda: cv2.Laplacian(gray, cv2.CV_64F).var()))


def resnet_recut(batch_size):
    rows = [r for r in held_images() if recut_file(r).exists() and r["image"] not in done_images("resnet_recut")]
    if not rows:
        print("resnet_recut: no re-cut images", flush=True)
        return
    model = load_classifier(classifier_checkpoint(), RUN_ON)
    progress = Progress("resnet_recut", len(rows), 1000)
    for k in range(0, len(rows), batch_size):
        batch = rows[k:k + batch_size]
        x = torch.stack([to_tensor(recut_file(r)) for r in batch]).to(RUN_ON)
        with torch.no_grad():
            probs = torch.softmax(model(pixel_values=x).logits.float(), -1).cpu().numpy()
        append("resnet_recut", [{"image": r["image"], "probs": dict(zip(CLASSES, map(float, p)))} for r, p in zip(batch, probs)])
        progress.tick(len(batch))
    one = to_tensor(recut_file(rows[0])).unsqueeze(0).to(RUN_ON)
    record_timing("resnet_recut", timed(lambda: model(pixel_values=one)))


def main():
    parser = argparse.ArgumentParser(description="DINOv2 probes and defect heads, CLIP zero-shot, image quality, ResNet on re-cut images.")
    parser.add_argument("parts", nargs="*", default=["probes", "clip", "quality", "recut"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    jobs = {"probes": lambda: probes(args.batch_size), "clip": lambda: clip_zeroshot(args.batch_size),
            "quality": quality, "recut": lambda: resnet_recut(args.batch_size)}
    for part in args.parts:
        start = time.time()
        jobs[part]()
        print(f"[{part}] done in {(time.time() - start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
