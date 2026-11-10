import argparse
import statistics
import time

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from transformers import AutoImageProcessor, AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

from fedorch.classifier import load_classifier, to_tensor
from fedorch.items import CLASSES
from fedorch.tools_v3 import (DEFECT_TYPES, Progress, append, classifier_checkpoint, condition_label, done_images,
                              held_images, image_file, models_dir, recut_file, record_timing, train_images)

CLIP_TEXT = {"good": "a photo of a {} in good condition with no defect", "rust": "a photo of a rusty {}",
             "corrosion": "a photo of a corroded {}", "missing-cap": "a photo of a {} with a missing cap",
             "nest": "a photo of a {} with a bird nest", "other-defect": "a photo of a damaged {} with peeling paint or a torn part"}
RUN_ON = "cuda" if torch.cuda.is_available() else "cpu"
FEATURES = {"dinov2": ("facebook/dinov2-small", "dinov2_probe", "head"),
            "dinov2b": ("facebook/dinov2-base", "dinov2b_probe", "dinov2b_head"),
            "clip": ("openai/clip-vit-base-patch32", "clip_probe", "clip_head")}
ZERO_SHOT = {"clip_zeroshot": "openai/clip-vit-base-patch32", "clip_large_zeroshot": "openai/clip-vit-large-patch14",
             "siglip_zeroshot": "google/siglip-base-patch16-224"}


def sync():
    if RUN_ON == "cuda":
        torch.cuda.synchronize()


def timed(fn, n=30):
    times = []
    for _ in range(n):
        sync()
        start = time.perf_counter()
        fn()
        sync()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times[5:])


def images(rows):
    return [Image.open(image_file(r)).convert("RGB") for r in rows]


def extractor(checkpoint):
    if "clip" in checkpoint:
        processor, model = CLIPProcessor.from_pretrained(checkpoint), CLIPModel.from_pretrained(checkpoint).to(RUN_ON).eval()

        def run(pixel_values):
            return model.visual_projection(model.vision_model(pixel_values=pixel_values).pooler_output)
    else:
        processor, model = AutoImageProcessor.from_pretrained(checkpoint), AutoModel.from_pretrained(checkpoint).to(RUN_ON).eval()

        def run(pixel_values):
            return model(pixel_values=pixel_values).last_hidden_state[:, 0]
    return processor, run


def features(key, rows, batch_size):
    checkpoint, probe_tool, head_prefix = FEATURES[key]
    path = models_dir() / f"{key}_features.npz"
    cache = dict(np.load(path, allow_pickle=True)) if path.exists() else {}
    known = dict(zip(cache.get("images", []), cache.get("features", [])))
    todo = [r for r in rows if r["image"] not in known]
    if todo:
        processor, run = extractor(checkpoint)
        progress = Progress(f"{key} features", len(todo), 2000)
        for k in range(0, len(todo), batch_size):
            batch = todo[k:k + batch_size]
            pixels = processor(images=images(batch), return_tensors="pt")["pixel_values"].to(RUN_ON)
            with torch.no_grad():
                known.update({r["image"]: f for r, f in zip(batch, run(pixels).float().cpu().numpy())})
            progress.tick(len(batch))
        one = processor(images=images(rows[:1]), return_tensors="pt")["pixel_values"].to(RUN_ON)
        with torch.no_grad():
            ms = timed(lambda: run(one))
        for tool in [probe_tool] + [f"{head_prefix}_{t}" for t in DEFECT_TYPES]:
            record_timing(tool, ms)
        np.savez(path, images=np.array(list(known)), features=np.stack(list(known.values())))
    return known


def probes(key, batch_size):
    _, probe_tool, head_prefix = FEATURES[key]
    train, held = train_images(), held_images()
    known = features(key, train + held, batch_size)
    x = np.stack([known[r["image"]] for r in train])
    y = np.array([CLASSES.index(condition_label(r)) for r in train])
    xh = np.stack([known[r["image"]] for r in held])
    if not done_images(probe_tool):
        model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(x, y)
        p = model.predict_proba(xh)
        append(probe_tool, [{"image": r["image"], "probs": {CLASSES[c]: float(p[i, k]) for k, c in enumerate(model.classes_)}}
                            for i, r in enumerate(held)])
        print(f"{probe_tool} cached", flush=True)
    for t in DEFECT_TYPES:
        tool = f"{head_prefix}_{t}"
        if done_images(tool):
            continue
        model = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(x, (y == CLASSES.index(t)).astype(int))
        append(tool, [{"image": r["image"], "defect": {t: float(v)}} for r, v in zip(held, model.predict_proba(xh)[:, 1])])
        print(f"{tool} cached", flush=True)


def zero_shot(tool, batch_size):
    rows = [r for r in held_images() if r["image"] not in done_images(tool)]
    if not rows:
        return
    checkpoint = ZERO_SHOT[tool]
    if "siglip" in checkpoint:
        processor, model = AutoProcessor.from_pretrained(checkpoint), AutoModel.from_pretrained(checkpoint).to(RUN_ON).eval()
        text_kwargs = {"padding": "max_length"}
    else:
        processor, model = CLIPProcessor.from_pretrained(checkpoint), CLIPModel.from_pretrained(checkpoint).to(RUN_ON).eval()
        text_kwargs = {"padding": True}

    def inputs_for(batch):
        texts = [CLIP_TEXT[c].format(r["component"]) for r in batch for c in CLASSES]
        return processor(text=texts, images=images(batch), return_tensors="pt", **text_kwargs).to(RUN_ON)

    progress = Progress(tool, len(rows), 2000)
    for k in range(0, len(rows), batch_size):
        batch = rows[k:k + batch_size]
        with torch.no_grad():
            logits = model(**inputs_for(batch)).logits_per_image.float()
        own = torch.stack([logits[j, j * len(CLASSES):(j + 1) * len(CLASSES)] for j in range(len(batch))])
        probs = torch.softmax(own, -1).cpu().numpy()
        append(tool, [{"image": r["image"], "probs": dict(zip(CLASSES, map(float, p)))} for r, p in zip(batch, probs)])
        progress.tick(len(batch))
    one = inputs_for(rows[:1])
    with torch.no_grad():
        record_timing(tool, timed(lambda: model(**one)))


def image_checks():
    checks = {"quality": lambda gray, hsv: (float(np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var())), float(gray.std())),
              "exposure": lambda gray, hsv: (float(abs(gray.mean() - 128) / 128 - hsv[..., 1].mean() / 255), float(gray.mean())),
              "size": lambda gray, hsv: (float(np.log(gray.shape[0] * gray.shape[1])), float(gray.shape[0]))}
    done = {t: done_images(t) for t in checks}
    rows = [r for r in held_images() if any(r["image"] not in done[t] for t in checks)]
    progress, out = Progress("image checks", len(rows), 2000), {t: [] for t in checks}
    for r in rows:
        image = cv2.imread(str(image_file(r)))
        gray, hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for t, fn in checks.items():
            if r["image"] not in done[t]:
                score, extra = fn(gray, hsv)
                out[t].append({"image": r["image"], "score": score, "extra": extra})
        progress.tick()
    for t, fn in checks.items():
        append(t, out[t])
        if rows:
            image = cv2.imread(str(image_file(rows[0])))
            gray, hsv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            record_timing(t, timed(lambda: fn(gray, hsv)))


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
    with torch.no_grad():
        record_timing("resnet_recut", timed(lambda: model(pixel_values=one)))


def main():
    parser = argparse.ArgumentParser(description="Feature probes and defect detectors (DINOv2, DINOv2-Base, CLIP), zero-shot models (CLIP, CLIP-Large, SigLIP), image checks, ResNet on re-cut images.")
    parser.add_argument("parts", nargs="*", default=[*FEATURES, *ZERO_SHOT, "checks", "recut"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    failed = []
    for part in args.parts:
        start = time.time()
        try:
            if part in FEATURES:
                probes(part, args.batch_size)
            elif part in ZERO_SHOT:
                zero_shot(part, args.batch_size)
            elif part == "checks":
                image_checks()
            else:
                resnet_recut(args.batch_size)
            print(f"[{part}] done in {(time.time() - start) / 60:.1f} min", flush=True)
        except Exception as error:
            failed.append(part)
            print(f"[{part}] FAILED: {error!r}", flush=True)
        torch.cuda.empty_cache()
    print(f"[features] finished, failed parts: {failed or 'none'}", flush=True)


if __name__ == "__main__":
    main()
