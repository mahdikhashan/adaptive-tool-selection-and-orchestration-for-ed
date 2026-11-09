"""Lower-shackle rust: contact sheets of ResNet misses and hits, and ResNet test-time zoom.

1. Contact sheets: polymer-insulator-lower-shackle defect images in train (out-of-fold step-2 answers)
   and val, split into ResNet misses and hits.
2. Zoom: the fine-tuned ResNet on every val and test image, full view plus five overlapping 2/3 crops
   (four corners, centre). Saves the six 6-class probability vectors per image for later fusion.
"""
import json
import time

import numpy as np
import torch
from PIL import Image, ImageDraw

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.classifier import MEAN, SIZE, STD, load_classifier
from fedorch.common import results_dir, runs_dir
from fedorch.items import INSPLAD, load_items

TARGET = "polymer-insulator-lower-shackle"
BOXES = [None, (0, 0, 2, 2), (1, 0, 3, 2), (0, 1, 2, 3), (1, 1, 3, 3), (.5, .5, 2.5, 2.5)]


def crops(image):
    w, h = image.size
    out = []
    for box in BOXES:
        out.append(image if box is None else image.crop(tuple(int(v * s / 3) for v, s in zip(box, (w, h, w, h)))))
    return out


def tensor(images):
    arrays = [(np.asarray(i.resize((SIZE, SIZE)), dtype=np.float32) / 255 - MEAN) / STD for i in images]
    return torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2).copy())


def contact_sheet(paths, title, out, size=160, columns=6):
    paths = paths[:24]
    rows = max(1, (len(paths) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * size, rows * size + 24), "white")
    ImageDraw.Draw(sheet).text((6, 6), title, fill="black")
    for n, (path, caption) in enumerate(paths):
        image = Image.open(INSPLAD / path).convert("RGB")
        image.thumbnail((size - 4, size - 22))
        x, y = (n % columns) * size, (n // columns) * size + 24
        sheet.paste(image, (x + 2, y + 2))
        ImageDraw.Draw(sheet).text((x + 4, y + size - 18), caption, fill="black")
    sheet.save(out)


def main():
    out = results_dir("zoom")
    items = load_items()
    cache = load_cache(("classifier",))
    for split in ("train", "val"):
        rows = [i for i in items if i["split"] == split and i["asset"] == TARGET and i["label"] not in ("good", "cannot-judge")]
        seen, missed, hit = set(), [], []
        for i in rows:
            if i["image"] in seen:
                continue
            seen.add(i["image"])
            r = cache[i["item_id"]]["classifier"]
            caption = f"{i['label']} P(good)={r['probs'].get('good', 0):.2f}"
            (missed if r["prediction"] == "good" else hit).append((i["image"], caption))
        contact_sheet(missed, f"{split}: {TARGET} defects ResNet MISSED ({len(missed)} images)", out / f"sheet_{split}_missed.jpg")
        contact_sheet(hit, f"{split}: {TARGET} defects ResNet CAUGHT ({len(hit)} images)", out / f"sheet_{split}_caught.jpg")
        print(f"{split}: {len(missed)} missed, {len(hit)} caught", flush=True)
    model = load_classifier(runs_dir("step2") / "classifier_full.pt", "cuda")
    rows = [r for r in vision.images(items) if r["split"] in ("val", "test")]
    path = out / "resnet_views.jsonl"
    done = {json.loads(l)["image"] for l in path.open()} if path.exists() else set()
    start = time.time()
    with path.open("a") as f:
        for n, row in enumerate([r for r in rows if r["image"] not in done], 1):
            views = crops(Image.open(INSPLAD / row["image"]).convert("RGB"))
            with torch.no_grad():
                probs = torch.softmax(model(pixel_values=tensor(views).cuda()).logits.float(), -1).cpu().numpy()
            f.write(json.dumps({"image": row["image"], "split": row["split"], "tower": row["tower"], "asset": row["asset"],
                                "defect": row["defect"], "probs": probs.round(5).tolist()}) + "\n")
            if n % 1000 == 0:
                print(f"zoom: {n} images, {(time.time() - start) / n * 1000:.1f} ms each", flush=True)
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
