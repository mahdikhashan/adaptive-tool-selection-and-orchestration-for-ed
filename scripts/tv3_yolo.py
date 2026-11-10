import argparse
import os
import statistics
from pathlib import Path

from ultralytics import YOLO

from fedorch.common import WORK
from fedorch.items import CLASSES
from fedorch.tools_v3 import Progress, append, condition_label, done_images, held_images, image_file, models_dir, record_timing, train_images


def folder_dataset(root):
    rows = train_images()
    towers = sorted({r["tower"] for r in rows})
    held = set(towers[::6])
    for r in rows:
        part = "val" if r["tower"] in held else "train"
        target = root / part / condition_label(r) / r["image"].replace("/", "__")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(image_file(r), target)
    for part in ("train", "val"):
        for c in CLASSES:
            (root / part / c).mkdir(parents=True, exist_ok=True)
    return root


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO classification model with Ultralytics and cache its answers.")
    parser.add_argument("--start", default=os.environ.get("TV3_YOLO_START", str(WORK / "models" / "ppo_v2" / "yolo26n-cls.pt")))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()
    out = models_dir() / "yolo_cls"
    best = out / "run" / "weights" / "best.pt"
    if not best.exists():
        start = args.start if Path(args.start).exists() else "yolo11n-cls.pt"
        print(f"[yolo] start checkpoint {start}", flush=True)
        data = folder_dataset(WORK / "data" / "tools_v3_yolo")
        YOLO(start).train(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=0,
                          project=str(out), name="run", exist_ok=True, plots=False, verbose=False)
    model = YOLO(str(best))
    names = model.names
    rows = [r for r in held_images() if r["image"] not in done_images("yolo_cls")]
    progress, speeds = Progress("yolo predict", len(rows), 1000), []
    for k in range(0, len(rows), args.batch):
        batch = rows[k:k + args.batch]
        results = model.predict(source=[str(image_file(r)) for r in batch], imgsz=args.imgsz, device=0, verbose=False)
        append("yolo_cls", [{"image": r["image"], "probs": {names[i]: float(p) for i, p in enumerate(res.probs.data.float().cpu().tolist())}}
                            for r, res in zip(batch, results)])
        speeds += [res.speed["inference"] for res in results]
        progress.tick(len(batch))
    if speeds:
        record_timing("yolo_cls", statistics.median(speeds))


if __name__ == "__main__":
    main()
