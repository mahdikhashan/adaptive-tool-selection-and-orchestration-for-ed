"""Protocol v2 preparation: immutable source-group splits and independent audits.

This module has privileged labels ONLY for offline data preparation. The learner
uses Bundle's restricted training-label view, never these preparation functions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import unicodedata

from fedorch.folds import tower_buckets

CLASSES = ["good", "rust", "corrosion", "missing_cap", "bird_nest", "other"]
ALIASES = {"good": "good", "normal": "good", "rust": "rust", "corrosion": "corrosion",
           "corrosão": "corrosion", "missing-cap": "missing_cap", "missing_cap": "missing_cap",
           "missingcap": "missing_cap", "bird-nest": "bird_nest", "bird_nest": "bird_nest",
           "nest": "bird_nest", "other-defect": "other", "other": "other",
           "peeling-paint": "other", "torned-up": "other", "torn-up": "other"}


def canonical_label(item):
    folder = unicodedata.normalize("NFC", Path(item["image"]).parent.name)
    label = ALIASES.get(folder, ALIASES.get(item.get("label")))
    if label is None:
        raise ValueError(f"No six-class condition for {item['item_id']}: folder {folder!r}")
    return label


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def jsonl_write(path, rows):
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stable_uniform(seed, key):
    digest = hashlib.sha256(f"ppo-v2-audit:{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def select_training(items, fraction, seed):
    """Approximate item fraction, stratified tower/class, keeping image groups intact."""
    if fraction == 1:
        return list(items)
    grouped = defaultdict(list)
    for item in items:
        grouped[item["image"]].append(item)
    strata = defaultdict(list)
    for image, group in grouped.items():
        strata[(str(group[0]["tower"]), canonical_label(group[0]))].append(image)
    rng = random.Random(seed)
    allocations = {key: math.floor(len(images) * fraction) for key, images in strata.items()}
    target = max(1, round(len(grouped) * fraction))
    rank = sorted(strata, key=lambda k: (-(len(strata[k]) * fraction - allocations[k]), k))
    for key in rank[:target - sum(allocations.values())]:
        allocations[key] += 1
    chosen = set()
    for key in sorted(strata):
        images = sorted(strata[key])
        rng.shuffle(images)
        chosen.update(images[:allocations[key]])
    return [item for item in items if item["image"] in chosen]


def select_validation(items, fraction, seed):
    """Stratified val subset; towers left with fewer than two source images are dropped,
    because multi-component missions need two distinct sources from one tower."""
    chosen = select_training(items, fraction, seed)
    sources = defaultdict(set)
    for item in chosen:
        sources[str(item["tower"])].add(item["image"])
    return [item for item in chosen if len(sources[str(item["tower"])]) >= 2]


def prepare(items, output, image_root, fraction=.1, seed=0, check_images=True, val_fraction=1.):
    if not 0 < fraction <= 1 or not 0 < val_fraction <= 1:
        raise ValueError("fraction and val_fraction must be in (0,1]")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output must be empty: immutable bundle, choose a new directory")
    all_items = sorted(items, key=lambda i: i["item_id"])
    if len({i["item_id"] for i in all_items}) != len(all_items):
        raise ValueError("Duplicate item IDs")
    for split in ("train", "val", "test"):
        if not any(i["split"] == split for i in all_items):
            raise ValueError(f"Missing split {split}")
    source_splits, tower_splits = defaultdict(set), defaultdict(set)
    for item in all_items:
        source_splits[item["image"]].add(item["split"])
        tower_splits[str(item["tower"])].add(item["split"])
    if any(len(s) != 1 for s in source_splits.values()):
        raise ValueError("Original source image crosses train/val/test")
    if any(len(s) != 1 for s in tower_splits.values()):
        raise ValueError("Tower crosses train/val/test")
    train = [i for i in all_items if i["split"] == "train"]
    clients = tower_buckets(train, 5)
    selected = select_training(train, fraction, seed)
    if val_fraction < 1:
        selected += select_validation([i for i in all_items if i["split"] == "val"], val_fraction, seed)
    selected_ids = {i["item_id"] for i in selected}
    effective = [i for i in all_items if i["split"] == "test" or i["item_id"] in selected_ids]
    public, labels, audits = [], [], {}
    for item in effective:
        image_path = Path(image_root) / item["image"]
        if check_images and not image_path.is_file():
            raise FileNotFoundError(image_path)
        public.append({"item_id": item["item_id"], "split": item["split"], "tower": str(item["tower"]),
                       "source_id": item["image"], "image": item["image"],
                       "client": clients[item["tower"]] if item["split"] == "train" else -1,
                       "task": item["task"]})
        labels.append({"item_id": item["item_id"], "label": canonical_label(item)})
        if item["split"] == "train":
            audits[item["item_id"]] = {"audited": stable_uniform(seed, item["item_id"]) < .05,
                                       "propensity": .05}
    output.mkdir(parents=True, exist_ok=True)
    jsonl_write(output / "items.jsonl", public)
    jsonl_write(output / "labels.jsonl", labels)
    json_write(output / "audits.json", audits)
    (output / "observations.jsonl").touch()
    manifest = {"schema": "fedorch-ppo-v2", "classes": CLASSES, "clients": 5, "seed": seed,
                "fraction": fraction, "val_fraction": val_fraction, "audit_rate": .05, "available_tools": [],
                "cache_complete": False, "controller_only_audit_budget": True,
                "image_root": str(Path(image_root).resolve()), "provenance": {},
                "timing_hardware": None, "counts": {}, "synthetic_fixture": False,
                "preparation_label_access": "privileged tower/class sampling and tool pretraining only",
                "augmentation": False, "enhance": False}
    for split in ("train", "val", "test"):
        rows = [i for i in public if i["split"] == split]
        manifest["counts"][split] = {"items": len(rows), "unique_images": len({i['source_id'] for i in rows}),
                                    "towers": len({i['tower'] for i in rows})}
    manifest["counts"]["audited_train_items"] = sum(a["audited"] for a in audits.values())
    manifest["counts"]["clients"] = {
        str(client): {"items": sum(i["client"] == client for i in public),
                      "towers": len({i["tower"] for i in public if i["client"] == client}),
                      "audited_items": sum(audits[i["item_id"]]["audited"] for i in public if i["client"] == client)}
        for client in range(5)}
    for name in ("items.jsonl", "labels.jsonl", "audits.json"):
        manifest.setdefault("input_sha256", {})[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
    json_write(output / "manifest.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare immutable tower14 S3/A2 bundle; no models/train/downloads")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fraction", type=float, default=.1)
    parser.add_argument("--val-fraction", type=float, default=1.,
                        help="Share of val source images, stratified by tower and class; test stays complete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-image-check", action="store_true", help="Metadata only; cache still requires actual images")
    args = parser.parse_args(argv)
    from fedorch.items import INSPLAD, load_items
    result = prepare(load_items(), args.output, INSPLAD, args.fraction, args.seed, not args.skip_image_check,
                     args.val_fraction)
    print(json.dumps(result["counts"], indent=2))


if __name__ == "__main__":
    main()
