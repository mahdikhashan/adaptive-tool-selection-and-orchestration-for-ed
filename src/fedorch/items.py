import json
import os
from pathlib import Path

from fedorch.labels import item_image, parse_options

DATA = Path(os.environ.get("FEDORCH_DATA", Path(__file__).resolve().parents[2] / "data"))
INSPLAD = Path(os.environ.get("INSPLAD_DIR", DATA / "insplad"))
SPLITS = DATA / "elecvqa-bench/data/benchmark/split_comparison/tower14"

MERGED = {"torn-up": "other-defect", "peeling-paint": "other-defect"}
CLASSES = ["good", "rust", "corrosion", "missing-cap", "nest", "other-defect"]
OPTION_TEXT = {
    "good": "good, no defect",
    "defect": "defect",
    "rust": "rust",
    "corrosion": "corrosion",
    "missing-cap": "missing cap",
    "nest": "bird nest",
    "other-defect": "other defect (torn part or peeling paint)",
    "cannot-judge": "cannot judge",
}


def asset_of(image):
    parts = image.split("/")
    return parts[2] if parts[1] == "defect_supervised" else parts[1]


def prompt_of(item):
    asset = item["asset"].replace("-", " ")
    question = (f"Does the {asset} have a defect?" if item["task"] == "T2"
                else f"Which condition does the {asset} have?")
    options = "\n".join(f"{letter}. {OPTION_TEXT[label]}" for letter, label in item["options"].items())
    return f"This image shows a {asset} on a power line. {question}\nOptions:\n{options}\nAnswer with one letter."


def parse(split, index, raw):
    options = {letter: MERGED.get(label, label) for letter, label in parse_options(raw["messages"][0]["content"][-1]["text"]).items()}
    answer = raw["messages"][1]["content"].strip()
    image = item_image(raw)
    item = {
        "item_id": f"{split}-{index:05d}",
        "split": split,
        "tower": raw["_tower"],
        "parent_id": raw["_parent_id"],
        "image": image,
        "asset": asset_of(image),
        "task": "T2" if "defect" in options.values() else "T3",
        "options": options,
        "label": options[answer],
    }
    item["prompt"] = prompt_of(item)
    return item


def load_items(splits=("train", "val", "test")):
    items = []
    for split in splits:
        with (SPLITS / f"{split}.jsonl").open() as file:
            items += [parse(split, index, json.loads(line)) for index, line in enumerate(file)]
    return items


def image_path(item):
    return INSPLAD / item["image"]
