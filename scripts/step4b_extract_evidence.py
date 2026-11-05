import json

import numpy as np
import torch
from PIL import Image

from fedorch.cache import load_cache
from fedorch.classifier import load_classifier, to_tensor
from fedorch.common import device, results_dir, runs_dir
from fedorch.items import CLASSES, image_path, load_items


def image_stats(path):
    image = Image.open(path).convert("RGB")
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    laplacian = 4 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    return {"width": image.width, "height": image.height, "blur": float(laplacian.var()),
            "brightness": float(gray.mean() / 255), "contrast": float(gray.std() / 255)}


def main():
    run_on, items, cache = device(), load_items(), load_cache(["classifier"])
    by_model = {}
    for item in items:
        by_model.setdefault(cache[item["item_id"]]["classifier"]["tool_model"], []).append(item)
    with (results_dir("step4b") / "evidence.jsonl").open("w") as file:
        for model_name, group in sorted(by_model.items()):
            model = load_classifier(runs_dir("step2") / f"classifier_{model_name}.pt", run_on)
            for item in group:
                pixels = to_tensor(image_path(item))[None].to(run_on)
                with torch.no_grad(), torch.autocast(run_on, dtype=torch.float16, enabled=run_on == "cuda"):
                    probs = torch.softmax(model(pixel_values=pixels).logits[0].float(), -1).tolist()
                row = {"item_id": item["item_id"], "classifier_model": model_name,
                       "class_probs": dict(zip(CLASSES, probs)), **image_stats(image_path(item))}
                file.write(json.dumps(row) + "\n")
            print(f"{model_name}: {len(group)} items")


if __name__ == "__main__":
    main()
