import json
import time
from pathlib import Path

from fedorch import vision
from fedorch.common import WORK, results_dir, runs_dir
from fedorch.items import CLASSES, INSPLAD, load_items

GROUPS = {"glass": ("glass",), "polymer": ("polymer",), "hardware": ("damper", "spacer", "plate"),
          "yoke": ("yoke",), "rodgrip": ("lightning", "vari-grip")}
DEFECT_TYPES = {"rust": "rust", "corrosion": "corrosion", "missing-cap": "missing cap", "nest": "bird nest"}
JETSON = 9.0
EDGE_RTT, EDGE_MBPS = 0.02, 50.0


def group_of(asset):
    return next(g for g, prefixes in GROUPS.items() if asset.startswith(prefixes))


TOOLS = {
    "resnet": ("drone", "ResNet-50 image classifier fine-tuned on all components; fast first check"),
    "resnet_crops_mean": ("drone", "ResNet-50 on five overlapping crops and the full image, mean score"),
    "resnet_crops_max": ("drone", "ResNet-50 on five overlapping crops and the full image, highest defect score"),
    "resnet_recut": ("drone", "ResNet-50 on the component cut again from the full-resolution drone photo with context"),
    "convnext": ("drone", "ConvNeXt-Tiny image classifier fine-tuned on all components; different architecture from ResNet"),
    "efficientnet": ("drone", "EfficientNet-B0 image classifier fine-tuned on all components; small and fast"),
    "dinov2_probe": ("drone", "DINOv2 self-supervised image features with a linear defect classifier"),
    "clip_zeroshot": ("drone", "CLIP zero-shot: compares the image with text prompts of each condition; no training"),
    "yolo_cls": ("drone", "YOLO classification model fine-tuned on the six conditions"),
    "quality": ("drone", "Image quality check: blur and contrast; tells if the image is too poor to judge"),
    "laya": ("drone", "laya-vision decision model, full precision; answers component, defect and condition"),
    "laya_q4": ("drone", "laya-vision decision model in 4-bit; cheaper memory, similar answers"),
    "laya_recut": ("drone", "laya-vision on the component cut again from the full-resolution drone photo"),
    "laya_edge": ("edge", "laya-vision on the ground-station edge server; image sent over the local link"),
    "smolvlm": ("drone", "SmolVLM-256M vision-language model fine-tuned with LoRA for power-line defects"),
    "smolvlm_base": ("drone", "SmolVLM-256M vision-language model without fine-tuning"),
    "smolvlm_split": ("cloud", "SmolVLM-256M split: vision part on the drone, language part in the cloud"),
    "qwen": ("cloud", "Qwen3-VL-8B large vision-language model in the cloud, general defect question"),
    "qwen_any": ("cloud", "Qwen3-VL-8B in the cloud asked: is there any visible defect?"),
    **{f"qwen_{t}": ("cloud", f"Qwen3-VL-8B in the cloud asked only about {DEFECT_TYPES[t]}") for t in DEFECT_TYPES},
    **{f"head_{t}": ("drone", f"Detector for {DEFECT_TYPES[t]} only, on DINOv2 features") for t in DEFECT_TYPES},
    **{f"spec_{g}": ("drone", f"ResNet-50 specialist trained only on {g} components; works only for those components")
       for g in GROUPS},
}
SPECIAL_ACTIONS = {"report": "Stop and report the current answer now", "defer": "Send the image to the human operator"}


def cache_dir():
    path = results_dir("tools_v3") / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir():
    path = WORK / "models" / "tools_v3"
    path.mkdir(parents=True, exist_ok=True)
    return path


def held_images():
    return [r for r in vision.images(load_items()) if r["split"] in ("val", "test")]


def train_images():
    return [r for r in vision.images(load_items()) if r["split"] == "train"]


def condition_label(row):
    return {v: k for k, v in vision.CONDITION.items()}[row["condition"]]


def image_file(row):
    return INSPLAD / row["image"]


def recut_file(row):
    return WORK / "data" / "recut" / row["image"]


def done_images(tool):
    path = cache_dir() / f"{tool}.jsonl"
    return {json.loads(line)["image"] for line in path.open()} if path.exists() else set()


def append(tool, rows):
    with (cache_dir() / f"{tool}.jsonl").open("a") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def load_tool(tool):
    path = cache_dir() / f"{tool}.jsonl"
    return {r["image"]: r for r in map(json.loads, path.open())} if path.exists() else {}


class Progress:
    def __init__(self, name, total, every=500):
        self.name, self.total, self.every, self.start, self.n = name, total, every, time.time(), 0

    def tick(self, k=1):
        self.n += k
        if self.n % self.every < k or self.n == self.total:
            rate = self.n / max(1e-6, time.time() - self.start)
            left = (self.total - self.n) / max(rate, 1e-6)
            print(f"[{self.name}] {self.n}/{self.total} ({100 * self.n / max(1, self.total):.0f}%), {rate:.1f}/s, about {left / 60:.1f} min left", flush=True)


def classifier_checkpoint():
    return runs_dir("step2") / "classifier_full.pt"


def laya_checkpoint():
    return WORK / "models" / "vision" / "good300_full"


def class_probs(probs):
    total = sum(probs.get(c, 0.) for c in CLASSES) or 1.
    return {c: probs.get(c, 0.) / total for c in CLASSES}


def timing_path():
    return results_dir("tools_v3") / "timing.json"


def record_timing(tool, ms, payload_bytes=0):
    path = timing_path()
    data = json.loads(path.read_text()) if path.exists() else {}
    data[tool] = {"gpu_ms": round(float(ms), 2), "payload_bytes": int(payload_bytes)}
    path.write_text(json.dumps(data, indent=1))
