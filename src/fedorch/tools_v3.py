import json
import os
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


def specialist_group(tool):
    return tool.split("_", 1)[1] if tool.startswith(("spec_", "cspec_")) else None


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
    "smolvlm": ("drone", "SmolVLM-256M vision-language model fine-tuned with LoRA for power-line defects"),
    "smolvlm_base": ("drone", "SmolVLM-256M vision-language model without fine-tuning"),
    "smolvlm_split": ("cloud", "SmolVLM-256M split: vision part on the drone, language part in the cloud"),
    "qwen": ("cloud", "Qwen3-VL-8B large vision-language model in the cloud, general defect question"),
    "qwen_any": ("cloud", "Qwen3-VL-8B in the cloud asked: is there any visible defect?"),
    **{f"qwen_{t}": ("cloud", f"Qwen3-VL-8B in the cloud asked only about {DEFECT_TYPES[t]}") for t in DEFECT_TYPES},
    **{f"head_{t}": ("drone", f"Detector for {DEFECT_TYPES[t]} only, on DINOv2 features") for t in DEFECT_TYPES},
    **{f"spec_{g}": ("drone", f"ResNet-50 specialist trained only on {g} components; works only for those components")
       for g in GROUPS},
    "resnet18": ("drone", "ResNet-18 small image classifier fine-tuned on all components; very fast"),
    "resnet101": ("drone", "ResNet-101 deep image classifier fine-tuned on all components"),
    "vit_base": ("drone", "Vision Transformer ViT-Base fine-tuned on all components; attention over image patches"),
    "deit_small": ("drone", "DeiT-Small data-efficient vision transformer fine-tuned on all components"),
    "swin_tiny": ("drone", "Swin-Tiny shifted-window transformer fine-tuned on all components"),
    "mobilenet_v2": ("drone", "MobileNet-V2 mobile classifier fine-tuned on all components; lowest compute"),
    "efficientnet_b3": ("drone", "EfficientNet-B3 classifier fine-tuned on all components; higher input resolution"),
    "convnext_small": ("drone", "ConvNeXt-Small classifier fine-tuned on all components; larger than ConvNeXt-Tiny"),
    **{f"cspec_{g}": ("drone", f"ConvNeXt-Tiny specialist trained only on {g} components; works only for those components")
       for g in GROUPS},
    "convnext_tta": ("drone", "ConvNeXt-Tiny on the image and its mirror image, mean score"),
    "efficientnet_tta": ("drone", "EfficientNet-B0 on the image and its mirror image, mean score"),
    "convnext_recut": ("drone", "ConvNeXt-Tiny on the component cut again from the full-resolution drone photo"),
    "clip_probe": ("drone", "CLIP image features with a linear defect classifier"),
    **{f"clip_head_{t}": ("drone", f"Detector for {DEFECT_TYPES[t]} only, on CLIP features") for t in DEFECT_TYPES},
    "dinov2b_probe": ("drone", "DINOv2-Base larger self-supervised features with a linear defect classifier"),
    **{f"dinov2b_head_{t}": ("drone", f"Detector for {DEFECT_TYPES[t]} only, on DINOv2-Base features") for t in DEFECT_TYPES},
    "clip_large_zeroshot": ("drone", "CLIP-Large zero-shot: compares the image with text prompts of each condition"),
    "siglip_zeroshot": ("drone", "SigLIP zero-shot: compares the image with text prompts of each condition"),
    "qwen_other": ("cloud", "Qwen3-VL-8B in the cloud asked only about peeling paint or torn parts"),
    "qwen_any_recut": ("cloud", "Qwen3-VL-8B in the cloud asked about any defect on the full-resolution re-cut"),
    "exposure": ("drone", "Exposure check: brightness and colour saturation of the image"),
    "size": ("drone", "Image size check: small crops of far components show less detail"),
}
DRONE = {"resnet", "resnet18", "mobilenet_v2", "efficientnet", "efficientnet_tta", "deit_small", "laya_q4", "smolvlm",
         "quality", "exposure", "size", "head_rust", "head_corrosion"}
TOOLS = {name: ("drone" if name in DRONE else "cloud", description) for name, (_, description) in TOOLS.items()}


BUDGETS = "4,10,30,inf"


def run_suffix():
    name = os.environ.get("TV3_RUN", "")
    return f"_{name}" if name else ""


def excluded_tools():
    return {t for t in os.environ.get("TV3_EXCLUDE", "").split(",") if t}


def selector_tag(model):
    return model.split("/")[-1].lower() + run_suffix()


def default_dtype(model):
    return "float32" if "0.6b" in model.lower() else "float16"


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
