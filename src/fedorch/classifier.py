import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageClassification

from fedorch.items import CLASSES

BACKBONE = "microsoft/resnet-50"
SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor(path, train=False):
    image = Image.open(path).convert("RGB").resize((SIZE, SIZE))
    if train and np.random.rand() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    pixels = (np.asarray(image, dtype=np.float32) / 255 - MEAN) / STD
    return torch.from_numpy(pixels.transpose(2, 0, 1).copy())


def new_classifier():
    return AutoModelForImageClassification.from_pretrained(BACKBONE, num_labels=len(CLASSES), ignore_mismatched_sizes=True)


def load_classifier(weights, device):
    model = new_classifier()
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return model.to(device).eval()


def option_probabilities(class_probs, item):
    probs = dict(zip(CLASSES, class_probs))
    if item["task"] == "T2":
        scores = {"good": probs["good"], "defect": 1 - probs["good"]}
    else:
        scores = {label: probs[label] for label in item["options"].values() if label in probs}
    total = sum(scores.values())
    return {label: scores.get(label, 0.0) / total for label in item["options"].values()}
