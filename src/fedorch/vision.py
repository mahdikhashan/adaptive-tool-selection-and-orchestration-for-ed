import functools
import random

import torch
from laya.vlm import collate_vlm, load_vlm
from laya.vlm_train import _forward, _to, make_item

from fedorch.items import OPTION_TEXT, image_path

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
RISK = {"miss": 10.0, "false_alarm": 1.0, "abstain": 0.5}
QUESTION = {"T2": "Does the power line component in this image have a defect?",
            "T3": "Which condition does the power line component in this image have?"}


def load(device):
    return load_vlm(CHECKPOINT, revision=REVISION, device=device, dtype="fp32")


def labels_of(item):
    return list(item["options"].values())


def example(item):
    labels = labels_of(item)
    return {"state": {"image": str(image_path(item))},
            "q": {"t": "choice", "ins": QUESTION[item["task"]], "crit": {OPTION_TEXT[label]: None for label in labels}},
            "target": [float(label == item["label"]) for label in labels]}


def answer_risk(prediction, label):
    if prediction == "cannot-judge":
        return RISK["abstain"]
    if label == "good":
        return 0.0 if prediction == "good" else RISK["false_alarm"]
    return RISK["miss"] if prediction == "good" else 0.0


class Items(torch.utils.data.Dataset):
    def __init__(self, processor, rows):
        self.processor, self.rows = processor, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return make_item(self.processor, self.rows[index], random.Random(0), shuffle=False)


@torch.no_grad()
def predict(agent, items, batch_size=32, workers=4):
    rows = [example(item) for item in items]
    collate = functools.partial(collate_vlm, pad_id=agent.processor.tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(Items(agent.processor, rows), batch_size=batch_size, num_workers=workers,
                                         collate_fn=collate)
    probs = []
    for batch in loader:
        b = _to(batch, agent.device, agent.model.encoder.dtype)
        logits, _ = _forward(agent.model, b)
        mask = b["marker_mask"]
        p = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1).cpu()
        probs += [row[: int(m.sum())].tolist() for row, m in zip(p, mask.cpu())]
    return probs
