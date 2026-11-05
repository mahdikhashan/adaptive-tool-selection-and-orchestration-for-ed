import functools
import math
import random

import torch
from laya.vlm import collate_vlm, load_vlm, set_trainable
from laya.vlm_train import _forward, _to, make_item, vlm_loss

from fedorch.folds import tower_buckets
from fedorch.items import OPTION_TEXT, image_path

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"
ROUTES = ["classifier", "smolvlm_local"]
ROUTE_TEXT = {"classifier": "accept the answer of the onboard defect classifier",
              "smolvlm_local": "ask the onboard vision-language model"}
ESCALATION_COST = {"classifier": 0.0, "smolvlm_local": 0.1}
RISK = {"miss": 10.0, "false_alarm": 1.0, "abstain": 0.5}
TEMPERATURE = 1.0


def load(device):
    return load_vlm(CHECKPOINT, revision=REVISION, device=device, dtype="fp32")


def question(item):
    asset = item["asset"].replace("-", " ")
    return f"Does the {asset} have a defect?" if item["task"] == "T2" else f"Which condition does the {asset} have?"


def labels_of(item):
    return list(item["options"].values())


def answer_question(item):
    return {"t": "choice", "ins": question(item), "crit": {OPTION_TEXT[label]: None for label in labels_of(item)}}


def route_question(item):
    ins = f"The drone must answer: {question(item)} Which onboard tool gives the correct answer for this image?"
    return {"t": "choice", "ins": ins, "crit": {ROUTE_TEXT[route]: None for route in ROUTES}}


def answer_risk(prediction, label):
    if prediction == "cannot-judge":
        return RISK["abstain"]
    if label == "good":
        return 0.0 if prediction == "good" else RISK["false_alarm"]
    return RISK["miss"] if prediction == "good" else 0.0


def route_target(item, cache):
    scores = [-(answer_risk(cache[item["item_id"]][route]["prediction"], item["label"]) + ESCALATION_COST[route])
              / TEMPERATURE for route in ROUTES]
    top = max(scores)
    weights = [math.exp(s - top) for s in scores]
    return [w / sum(weights) for w in weights]


def examples(items, cache=None):
    out = []
    for item in items:
        state = {"image": str(image_path(item))}
        labels = labels_of(item)
        out.append({"id": item["item_id"], "kind": "answer", "state": state, "q": answer_question(item),
                    "target": [float(label == item["label"]) for label in labels]})
        if cache is not None:
            out.append({"id": item["item_id"], "kind": "route", "state": state, "q": route_question(item),
                        "target": route_target(item, cache)})
    return out


def drones(train_items, k):
    bucket = tower_buckets(train_items, k)
    groups = [[] for _ in range(k)]
    for item in train_items:
        groups[bucket[item["tower"]]].append(item)
    return [g for g in groups if g]


class Items(torch.utils.data.Dataset):
    def __init__(self, processor, rows, shuffle, seed):
        self.processor, self.rows, self.shuffle, self.seed = processor, rows, shuffle, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return make_item(self.processor, self.rows[index], random.Random(self.seed * 1000003 + index),
                         shuffle=self.shuffle)


def loader(agent, rows, batch_size, shuffle, samples=None, seed=0, workers=4):
    sampler = None
    if samples:
        sampler = torch.utils.data.RandomSampler(range(len(rows)), replacement=True, num_samples=samples,
                                                 generator=torch.Generator().manual_seed(seed))
    collate = functools.partial(collate_vlm, pad_id=agent.processor.tokenizer.pad_token_id)
    return torch.utils.data.DataLoader(Items(agent.processor, rows, shuffle, seed), batch_size=batch_size, sampler=sampler,
                                       num_workers=workers, collate_fn=collate)


def trainable(agent, mode, n_last):
    set_trainable(agent.model, mode, n_last=n_last)
    return {name: p for name, p in agent.model.named_parameters() if p.requires_grad}


def train(agent, params, rows, steps, batch_size, lr_head, lr_backbone, seed, workers=4):
    model = agent.model
    encoder = [p for name, p in params.items() if name.startswith("encoder.")]
    head = [p for name, p in params.items() if not name.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": encoder, "lr": lr_backbone}] if encoder else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    model.train()
    losses = []
    for batch in loader(agent, rows, batch_size, True, samples=steps * batch_size, seed=seed, workers=workers):
        b = _to(batch, agent.device, model.encoder.dtype)
        logits, _ = _forward(model, b)
        loss, _ = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


@torch.no_grad()
def predict(agent, rows, batch_size=32, workers=4):
    probs = []
    for batch in loader(agent, rows, batch_size, False, workers=workers):
        b = _to(batch, agent.device, agent.model.encoder.dtype)
        logits, _ = _forward(agent.model, b)
        mask = b["marker_mask"]
        p = torch.softmax(logits.float().masked_fill(~mask, -1e4), -1).cpu()
        probs += [row[: int(m.sum())].tolist() for row, m in zip(p, mask.cpu())]
    return probs


def snapshot(params):
    return {name: p.detach().clone() for name, p in params.items()}


@torch.no_grad()
def assign(params, state):
    for name, p in params.items():
        p.copy_(state[name])


def fedavg(states, weights):
    total = sum(weights)
    return {name: sum(w / total * s[name] for s, w in zip(states, weights)) for name in states[0]}


def megabytes(params, bytes_per_value=2):
    return sum(p.numel() for p in params.values()) * bytes_per_value / 1e6
