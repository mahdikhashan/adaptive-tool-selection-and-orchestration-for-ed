import functools
import math

import numpy as np
import torch
from laya.vlm import set_trainable
from laya.vlm_train import ItemStream, _collate_train, _forward, _to, collect_logits, jsonl_example, vlm_loss

from fedorch import vision
from fedorch.folds import tower_buckets


def examples(rows):
    return [ex for row in rows for ex in (jsonl_example(r, "/", "fl") for r in vision.labelled(row)) if ex]


def split_towers(rows, group):
    if group == "all":
        return rows
    bucket = tower_buckets(rows, 2)
    keep = 0 if group == "server" else 1
    return [r for r in rows if bucket[r["tower"]] == keep]


COMPONENT_DRONES = [["shackle"], ["vari-grip clamp"], ["glass insulator", "spacer"], ["lightning rod suspension", "plate"],
                    ["yoke suspension", "yoke"], ["polymer insulator", "stockbridge damper", "preformed damper"]]
PARTITIONS = ["tower", "iid", "component", "dirichlet"]


def assignment(rows, k, partition, alpha, rng):
    if partition == "tower":
        bucket = tower_buckets(rows, k)
        return [bucket[r["tower"]] for r in rows]
    if partition == "iid":
        order = list(range(len(rows)))
        rng.shuffle(order)
        where = [0] * len(rows)
        for i, j in enumerate(order):
            where[j] = i % k
        return where
    if partition == "component":
        if k != len(COMPONENT_DRONES):
            raise ValueError(f"the component partition needs {len(COMPONENT_DRONES)} drones")
        drone_of = {c: d for d, group in enumerate(COMPONENT_DRONES) for c in group}
        return [drone_of[r["component"]] for r in rows]
    gen = np.random.default_rng(rng.randrange(2 ** 31))
    where = [0] * len(rows)
    for condition in vision.CONDITION.values():
        idx = [i for i, r in enumerate(rows) if r["condition"] == condition]
        gen.shuffle(idx)
        cuts = (np.cumsum(gen.dirichlet([alpha] * k)) * len(idx)).astype(int)[:-1]
        for d, part in enumerate(np.split(np.array(idx, dtype=int), cuts)):
            for i in part:
                where[i] = d
    return where


def drones(rows, k, holdout, rng, partition="tower", alpha=0.3, budget=0):
    where = assignment(rows, k, partition, alpha, rng)
    out = []
    for d in range(k):
        mine = [r for r, w in zip(rows, where) if w == d]
        rng.shuffle(mine)
        if budget:
            mine = mine[:budget]
        n = int(round(holdout * len(mine)))
        out.append({"drone": d, "train": mine[n:], "holdout": mine[:n]})
    return [d for d in out if d["train"]]


def merged(groups):
    return [{"drone": 0, "train": [r for g in groups for r in g["train"]], "holdout": [r for g in groups for r in g["holdout"]]}]


def shares_of(rows):
    counts = np.array([sum(r["condition"] == c for r in rows) for c in vision.CONDITION.values()], dtype=float)
    return counts / max(1.0, counts.sum())


def divergence(rows, everything):
    p, q = shares_of(rows), shares_of(everything)
    m = (p + q) / 2
    kl = lambda a: float(np.sum([x * np.log2(x / y) for x, y in zip(a, m) if x > 0]))
    return round(0.5 * kl(p) + 0.5 * kl(q), 4)


LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SHARES = {"full": ("full", None), "layers": ("last_n", 20), "last_n": ("last_n", None), "head": ("head", None)}


class LoRALinear(torch.nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    @torch.no_grad()
    def merged(self):
        self.base.weight += (self.lora_b @ self.lora_a).to(self.base.weight.dtype) * self.scale
        return self.base


def lora_parents(model):
    for layer in model.encoder.text_model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name in LORA_TARGETS:
                if hasattr(parent, name):
                    yield parent, name


def add_lora(model, rank):
    for parent, name in lora_parents(model):
        setattr(parent, name, LoRALinear(getattr(parent, name), rank, 2 * rank))


def merge_lora(model):
    for parent, name in lora_parents(model):
        module = getattr(parent, name)
        if isinstance(module, LoRALinear):
            setattr(parent, name, module.merged())


def trainable(model, share, n_last, rank=8):
    if share == "lora":
        set_trainable(model, "head")
        add_lora(model, rank)
    else:
        mode, n = SHARES[share]
        set_trainable(model, mode, n_last=n or n_last)
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def int8(delta):
    scale = delta.abs().max() / 127
    return delta if scale == 0 else (delta / scale).round().clamp(-127, 127) * scale


def quantized(local, state):
    return {name: state[name] + int8(local[name] - state[name]) for name in local}


def snapshot(params):
    return {name: p.detach().to("cpu", copy=True) for name, p in params.items()}


@torch.no_grad()
def assign(params, state):
    for name, p in params.items():
        p.copy_(state[name].to(p.device))


def lr_factor(round_index, rounds):
    return 0.1 + 0.45 * (1 + math.cos(math.pi * round_index / max(1, rounds)))


def local_train(agent, params, rows, steps, batch_size, lr_head, lr_backbone, prox, anchor, seed, workers):
    model = agent.model
    head = [p for n, p in params.items() if not n.startswith("encoder.")]
    encoder = [p for n, p in params.items() if n.startswith("encoder.")]
    groups = [{"params": head, "lr": lr_head}] + ([{"params": encoder, "lr": lr_backbone}] if encoder else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    stream = ItemStream(agent.processor, examples(rows), seed)
    collate = functools.partial(_collate_train, pad_id=agent.processor.tokenizer.pad_token_id)
    loader = torch.utils.data.DataLoader(stream, batch_size=batch_size, num_workers=workers, collate_fn=collate)
    near = [(p, anchor[n].to(p.device)) for n, p in params.items()] if prox else []
    model.train()
    losses = []
    for _, batch in zip(range(steps), loader):
        b = _to(batch, agent.device, model.encoder.dtype)
        logits, _ = _forward(model, b)
        loss, _ = vlm_loss(logits, b["target"], b["qtype"], b["marker_mask"])
        if prox:
            loss = loss + prox / 2 * sum(((p - a) ** 2).sum() for p, a in near)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(params.values()), 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    return losses


def steps_for(rows, passes, batch_size):
    return max(1, math.ceil(passes * 3 * len(rows) / batch_size))


def holdout_logits(agent, rows):
    return collect_logits(agent.model, agent.processor, examples(rows), batch_size=16) if rows else []


def megabytes(params, bytes_per_value=2):
    return round(sum(p.numel() for p in params.values()) * bytes_per_value / 1e6, 1)


def seed_for(base, round_index, drone):
    return base * 100003 + round_index * 101 + drone
