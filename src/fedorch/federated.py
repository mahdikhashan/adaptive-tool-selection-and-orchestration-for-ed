import functools
import math

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


def drones(rows, k, holdout, rng):
    bucket = tower_buckets(rows, k)
    out = []
    for d in range(k):
        mine = [r for r in rows if bucket[r["tower"]] == d]
        rng.shuffle(mine)
        n = int(round(holdout * len(mine)))
        out.append({"drone": d, "train": mine[n:], "holdout": mine[:n]})
    return [d for d in out if d["train"]]


def trainable(model, freeze, n_last):
    set_trainable(model, freeze, n_last=n_last)
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


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
