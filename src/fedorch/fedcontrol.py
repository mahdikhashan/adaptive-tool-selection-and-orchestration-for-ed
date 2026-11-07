import copy

import torch

from fedorch.folds import tower_buckets


def regions(train, k):
    bucket = tower_buckets(train, k)
    return [[i for i in train if bucket[i["tower"]] == r] for r in range(k)]


def federated_norm(xs):
    n = sum(len(x) for x in xs)
    mean = sum(x.sum(0) for x in xs) / n
    var = sum(((x - mean) ** 2).sum(0) for x in xs) / (n - 1)
    return mean, var.sqrt().clamp(min=1e-6)


def update_kb(model, bytes_per_value=4):
    return sum(p.numel() for p in model.parameters()) * bytes_per_value / 1000


def local_steps(model, loss_fn, x, y, steps, prox=0.0, anchor=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        if prox:
            loss = loss + prox / 2 * sum(((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor))
        loss.backward()
        optimizer.step()
    return model


def average(states, weights):
    total = sum(weights)
    return {k: sum(s[k] * w for s, w in zip(states, weights)) / total for k in states[0]}


def fedavg(model, loss_fn, clients, x_val, y_val, rounds=300, steps=10, prox=0.0, patience=30):
    best, best_state, waited, history = float("inf"), None, 0, []
    for _ in range(rounds):
        anchor = [p.detach().clone() for p in model.parameters()]
        states = [local_steps(copy.deepcopy(model), loss_fn, x, y, steps, prox, anchor).state_dict() for x, y in clients]
        model.load_state_dict(average(states, [len(x) for x, _ in clients]))
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        history.append(round(val_loss, 5))
        if val_loss < best:
            best, best_state, waited = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waited += 1
            if waited > patience:
                break
    model.load_state_dict(best_state)
    return {"val_loss": best, "rounds_run": len(history), "best_round": history.index(min(history)) + 1}
