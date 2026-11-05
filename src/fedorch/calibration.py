import math

import torch

from fedorch.features import TOOLS


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def platt(scores, labels):
    x = torch.tensor([logit(s) for s in scores])
    y = torch.tensor(labels, dtype=torch.float32)
    a, b = torch.ones(1, requires_grad=True), torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([a, b], max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * x + b, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(a.detach()), float(b.detach())


def calibrate(q, cache, fit_ids):
    params = {}
    for tool in TOOLS:
        params[tool] = platt([q[i][tool] for i in fit_ids], [float(cache[i][tool]["correct"]) for i in fit_ids])
    sigmoid = lambda z: 1 / (1 + math.exp(-z))
    calibrated = {i: {t: sigmoid(params[t][0] * logit(v[t]) + params[t][1]) for t in TOOLS} for i, v in q.items()}
    return calibrated, params


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, 1) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def ece(scores, labels, bins=10):
    total, error = len(scores), 0.0
    for k in range(bins):
        members = [(s, l) for s, l in zip(scores, labels) if k / bins <= s < (k + 1) / bins or (k == bins - 1 and s == 1.0)]
        if members:
            error += len(members) / total * abs(sum(s for s, _ in members) / len(members) - sum(l for _, l in members) / len(members))
    return error


def quality(q, cache, ids):
    report = {}
    for tool in TOOLS:
        scores, labels = [q[i][tool] for i in ids], [int(cache[i][tool]["correct"]) for i in ids]
        report[tool] = {"auroc": round(auroc(scores, labels), 4), "ece": round(ece(scores, labels), 4)}
    return report
