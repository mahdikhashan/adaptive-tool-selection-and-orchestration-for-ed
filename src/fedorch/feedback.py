import collections

import torch

from fedorch.decision import ACTIONS, soft_target

LEARNERS = ["full", "cloud_oracle", "truth_naive", "truth_ipw", "multifid"]


def behaviour(p_model, greedy_fallback, epsilon):
    probs = [epsilon / len(ACTIONS)] * len(ACTIONS)
    probs[ACTIONS.index(p_model if p_model else greedy_fallback)] += 1 - epsilon
    return probs


def reveal(action, rng, audit, review):
    if action == "cloud":
        return "truth" if rng.random() < review else "cloud"
    return "truth" if rng.random() < audit else None


def truth_propensity(p_cloud, audit, review):
    return p_cloud * review + (1 - p_cloud) * audit


class Confusion:
    def __init__(self, prior=1.0):
        self.counts = collections.defaultdict(collections.Counter)
        self.prior = prior

    def add(self, label, cloud_answer):
        self.counts[cloud_answer][label] += 1

    def posterior(self, cloud_answer, options):
        row = self.counts[cloud_answer]
        weights = {y: row[y] + self.prior * (1.0 if y == cloud_answer else 0.1) for y in options}
        total = sum(weights.values())
        return {y: w / total for y, w in weights.items()}


def target_for(learner, item, rows, kind, confusion, temperature):
    cloud_answer = rows["cloud"]["prediction"]
    if learner == "full" or kind == "truth":
        return soft_target(item, rows, temperature)
    if kind != "cloud" or learner in ("truth_naive", "truth_ipw") or cloud_answer == "cannot-judge":
        return None
    if learner == "cloud_oracle":
        return soft_target({**item, "label": cloud_answer}, rows, temperature)
    options = sorted(set(item["options"].values()) - {"cannot-judge"})
    mix = collections.Counter()
    for label, w in confusion.posterior(cloud_answer, options).items():
        for a, v in soft_target({**item, "label": label}, rows, temperature).items():
            mix[a] += w * v
    return dict(mix)


def weight_for(learner, kind, propensity, clip=0.02):
    if learner in ("truth_ipw", "multifid") and kind == "truth":
        return 1.0 / max(propensity, clip)
    return 1.0


def weighted_soft_cross_entropy(logits, target, weight):
    return -(weight * (target * torch.log_softmax(logits, -1)).sum(-1)).sum() / weight.sum()
