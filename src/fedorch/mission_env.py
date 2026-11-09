"""Mission-segment environment for tool selection.

One episode is a segment of consecutive items from one tower. The segment has one time budget
(items x seconds per item), a cap on deferrals to the operator, and a network that follows an LTE trace
in mission time, so outages last over several items. ResNet time for every item is reserved first.
Per item the agent calls laya, SmolVLM or Qwen, reports, or defers. Reward per item: minus the risk.
"""
import collections
import math

import numpy as np

from fedorch.route_env import Episode


def tower_runs(items):
    runs = collections.defaultdict(list)
    for i in items:
        runs[i["tower"]].append(i)
    return list(runs.values())


def fixed_segments(items, length):
    return [run[k:k + length] for run in tower_runs(items) for k in range(0, len(run), length)]


def random_segment(runs, length, rng):
    run = runs[rng.choice(len(runs), p=np.array([len(r) for r in runs]) / sum(len(r) for r in runs))]
    start = rng.integers(max(1, len(run) - length + 1))
    return run[start:start + length]


class Segment:
    def __init__(self, data, table, items, per_item, cap, trace, start, prior_train, prior_new):
        self.d, self.table, self.items, self.trace, self.start = data, table, items, trace, start
        self.prior_train, self.prior_new = prior_train, prior_new
        self.budget = per_item * len(items)
        self.cap_total = math.floor(cap * len(items) + 1e-9)
        self.n, self.elapsed, self.deferred, self.bytes = 0, 0., 0, 0
        self.log, self.recent = [], collections.deque(maxlen=5)
        self.open_item()

    def reserve(self):
        return sum(self.d.time[i["item_id"]]["classifier"] for i in self.items[self.n + 1:])

    def mbps(self):
        return self.trace[int(self.start + self.elapsed) % len(self.trace)]

    def open_item(self):
        available = max(0., self.budget - self.elapsed - self.reserve())
        self.ep = Episode(self.d, self.table, self.items[self.n], max(available, 1e-3), self.mbps(),
                          self.prior_train, self.prior_new, 0., True)
        self.first_risk = self.ep.report_risk()

    def link_age(self):
        now = int(self.start + self.elapsed)
        up = self.trace[now % len(self.trace)] > 0
        age = 0
        while age < 120 and (self.trace[(now - age - 1) % len(self.trace)] > 0) == up:
            age += 1
        return age

    def mask(self):
        m = self.ep.mask()
        m[4] = self.deferred < self.cap_total
        return m

    def features(self):
        left = len(self.items) - self.n
        time_left = max(0., self.budget - self.elapsed)
        recent = float(np.mean(self.recent)) if self.recent else self.first_risk
        return np.concatenate([self.ep.features(), np.array([
            left / len(self.items), time_left / self.budget, math.log1p(time_left / left),
            float(self.cap_total - self.deferred), (self.cap_total - self.deferred) / left, float(self.mask()[4]),
            math.log1p(self.link_age()), recent, self.first_risk - recent], dtype=np.float32)])

    def step(self, action):
        reward, done, risk = self.ep.step(action)
        if not done:
            return reward, False, None
        self.elapsed += self.ep.elapsed
        self.bytes += self.ep.bytes
        self.deferred += action == 4
        self.recent.append(self.first_risk)
        self.log.append({"risk": risk, "time": self.ep.elapsed, "called": self.ep.called, "deferred": action == 4})
        self.n += 1
        if self.n == len(self.items):
            return reward, True, risk
        self.open_item()
        return reward, False, risk
