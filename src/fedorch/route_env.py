"""Tool-selection environment for one inspection item.

ResNet always runs first. The agent then calls laya, SmolVLM or Qwen (cloud), or writes the report.
The report is the fused defect decision from the tools called so far (logistic fusion that accepts
missing tools), with a label-free threshold. Costs: measured tool times (onboard tools x Jetson factor),
Qwen upload over an LTE trace (no link = Qwen masked). Reward: -(miss 10 / false alarm 1) - lambda * time / budget.
"""
import itertools
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.decision import answer_risk, load_evidence
from fedorch.items import load_items
from fedorch.network import bandwidth_trace, lte_mbps

TOOLS = ["classifier", "laya", "smolvlm_local", "cloud"]
EXTRA = ["laya", "smolvlm_local", "cloud"]
ACTIONS = ["REPORT", "CALL_LAYA", "CALL_SMOLVLM", "CALL_QWEN"]
JETSON, RTT, BAYES = 9.0, .05, 1 / 11
SUBSETS = [tuple(s) for r in range(4) for s in itertools.combinations(EXTRA, r)]


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class Data:
    def __init__(self, laya_run="good300_full"):
        self.cache = load_cache(("classifier", "smolvlm_local", "cloud"))
        self.laya = {p["image"]: p for p in map(json.loads, open(f"results/vision/{laya_run}/predictions.jsonl"))}
        self.evidence = load_evidence()
        self.items = [i for i in load_items() if i["split"] in ("val", "test") and i["image"] in self.laya]
        self.assets = sorted({i["asset"] for i in self.items})
        self.p = {i["item_id"]: {t: self.p_defect(i, t) for t in TOOLS} for i in self.items}
        self.time = {i["item_id"]: self.times(i) for i in self.items}

    @staticmethod
    def defect_option(i):
        return next(o for o in i["options"].values() if o not in ("good", "cannot-judge"))

    def p_defect(self, i, t):
        if t == "laya":
            a = self.laya[i["image"]]["answers"]
            if i["task"] == "T2":
                return a["defect"]
            d, g = a["condition"][vision.CONDITION[self.defect_option(i)]], a["condition"][vision.CONDITION["good"]]
            return d / (d + g)
        r = self.cache[i["item_id"]][t]["probs"]
        good, cannot = r.get("good", 0.), r.get("cannot-judge", 0.)
        return max(0., 1 - good - cannot) / max(1e-6, 1 - cannot)

    def times(self, i):
        rows = self.cache[i["item_id"]]
        return {"classifier": JETSON * rows["classifier"]["compute_ms"] / 1000,
                "laya": JETSON * self.laya[i["image"]]["ms"] / 1000,
                "smolvlm_local": JETSON * rows["smolvlm_local"]["compute_ms"] / 1000,
                "cloud": rows["cloud"]["compute_ms"] / 1000, "cloud_bytes": rows["cloud"]["payload_bytes"]}

    def fusion_x(self, i, called):
        x = []
        for t in TOOLS:
            on = t == "classifier" or t in called
            x += [logit(self.p[i["item_id"]][t]) if on else 0., float(on)]
        return x + [float(i["task"] == "T2")] + [float(i["asset"] == a) for a in self.assets]

    def state_x(self, i):
        e = self.evidence[i["item_id"]]
        return [math.log(e["width"] * e["height"]), math.log(e["blur"] + 1), e["contrast"], e["brightness"],
                float(i["task"] == "T2")] + [float(i["asset"] == a) for a in self.assets]


class Fusion:
    """One logistic model over all tool subsets (missing tools = 0 + flag). Out-of-fold on val towers."""

    def __init__(self, data, C=.03):
        self.data, self.C = data, C

    def fit(self, train):
        X = [self.data.fusion_x(i, s) for i in train for s in SUBSETS]
        y = [i["label"] != "good" for i in train for _ in SUBSETS]
        self.model = LogisticRegression(max_iter=5000, C=self.C).fit(np.array(X), np.array(y))
        self.prior_train = float(np.mean([i["label"] != "good" for i in train]))
        return self

    def table(self, rows):
        X = np.array([self.data.fusion_x(i, s) for i in rows for s in SUBSETS])
        p = self.model.predict_proba(X)[:, 1].reshape(len(rows), len(SUBSETS))
        return {i["item_id"]: dict(zip(SUBSETS, p[n])) for n, i in enumerate(rows)}


def out_of_fold(data, rows, folds=5):
    towers = sorted({i["tower"] for i in rows})
    table = {}
    for k in range(folds):
        fit = [i for i in rows if towers.index(i["tower"]) % folds != k]
        hold = [i for i in rows if towers.index(i["tower"]) % folds == k]
        table.update(Fusion(data).fit(fit).table(hold))
    return table


def em_prior(p, prior_train, steps=200):
    prior = prior_train
    for _ in range(steps):
        a = p * prior / prior_train
        b = (1 - p) * (1 - prior) / (1 - prior_train)
        prior = float((a / (a + b)).mean())
    return prior


def corrected(p, prior_train, prior_new):
    a = p * prior_new / prior_train
    b = (1 - p) * (1 - prior_new) / (1 - prior_train)
    return a / (a + b)


class Episode:
    def __init__(self, data, table, item, budget, mbps, prior_train, prior_new, lambda_time=1.):
        self.d, self.table, self.i, self.budget, self.mbps = data, table, item, budget, mbps
        self.prior_train, self.prior_new, self.lambda_time = prior_train, prior_new, lambda_time
        self.t = data.time[item["item_id"]]
        self.called, self.elapsed, self.bytes = (), self.t["classifier"], 0

    def p_now(self):
        return corrected(self.table[self.i["item_id"]][tuple(sorted(self.called, key=EXTRA.index))], self.prior_train, self.prior_new)

    def cost(self, tool):
        if tool == "cloud":
            if self.mbps <= 0:
                return math.inf
            return self.t["cloud"] + RTT + self.t["cloud_bytes"] * 8 / (self.mbps * 1e6)
        return self.t[tool]

    def mask(self):
        m = [True]
        for tool in EXTRA:
            m.append(tool not in self.called and self.elapsed + self.cost(tool) <= self.budget)
        return np.array(m)

    def features(self):
        p = self.p_now()
        own = []
        for tool in EXTRA:
            on = tool in self.called
            own += [logit(self.d.p[self.i["item_id"]][tool]) if on else 0., float(on)]
        left = max(0., self.budget - self.elapsed)
        return np.array([logit(p), abs(p - BAYES), logit(self.d.p[self.i["item_id"]]["classifier"])] + own +
                        [left / self.budget, math.log(self.budget), math.log1p(self.mbps), float(self.mbps > 0)] +
                        [math.log1p(self.cost(t)) if math.isfinite(self.cost(t)) else 5. for t in EXTRA] +
                        self.d.state_x(self.i), dtype=np.float32)

    def answer(self):
        return self.d.defect_option(self.i) if self.p_now() >= BAYES else "good"

    def step(self, action):
        if action == 0:
            risk = answer_risk(self.answer(), self.i["label"])
            return -risk - self.lambda_time * self.elapsed / self.budget, True, risk
        tool = EXTRA[action - 1]
        dt = self.cost(tool)
        self.elapsed += dt
        self.called = self.called + (tool,)
        if tool == "cloud":
            self.bytes += self.t["cloud_bytes"]
        return 0., False, None


class Network:
    def __init__(self, regime, seed, seconds=7200):
        self.trace = bandwidth_trace(regime, seconds, seed, lte_mbps())
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return self.trace[self.rng.integers(len(self.trace))]
