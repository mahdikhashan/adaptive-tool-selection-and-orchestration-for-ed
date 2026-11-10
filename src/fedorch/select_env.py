import json
import math

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.linear_model import LogisticRegression

from fedorch.common import results_dir
from fedorch.decision import answer_risk
from fedorch.tools_v3 import EDGE_MBPS, EDGE_RTT, JETSON, SPECIAL_ACTIONS, TOOLS

BAYES, RTT = 1 / 11, 0.05


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load_table():
    root = results_dir("tools_v3")
    items = [json.loads(l) for l in (root / "items.jsonl").open()]
    tools = json.loads((root / "tools.json").read_text())
    return items, tools


class Fusion:
    def __init__(self, tools, assets, C=0.05):
        self.tools, self.assets, self.C = tools, assets, C

    def x(self, item, called):
        out = []
        for t in self.tools:
            p = item["p"].get(t)
            on = t in called and p is not None
            out += [logit(p) if on else 0., float(on)]
        return out + [float(item["task"] == "T2")] + [float(item["asset"] == a) for a in self.assets]

    def fit(self, items, rng=np.random.default_rng(0), subsets=6):
        X, y = [], []
        for i in items:
            avail = [t for t in self.tools if i["p"].get(t) is not None]
            for k in range(subsets):
                called = set(avail) if k == 0 else {"resnet"} | {t for t in avail if rng.random() < 0.4}
                X.append(self.x(i, called))
                y.append(i["defect"])
        self.model = LogisticRegression(max_iter=4000, C=self.C).fit(np.array(X), np.array(y))
        self.prior = float(np.mean([i["defect"] for i in items]))
        return self

    def prob(self, items, called_sets):
        return self.model.predict_proba(np.array([self.x(i, c) for i, c in zip(items, called_sets)]))[:, 1]


def out_of_fold(items, tools, assets, folds=5):
    towers = sorted({i["tower"] for i in items})
    models = {}
    for k in range(folds):
        fit = [i for i in items if towers.index(i["tower"]) % folds != k]
        models[k] = Fusion(tools, assets).fit(fit)
    return {i["item_id"]: models[towers.index(i["tower"]) % folds] for i in items}


def em_prior(p, prior, steps=200):
    pi = prior
    for _ in range(steps):
        pi = float(corrected(p, prior, pi).mean())
    return pi


def corrected(p, prior, new):
    a = p * new / prior
    b = (1 - p) * (1 - new) / (1 - prior)
    return a / (a + b)


def risk_of(item, q):
    return answer_risk(item["defect_option"] if q >= BAYES else "good", item["label"])


class Retriever:
    def __init__(self, tools):
        self.names = list(tools)
        self.bm25 = BM25Okapi([(n.replace("_", " ") + " " + tools[n]["description"]).lower().split() for n in self.names])

    def rank(self, query):
        scores = self.bm25.get_scores(query.lower().split())
        return [self.names[k] for k in np.argsort(-scores, kind="stable")]


class Episode:
    def __init__(self, item, tools, fusion, prior, new_prior, budget, mbps, lam=0.01, k=8, retriever=None):
        self.i, self.tools, self.fusion, self.prior, self.new_prior = item, tools, fusion, prior, new_prior
        self.budget, self.mbps, self.lam, self.k, self.retriever = budget, mbps, lam, k, retriever
        self.called, self.elapsed, self.done, self.action_log = ["resnet"], self.cost("resnet"), False, []

    def cost(self, t):
        c = self.i["cost"].get(t)
        if c is None:
            return math.inf
        if self.tools[t]["location"] == "cloud":
            return math.inf if self.mbps <= 0 else c["compute_s"] + RTT + c["bytes"] * 8 / (self.mbps * 1e6)
        return c["s"]

    def q(self, called=None):
        called = self.called if called is None else called
        return float(corrected(self.fusion.prob([self.i], [set(called)])[0], self.prior, self.new_prior))

    def feasible(self):
        return [t for t in self.tools if t not in self.called and self.i["p"].get(t) is not None
                and self.elapsed + self.cost(t) <= self.budget]

    def candidates(self):
        feasible = self.feasible()
        if self.retriever is not None:
            order = self.retriever.rank(f"{self.i['component']} {self.i['question']} defect")
            feasible = [t for t in order if t in feasible]
        return feasible[: self.k] + ["report", "defer"]

    def values(self, actions):
        q_now = self.q()
        out = {"report": risk_of(self.i, q_now) + self.lam * self.elapsed / self.budget,
               "defer": 0.5 + self.lam * self.elapsed / self.budget}
        tools = [a for a in actions if a in self.tools]
        if tools:
            p = self.fusion.prob([self.i] * len(tools), [set(self.called) | {t} for t in tools])
            for t, v in zip(tools, corrected(p, self.prior, self.new_prior)):
                out[t] = risk_of(self.i, float(v)) + self.lam * (self.elapsed + self.cost(t)) / self.budget
        return {a: out[a] for a in actions}

    def step(self, action):
        self.action_log.append(action)
        if action in ("report", "defer") or action not in self.tools or action in self.called:
            self.done = True
            self.final = "defer" if action == "defer" else "report"
            return
        self.elapsed += self.cost(action)
        self.called.append(action)
        if not self.feasible():
            self.done, self.final = True, "report"

    def outcome(self):
        if self.final == "defer":
            risk = 0.5
        else:
            risk = risk_of(self.i, self.q())
        return {"risk": risk, "time": self.elapsed, "deferred": self.final == "defer", "called": list(self.called),
                "bytes": sum(self.i["cost"][t].get("bytes", 0) for t in self.called if self.tools[t]["location"] != "drone")}

    def describe(self):
        link = f"up, {self.mbps:.1f} Mbit/s" if self.mbps > 0 else "down"
        seen = "; ".join(f"{t}: defect {100 * self.i['p'][t]:.0f}%" for t in self.called)
        return (f"Component: {self.i['component']}. Question: {self.i['question']}\n"
                f"Time left: {max(0., self.budget - self.elapsed):.2f} s of {self.budget:.2f} s. Cloud link: {link}.\n"
                f"Tools called: {seen}.\nCurrent defect estimate: {100 * self.q():.0f}%.")

    def option_text(self, a):
        if a in SPECIAL_ACTIONS:
            return f"{a}: {SPECIAL_ACTIONS[a]}"
        return f"{a} ({self.cost(a):.2f} s): {self.tools[a]['description']}"


def prompt(ep, actions):
    lines = "\n".join(f"- {ep.option_text(a)}" for a in actions)
    return (f"You choose the next inspection step for a drone. A missed defect costs 10, a false alarm 1, "
            f"sending the image to the operator 0.5. Time costs a little.\n{ep.describe()}\nCandidates:\n{lines}\n"
            f"Answer with one candidate name only.\nAnswer:")


def parse(text, actions):
    words = text.strip().lower().replace(":", " ").replace(",", " ").split()
    for w in words[:3]:
        if w in actions:
            return w
    return next((a for a in sorted(actions, key=len, reverse=True) if a in text.lower()), None)


def setup(lam=0.01):
    items, tools = load_table()
    names = list(tools)
    assets = sorted({i["asset"] for i in items})
    val = [i for i in items if i["split"] == "val"]
    test = [i for i in items if i["split"] == "test"]
    oof = out_of_fold(val, names, assets)
    full = Fusion(names, assets).fit(val)
    prior = float(np.mean([i["defect"] for i in val]))
    p_all = full.prob(test, [set(names)] * len(test))
    new_prior = em_prior(p_all, prior)
    return dict(items=items, tools=tools, val=val, test=test, oof=oof, full=full, prior=prior, test_prior=new_prior,
                retriever=Retriever(tools), lam=lam)


def episode(env, item, budget, mbps, train):
    fusion = env["oof"][item["item_id"]] if train else env["full"]
    new_prior = env["prior"] if train else env["test_prior"]
    return Episode(item, env["tools"], fusion, env["prior"], new_prior, budget, mbps, env["lam"], retriever=env["retriever"])
