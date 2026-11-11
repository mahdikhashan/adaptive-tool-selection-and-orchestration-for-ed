import json
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

from fedorch.common import results_dir
from fedorch.decision import answer_risk
from fedorch.tools_v3 import ANSWERS, SPECIAL_ACTIONS

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


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def documentation(name, tool):
    return f"{name} ({tool['location']}): {tool['description']}"


class Retriever:
    def __init__(self, tools, tau=0.0):
        self.names, self.tau = list(tools), tau
        self.bm25 = BM25Okapi([words(name.replace("_", " ") + " " + documentation(name, tools[name])) for name in self.names], k1=1.8, b=1.0)

    def rank(self, query):
        scores = self.bm25.get_scores(words(query))
        return [self.names[k] for k in np.argsort(-scores, kind="stable") if scores[k] >= self.tau]


class Episode:
    def __init__(self, item, tools, fusion, prior, new_prior, budget, mbps, lam=0.01, k=15, retriever=None, estimate=True):
        self.i, self.tools, self.fusion, self.prior, self.new_prior = item, tools, fusion, prior, new_prior
        self.budget, self.mbps, self.lam, self.k, self.retriever, self.estimate = budget, mbps, lam, k, retriever, estimate
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
        return feasible[: self.k] + list(ANSWERS)

    def fused_answer(self):
        return "answer_defect" if self.q() >= BAYES else "answer_good"

    def answer_risk(self, action):
        if action == "defer":
            return 0.5
        return answer_risk(self.i["defect_option"] if action == "answer_defect" else "good", self.i["label"])

    def values(self, actions, true_answers=False):
        now = self.lam * self.elapsed / self.budget
        evidence = self.fused_answer()
        out = {"defer": 0.5 + now}
        for a in ("answer_defect", "answer_good"):
            if true_answers:
                out[a] = self.answer_risk(a) + now
            else:
                out[a] = self.answer_risk(evidence) + now + (0. if a == evidence else 1.)
        tools = [a for a in actions if a in self.tools]
        if tools:
            p = self.fusion.prob([self.i] * len(tools), [set(self.called) | {t} for t in tools])
            for t, v in zip(tools, corrected(p, self.prior, self.new_prior)):
                out[t] = risk_of(self.i, float(v)) + self.lam * (self.elapsed + self.cost(t)) / self.budget
        return {a: out[a] for a in actions}

    def step(self, action):
        self.action_log.append(action)
        if action in ANSWERS or action not in self.tools or action in self.called:
            self.done = True
            self.final = action if action in ANSWERS else "defer"
            return
        self.elapsed += self.cost(action)
        self.called.append(action)

    def outcome(self):
        return {"risk": self.answer_risk(self.final), "time": self.elapsed, "deferred": self.final == "defer",
                "answer": self.final, "called": list(self.called),
                "bytes": sum(self.i["cost"][t].get("bytes", 0) for t in self.called if self.tools[t]["location"] != "drone")}

    def task(self):
        return f"Inspect this {self.i['component']} in a drone photo and answer: {self.i['question']}"

    def substep(self):
        link = f"up, {self.mbps:.1f} Mbit/s" if self.mbps > 0 else "down"
        seen = "; ".join(f"{t} says defect {100 * self.i['p'][t]:.0f}%" for t in self.called)
        return (f"Choose the next step. Time left: {max(0., self.budget - self.elapsed):.2f} s of {self.budget:.2f} s. "
                f"Cloud link: {link}. Tools already called: {seen}. "
                + (f"Combined defect estimate of these tools: {100 * self.q():.0f}%. " if self.estimate else "")
                + "Call another tool, or finish with an answer. A missed defect costs 10, a false alarm 1, sending to the operator 0.5.")

    def doc(self, a):
        if a in SPECIAL_ACTIONS:
            return SPECIAL_ACTIONS[a]
        return f"{documentation(a, self.tools[a])} Time: {self.cost(a):.2f} s."



SYSTEM = ("You are a tool-search agent in the SELECT phase. Pick id(s) from the candidate list that solve the sub-step.\n\n"
          "Output: brief reasoning, then a JSON object\n{\"name\": \"SELECT\", \"ids\": [\"<id>\", ...]}")


def messages(ep, actions):
    candidates = "\n\n".join(f"[{a}]\n{ep.doc(a)}" for a in actions)
    user = f"User task: {ep.task()}\n\nSub-step: {ep.substep()}\n\nCandidates ({len(actions)}):\n{candidates}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render(tokenizer, ep, actions):
    return tokenizer.apply_chat_template(messages(ep, actions), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def answer(action):
    return json.dumps({"name": "SELECT", "ids": [action]})


def parse(text, actions):
    text = text.split("</think>")[-1]
    for block in re.findall(r"\{.*?\}", text, re.S):
        try:
            ids = json.loads(block).get("ids") or []
        except (ValueError, AttributeError):
            continue
        for a in ids:
            if a in actions:
                return a
    return None


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


def episode(env, item, budget, mbps, train, estimate=True):
    fusion = env["oof"][item["item_id"]] if train else env["full"]
    new_prior = env["prior"] if train else env["test_prior"]
    return Episode(item, env["tools"], fusion, env["prior"], new_prior, budget, mbps, env["lam"], retriever=env["retriever"],
                   estimate=estimate)
