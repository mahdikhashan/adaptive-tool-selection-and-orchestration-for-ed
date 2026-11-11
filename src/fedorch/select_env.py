import json
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression

from fedorch import vision
from fedorch.common import results_dir
from fedorch.items import CLASSES
from fedorch.tools_v3 import JETSON

RTT = 0.05
COMPONENTS = list(dict.fromkeys(vision.COMPONENT.values()))
CONDITIONS = list(vision.CONDITION.values())
CONDITION_OF = dict(zip(CLASSES, [vision.CONDITION[c] for c in CLASSES]))
TASK = "Inspect this power line component in a drone photo. Report its component type, whether it has a defect, and its condition."
SYSTEM = ("You are an inspection agent. At each step, either pick one tool id from the candidate list, or finish with the report.\n\n"
          "Output: brief reasoning, then one JSON object, either\n{\"name\": \"SELECT\", \"ids\": [\"<id>\"]}\nor\n"
          "{\"name\": \"REPORT\", \"component\": \"<component>\", \"defect\": \"yes\" or \"no\", \"condition\": \"<condition>\"}\n\n"
          f"Components: {', '.join(COMPONENTS)}.\nConditions: {', '.join(CONDITIONS)}.")
REPORT_DOC = "Finish now with the three-part report: component type, defect yes or no, condition."
COMPONENT_WEIGHT, CONDITION_WEIGHT = 0.5, 0.5


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load_table():
    root = results_dir("tools_v3")
    images = [json.loads(l) for l in (root / "images.jsonl").open()]
    tools = json.loads((root / "tools.json").read_text())
    return images, tools


def defect_risk(said_defect, true_defect):
    return 10. if true_defect and not said_defect else (1. if said_defect and not true_defect else 0.)


def report_loss(report, truth):
    if report is None:
        return 10. * truth["defect"] + COMPONENT_WEIGHT + CONDITION_WEIGHT
    return (defect_risk(report["defect"] == "yes", truth["defect"]) + COMPONENT_WEIGHT * (report["component"] != truth["component"])
            + CONDITION_WEIGHT * (report["condition"] != truth["condition"]))


def truth_of(row):
    return {"component": row["component"], "defect": row["defect"], "condition": row["condition"]}


def summary(name, out):
    parts = []
    if "component" in out:
        top = max(out["component"], key=out["component"].get)
        parts.append(f"component {top} {100 * out['component'][top]:.0f}%")
    if "defect" in out:
        parts.append(f"defect {100 * out['defect']:.0f}%")
    if "condition" in out:
        top2 = sorted(out["condition"], key=out["condition"].get, reverse=True)[:2]
        parts.append(", ".join(f"{CONDITION_OF[c]} {100 * out['condition'][c]:.0f}%" for c in top2))
    if "kind" in out:
        parts.extend(f"{CONDITION_OF.get(k, k)} {100 * v:.0f}%" for k, v in out["kind"].items())
    if "score" in out:
        parts.append(f"score {out['score']:+.1f} (0 = average image)")
    return f"{name}: " + "; ".join(parts)


class Teacher:
    def __init__(self, tools, C=0.1):
        self.tools, self.C = tools, C

    def x(self, row, called):
        out = []
        for t in self.tools:
            o = row["out"].get(t) if t in called else None
            cond = o.get("condition") if o else None
            out += [math.log(max(cond.get(c, 0.), 1e-4)) if cond else 0. for c in CLASSES]
            out += [logit(o["defect"]) if o and "defect" in o else 0.]
            out += [logit(next(iter(o["kind"].values()))) if o and "kind" in o else 0.]
            out += [o["score"] if o and "score" in o else 0., float(o is not None)]
        return out

    def fit(self, rows, rng=np.random.default_rng(0), subsets=6):
        X, y = [], []
        for r in rows:
            avail = [t for t in self.tools if t in r["out"]]
            for k in range(subsets):
                called = set(avail) if k == 0 else (set() if k == 1 else {t for t in avail if rng.random() < 0.3})
                X.append(self.x(r, called))
                y.append(CLASSES.index(r["condition_label"]))
        self.model = LogisticRegression(max_iter=4000, C=self.C).fit(np.array(X), np.array(y))
        return self

    def condition_probs(self, rows, called_sets):
        p = self.model.predict_proba(np.array([self.x(r, c) for r, c in zip(rows, called_sets)]))
        full = np.zeros((len(rows), len(CLASSES)))
        full[:, self.model.classes_] = p
        return full

    def report(self, row, called):
        p = self.condition_probs([row], [called])[0]
        defect = 1 - p[CLASSES.index("good")] >= 1 / 11
        condition = CLASSES[int(np.argmax(np.where(np.array(CLASSES) == "good", -1., p)))] if defect else "good"
        comps = [row["out"][t]["component"] for t in called if t in row["out"] and "component" in row["out"][t]]
        component = max(COMPONENTS, key=lambda c: sum(d.get(c, 0.) for d in comps)) if comps else None
        return {"component": component, "defect": "yes" if defect else "no", "condition": CONDITION_OF[condition]}


def out_of_fold(rows, tools, folds=5):
    towers = sorted({r["tower"] for r in rows})
    models = {k: Teacher(tools).fit([r for r in rows if towers.index(r["tower"]) % folds != k]) for k in range(folds)}
    return {r["image"]: models[towers.index(r["tower"]) % folds] for r in rows}


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
    def __init__(self, row, tools, teacher, budget, mbps, lam=0.01, k=15, retriever=None, max_steps=8):
        self.r, self.tools, self.teacher, self.budget, self.mbps = row, tools, teacher, budget, mbps
        self.lam, self.k, self.retriever, self.max_steps = lam, k, retriever, max_steps
        self.called, self.elapsed, self.done, self.final, self.steps, self.uploaded = [], 0., False, None, 0, {}

    def upload_key(self, t):
        return "recut" if "recut" in t else "image"

    def cost(self, t):
        c = self.r["cost"].get(t)
        if c is None:
            return math.inf
        if self.tools[t]["location"] == "drone":
            return JETSON * c["gpu_ms"] / 1000
        if self.mbps <= 0:
            return math.inf
        transfer = 0. if self.upload_key(t) in self.uploaded else c["bytes"] * 8 / (self.mbps * 1e6)
        return c["gpu_ms"] / 1000 + RTT + transfer

    def feasible(self):
        return [t for t in self.tools if t not in self.called and t in self.r["out"] and self.elapsed + self.cost(t) <= self.budget]

    def query(self):
        seen = " ".join(summary(t, self.r["out"][t]) for t in self.called)
        return f"inspect power line component defect condition component type {seen}"

    def candidates(self):
        if self.steps >= self.max_steps - 1:
            return ["report"]
        feasible = self.feasible()
        if self.retriever is not None:
            order = self.retriever.rank(self.query())
            feasible = [t for t in order if t in feasible]
        return feasible[: self.k] + ["report"]

    def teacher_report(self, called=None):
        return self.teacher.report(self.r, set(self.called if called is None else called))

    def values(self, actions):
        truth = truth_of(self.r)
        out = {"report": report_loss(self.teacher_report(), truth) + self.lam * self.elapsed}
        for t in actions:
            if t in self.tools:
                out[t] = report_loss(self.teacher_report(self.called + [t]), truth) + self.lam * (self.elapsed + self.cost(t))
        return {a: out[a] for a in actions}

    def step(self, action):
        self.steps += 1
        if isinstance(action, dict) or action == "report" or action not in self.tools or action in self.called:
            self.done = True
            self.final = action if isinstance(action, dict) else None
            return
        self.elapsed += self.cost(action)
        if self.tools[action]["location"] == "cloud" and self.upload_key(action) not in self.uploaded:
            self.uploaded[self.upload_key(action)] = self.r["cost"][action]["bytes"]
        self.called.append(action)

    def outcome(self):
        truth, report = truth_of(self.r), self.final
        return {"loss": report_loss(report, truth), "time": self.elapsed, "called": list(self.called), "report": report,
                "component_ok": bool(report and report["component"] == truth["component"]),
                "condition_ok": bool(report and report["condition"] == truth["condition"]),
                "defect_said": bool(report and report["defect"] == "yes"), "defect_true": truth["defect"],
                "defect_risk": defect_risk(bool(report and report["defect"] == "yes"), truth["defect"]),
                "true_condition": truth["condition"], "said_condition": report["condition"] if report else None,
                "bytes": sum(self.uploaded.values())}

    def substep(self):
        link = f"up, {self.mbps:.1f} Mbit/s" if self.mbps > 0 else "down"
        uploaded = "the image is already uploaded" if "image" in self.uploaded else "the image is not uploaded yet"
        seen = "\n".join(summary(t, self.r["out"][t]) for t in self.called) or "none yet"
        budget = (f"Time budget {self.budget:.0f} s, {max(0., self.budget - self.elapsed):.2f} s left." if math.isfinite(self.budget)
                  else f"No time limit, {self.elapsed:.2f} s used.")
        return (f"{budget} Cloud link: {link}; {uploaded}. Each second costs {self.lam}. "
                f"A missed defect costs 10, a false alarm 1, a wrong component or condition 0.5.\nTool results so far:\n{seen}")

    def doc(self, a):
        if a == "report":
            return REPORT_DOC
        where = "runs on the drone" if self.tools[a]["location"] == "drone" else "runs in the cloud"
        return f"{documentation(a, self.tools[a])} {where.capitalize()}. Time now: {self.cost(a):.2f} s."


def messages(ep, actions):
    candidates = "\n\n".join(f"[{a}]\n{ep.doc(a)}" for a in actions)
    user = f"User task: {TASK}\n\nSub-step: {ep.substep()}\n\nCandidates ({len(actions)}):\n{candidates}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def render(tokenizer, ep, actions):
    return tokenizer.apply_chat_template(messages(ep, actions), tokenize=False, add_generation_prompt=True, enable_thinking=False)


def select_json(tool):
    return json.dumps({"name": "SELECT", "ids": [tool]})


def report_json(report):
    return json.dumps({"name": "REPORT", "component": report["component"], "defect": report["defect"], "condition": report["condition"]})


def parse(text, actions):
    text = text.split("</think>")[-1]
    for block in re.findall(r"\{.*?\}", text, re.S):
        try:
            obj = json.loads(block)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("name") == "REPORT" and "report" in actions:
            return {"component": str(obj.get("component", "")), "defect": "yes" if str(obj.get("defect", "")).lower() in ("yes", "true", "1") else "no",
                    "condition": str(obj.get("condition", ""))}
        for a in obj.get("ids") or []:
            if a in actions and a != "report":
                return a
    return None


def setup(lam=0.01):
    images, tools = load_table()
    names = list(tools)
    val = [r for r in images if r["split"] == "val"]
    test = [r for r in images if r["split"] == "test"]
    return dict(images=images, tools=tools, val=val, test=test, oof=out_of_fold(val, names), full=Teacher(names).fit(val),
                retriever=Retriever(tools), lam=lam)


def episode(env, row, budget, mbps, train):
    teacher = env["oof"][row["image"]] if train else env["full"]
    return Episode(row, env["tools"], teacher, budget, mbps, env["lam"], retriever=env["retriever"])
