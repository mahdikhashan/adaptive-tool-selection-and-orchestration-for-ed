import json
import math
import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from fedorch import vision
from fedorch.common import results_dir
from fedorch.items import CLASSES
from fedorch.tools_v3 import JETSON

RTT = 0.05
FIELDS = ("component", "defect", "condition")
COMPONENTS = list(dict.fromkeys(vision.COMPONENT.values()))
CONDITIONS = list(vision.CONDITION.values())
CONDITION_OF = dict(zip(CLASSES, [vision.CONDITION[c] for c in CLASSES]))
TASK = "Inspect this power line component in a drone photo. Report its component type, whether it has a defect, and its condition."
SYSTEM = ("You are an inspection agent. At each step, either pick one tool id from the candidate list, or finish with the report.\n\n"
          "Output: brief reasoning, then one JSON object, either\n{\"name\": \"SELECT\", \"ids\": [\"<id>\"]}\nor\n"
          "{\"name\": \"REPORT\", \"component\": \"<component>\", \"defect\": \"yes\" or \"no\", \"condition\": \"<condition>\"}\n\n"
          f"Components: {', '.join(COMPONENTS)}.\nConditions: {', '.join(CONDITIONS)}.")
REPORT_DOC = "Finish now with the three-part report: component type, defect yes or no, condition."


def load_table():
    root = results_dir("tools_v3")
    images = [json.loads(l) for l in (root / "images.jsonl").open()]
    tools = json.loads((root / "tools.json").read_text())
    return images, tools


def truth_of(row):
    return {"component": row["component"], "defect": "yes" if row["defect"] else "no", "condition": row["condition"]}


def own_answer(out):
    answer = {}
    if "component" in out:
        answer["component"] = max(out["component"], key=out["component"].get)
    if "condition" in out:
        top = max(out["condition"], key=out["condition"].get)
        answer["condition"] = CONDITION_OF[top]
        answer["defect"] = "no" if top == "good" else "yes"
    if "defect" in out:
        answer["defect"] = "yes" if out["defect"] >= 0.5 else "no"
    if "kind" in out:
        kind, p = next(iter(out["kind"].items()))
        if p >= 0.5:
            answer["defect"], answer["condition"] = "yes", CONDITION_OF.get(kind, kind)
    return answer


def correct_fields(report, truth):
    return {f for f in FIELDS if report and report.get(f) == truth[f]}


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
    def __init__(self, row, tools, budget, mbps, k=15, retriever=None, max_steps=8):
        self.r, self.tools, self.budget, self.mbps = row, tools, budget, mbps
        self.k, self.retriever, self.max_steps = k, retriever, max_steps
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

    def covered(self, called):
        truth = truth_of(self.r)
        return set().union(*[correct_fields(own_answer(self.r["out"][t]), truth) for t in called]) if called else set()

    def gold(self, actions):
        have = self.covered(self.called)
        if have == set(FIELDS):
            return "report"
        gains = [(len(self.covered(self.called + [t]) - have), -self.cost(t), t) for t in actions if t != "report"]
        best = max(gains, default=(0, 0., "report"))
        return best[2] if best[0] > 0 else "report"

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
        right = correct_fields(report, truth)
        return {"report": report, "truth": truth, "right": sorted(right), "time": self.elapsed, "called": list(self.called),
                "bytes": sum(self.uploaded.values())}

    def substep(self):
        link = f"up, {self.mbps:.1f} Mbit/s" if self.mbps > 0 else "down"
        uploaded = "the image is already uploaded" if "image" in self.uploaded else "the image is not uploaded yet"
        seen = "\n".join(summary(t, self.r["out"][t]) for t in self.called) or "none yet"
        budget = (f"Time budget {self.budget:.0f} s, {max(0., self.budget - self.elapsed):.2f} s left." if math.isfinite(self.budget)
                  else f"No time limit, {self.elapsed:.2f} s used.")
        return f"{budget} Cloud link: {link}; {uploaded}.\nTool results so far:\n{seen}"

    def doc(self, a):
        if a == "report":
            return REPORT_DOC
        where = "Runs on the drone" if self.tools[a]["location"] == "drone" else "Runs in the cloud"
        return f"{documentation(a, self.tools[a])} {where}. Time now: {self.cost(a):.2f} s."


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


def setup():
    images, tools = load_table()
    return dict(images=images, tools=tools, val=[r for r in images if r["split"] == "val"],
                test=[r for r in images if r["split"] == "test"], retriever=Retriever(tools))


def episode(env, row, budget, mbps):
    return Episode(row, env["tools"], budget, mbps, retriever=env["retriever"])
