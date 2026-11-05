import json
import math
import random

from fedorch.common import results_dir
from fedorch.controller import expected_latency
from fedorch.features import numeric
from fedorch.items import CLASSES
from fedorch.simulate import RISK

ACTIONS = ["accept", "smolvlm_local", "cloud"]
ACTION_TOOL = {"accept": "classifier", "smolvlm_local": "smolvlm_local", "cloud": "cloud"}
ACTION_TEXT = {
    "accept": "accept the answer of the onboard defect classifier",
    "smolvlm_local": "ask the onboard vision-language model",
    "cloud": "ask the cloud vision-language model",
}
ESCALATION_COST = {"accept": 0.0, "smolvlm_local": 0.1, "cloud": 0.2}
GRID = [(1.0, 1), (1.0, 5), (0.25, 1), (0.25, 5)]
QUESTION = "Which action gives the lowest risk for this frame?"
QUANTILES = [0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]


def answer_risk(answer, label):
    if label != "good" and answer == "good":
        return RISK["miss"]
    if label == "good" and answer not in ("good", "cannot-judge"):
        return RISK["false_alarm"]
    return RISK["abstain"] if answer == "cannot-judge" else 0.0


def action_risks(item, rows):
    return {a: answer_risk(rows[ACTION_TOOL[a]]["prediction"], item["label"]) + ESCALATION_COST[a] for a in ACTIONS}


def informative(item, rows):
    risks = action_risks(item, rows)
    return max(risks.values()) - min(risks.values()) > max(ESCALATION_COST.values())


def training_items(items, cache, oversample):
    return [i for i in items for _ in range(oversample if informative(i, cache[i["item_id"]]) else 1)]


def soft_target(item, rows, temperature):
    risks = action_risks(item, rows)
    weights = {a: math.exp(-r / temperature) for a, r in risks.items()}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def load_evidence():
    with (results_dir("step4b") / "evidence.jsonl").open() as file:
        return {row["item_id"]: row for row in map(json.loads, file)}


def evidence_text(item, row, ev):
    task = "yes/no defect question" if item["task"] == "T2" else "defect type question"
    classes = ", ".join(f"{c} {ev['class_probs'][c]:.3f}" for c in CLASSES)
    options = ", ".join(f"{label} {p:.3f}" for label, p in sorted(row["probs"].items(), key=lambda pair: -pair[1]))
    good = min(max(ev["class_probs"]["good"], 1e-6), 1 - 1e-6)
    return (f"Task: {task}. Component: {item['asset'].replace('-', ' ')}. "
            f"Classifier classes: {classes}. Answer options: {options}. Log-odds of good: {math.log(good / (1 - good)):.2f}. "
            f"Image: {ev['width']} x {ev['height']} px, blur {ev['blur']:.0f}, brightness {ev['brightness']:.2f}, contrast {ev['contrast']:.2f}.")


def numeric_v2(item, row, ev, assets):
    good = min(max(ev["class_probs"]["good"], 1e-6), 1 - 1e-6)
    return (numeric(item, row, assets) + [ev["class_probs"][c] for c in CLASSES]
            + [math.log(good / (1 - good)), math.log(ev["width"]), math.log(ev["height"]), math.log(ev["blur"] + 1),
               ev["brightness"], ev["contrast"]])


def config_name(model, temperature, oversample, fraction=1.0, seed=0):
    name = f"{model}_T{temperature}_O{oversample}"
    return name if fraction == 1.0 and seed == 0 else f"{name}_f{fraction}_s{seed}"


def subsample(items, fraction, seed):
    return items if fraction >= 1 else random.Random(seed).sample(items, max(1, round(fraction * len(items))))


def write_p(name, p_by_item, step="step4b"):
    with (results_dir(step) / f"p_{name}.jsonl").open("w") as file:
        for item_id, p in p_by_item.items():
            file.write(json.dumps({"item_id": item_id, "p": p}) + "\n")


def read_p(path):
    return {row["item_id"]: row["p"] for row in map(json.loads, path.open())}


def gap(probs, tool):
    return probs[tool] - probs["accept"]


def thresholds(p, item_ids):
    gaps = sorted(max(gap(p[i], "smolvlm_local"), gap(p[i], "cloud")) for i in item_ids)
    return [gaps[min(len(gaps) - 1, int(q * len(gaps)))] if q < 1 else gaps[-1] + 1e-9 for q in QUANTILES]


def choice_rule(p, cache, threshold):
    def policy(item, history, state):
        if not history:
            return "classifier"
        if len(history) > 1:
            return None
        probs = p[item["item_id"]]
        best, best_gap = None, threshold
        for tool in ("smolvlm_local", "cloud"):
            if expected_latency(tool, cache[item["item_id"]][tool], state) is not None and gap(probs, tool) >= best_gap:
                best, best_gap = tool, gap(probs, tool)
        return best
    return policy
