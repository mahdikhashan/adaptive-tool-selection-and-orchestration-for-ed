import gzip
import itertools
import json
import math
import random
import re
import zlib

from fedorch.common import results_dir
from fedorch.decision import ACTION_TOOL, ACTIONS, ESCALATION_COST

DEFECTS = ["defect", "rust", "corrosion", "missing-cap", "nest", "other-defect"]
NAMES = {"defect": "general defects", "rust": "rust", "corrosion": "corrosion", "missing-cap": "missing caps",
         "nest": "bird nests", "other-defect": "torn parts or peeling paint"}
COMPONENTS = ["polymer-insulator-upper-shackle", "glass-insulator", "yoke-suspension", "vari-grip",
              "polymer-insulator-lower-shackle", "lightning-rod-suspension"]
WEIGHTS = {"critical_miss": 20.0, "other_miss": 2.0, "false_alarm": 1.0, "strict_false_alarm": 3.0, "abstain": 0.5}

TRAIN_TEMPLATES = [
    "Find every {focus}. Other defects are less important.",
    "Priority defects: {focus}. Report the rest only if you are sure.",
    "Do not miss any {focus} on this line.",
    "The main goal of this flight is {focus}.",
]
TEST_TEMPLATES = [
    "Our maintenance team cares most about {focus}, the remaining defect types matter less.",
    "Please make sure that no {focus} goes unreported.",
    "Look out for {focus} above everything else.",
]
COMPONENT_TRAIN_TEMPLATES = [
    "Inspect the {focus} carefully. Other parts are less important.",
    "Today we only care about the {focus}.",
    "Do not miss any defect on the {focus}.",
    "The {focus} are the priority of this flight.",
]
COMPONENT_TEST_TEMPLATES = [
    "The maintenance crew asks for a close look at the {focus}, the other components matter less.",
    "Please report every problem on the {focus}.",
    "Focus the inspection on the {focus}.",
]
STRICT_TRAIN = " Avoid false alarms."
STRICT_TEST = " False reports waste crew time, so keep them rare."

KINDS = {
    "label": (DEFECTS, NAMES, TRAIN_TEMPLATES, TEST_TEMPLATES),
    "component": (COMPONENTS, {c: c.replace("-", " ") for c in COMPONENTS}, COMPONENT_TRAIN_TEMPLATES, COMPONENT_TEST_TEMPLATES),
}


def focus_sets(kind):
    sets = [tuple(s) for k in (1, 2) for s in itertools.combinations(KINDS[kind][0], k)]
    unseen = sets[1::4][:6]
    return [f for f in sets if f not in unseen], unseen


def make_intent(kind, focus, strict, template, strict_text, group):
    text = template.format(focus=" and ".join(KINDS[kind][1][d] for d in focus)) + (strict_text if strict else "")
    return {"id": f"{group}:{'+'.join(focus)}:{int(strict)}:{zlib.crc32(text.encode()) % 10000:04d}",
            "group": group, "kind": kind, "focus": list(focus), "strict": strict, "text": text}


def intent_sets(kind="label"):
    rng = random.Random(0)
    seen, unseen = focus_sets(kind)
    _, _, train_templates, test_templates = KINDS[kind]
    train = [make_intent(kind, f, s, t, STRICT_TRAIN, "train") for f in seen for s in (False, True) for t in train_templates]
    val = rng.sample(train, 6)
    test = ([make_intent(kind, f, s, rng.choice(test_templates), STRICT_TEST, "new_wording") for f in seen[:6] for s in (False, True)]
            + [make_intent(kind, f, s, rng.choice(test_templates), STRICT_TEST, "new_focus") for f in unseen for s in (False, True)])
    return {"train": train, "val": val, "test": test}


def critical(item, intent):
    key = item["label"] if intent["kind"] == "label" else item["asset"]
    return item["label"] != "good" and key in intent["focus"]


def answer_risk(answer, item, intent):
    label = item["label"]
    if label != "good" and answer in (None, "good"):
        return WEIGHTS["critical_miss"] if critical(item, intent) else WEIGHTS["other_miss"]
    if label == "good" and answer not in (None, "good", "cannot-judge"):
        return WEIGHTS["strict_false_alarm"] if intent["strict"] else WEIGHTS["false_alarm"]
    return WEIGHTS["abstain"] if answer in (None, "cannot-judge") else 0.0


def action_risks(item, rows, intent):
    return {a: answer_risk(rows[ACTION_TOOL[a]]["prediction"], item, intent) + ESCALATION_COST[a] for a in ACTIONS}


def soft_target(item, rows, intent, temperature=1.0):
    weights = {a: math.exp(-r / temperature) for a, r in action_risks(item, rows, intent).items()}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def informative(item, rows, intent):
    risks = action_risks(item, rows, intent)
    return max(risks.values()) - min(risks.values()) > max(ESCALATION_COST.values())


def training_pairs(items, cache, intents, per_item=2, oversample=5, seed=0):
    rng = random.Random(seed)
    pairs = []
    for item in items:
        for intent in rng.sample(intents, per_item):
            pairs += [(item, intent)] * (oversample if informative(item, cache[item["item_id"]], intent) else 1)
    return pairs


def parsed(intent):
    return [float(d in intent["focus"]) for d in KINDS[intent["kind"]][0]] + [float(intent["strict"])]


def bag_of_words(intent, size=64):
    vector = [0.0] * size
    for word in re.findall(r"[a-z]+", intent["text"].lower()):
        vector[zlib.crc32(word.encode()) % size] += 1.0
    return vector


def budgeted(policy, budget):
    if budget is None:
        return policy
    count = {"frames": 0, "escalations": 0}

    def limited(item, history, state):
        if not history:
            count["frames"] += 1
            return policy(item, history, state)
        if len(history) == 1 and count["escalations"] >= budget * count["frames"]:
            return None
        tool = policy(item, history, state)
        if tool is not None and len(history) == 1:
            count["escalations"] += 1
        return tool
    return limited


def write_p(name, rows):
    with gzip.open(results_dir("planF") / f"p_{name}.jsonl.gz", "wt") as file:
        for item_id, intent_id, p in rows:
            file.write(json.dumps({"item_id": item_id, "intent_id": intent_id, "p": {a: round(v, 5) for a, v in p.items()}}) + "\n")


def read_p(path):
    table = {}
    for row in map(json.loads, gzip.open(path, "rt")):
        table.setdefault(row["intent_id"], {})[row["item_id"]] = row["p"]
    return table


def mission_metrics(records, intent):
    n = len(records)
    focus = [r for r in records if critical(r["item"], intent)]
    return {
        "risk_per_1000": 1000 * sum(answer_risk(r["answer"], r["item"], intent) for r in records) / n,
        "critical_recall": sum(r["answer"] == r["item"]["label"] for r in focus) / len(focus) if focus else None,
        "critical_items": len(focus),
        "escalation_rate": sum(r["tools"] != ["classifier"] for r in records) / n,
    }
