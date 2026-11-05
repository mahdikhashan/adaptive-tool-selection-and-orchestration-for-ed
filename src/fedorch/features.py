import json
import math

from fedorch.common import results_dir

TOOLS = ["classifier", "smolvlm_local", "cloud"]
LABELS = ["good", "defect", "rust", "corrosion", "missing-cap", "nest", "other-defect", "cannot-judge"]
QUESTION = "Will this tool give the correct answer for this frame?"
TOOL_TEXT = {
    "classifier": "accept the answer of the onboard defect classifier",
    "smolvlm_local": "ask the onboard vision-language model",
    "cloud": "ask the cloud vision-language model",
}


def ranked(row):
    return sorted(row["probs"].items(), key=lambda pair: -pair[1])


def numeric(item, row, assets):
    probs = [p for _, p in ranked(row)]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    return ([float(item["task"] == "T2")]
            + [float(item["asset"] == a) for a in assets]
            + [row["confidence"], probs[0] - probs[1], entropy, row["probs"].get("good", 0.0), float(len(probs))]
            + [float(row["prediction"] == label) for label in LABELS])


def evidence(item, row):
    (first, p1), (second, p2) = ranked(row)[:2]
    task = "yes/no defect question" if item["task"] == "T2" else "defect type question"
    return (f"Task: {task}. Component: {item['asset'].replace('-', ' ')}. "
            f"Classifier answer: {first} with probability {p1:.2f}. Second answer: {second} with probability {p2:.2f}.")


def write_q(name, q_by_item):
    with (results_dir("step4") / f"q_{name}.jsonl").open("w") as file:
        for item_id, q in q_by_item.items():
            file.write(json.dumps({"item_id": item_id, "q": q}) + "\n")


def read_q(path):
    return {row["item_id"]: row["q"] for row in map(json.loads, path.open())}
