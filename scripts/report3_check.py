"""Per-question check of the three-question report: tools alone, oracle, static cascade,
learned whose-answer selector and learned fusion. Fit on val towers, scored on test towers."""
import json
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression

from fedorch import vision
from fedorch.common import results_dir, write_json
from fedorch.items import load_items

COMPONENTS = list(dict.fromkeys(vision.COMPONENT.values()))
CONDITIONS = list(vision.CONDITION.values())
TOOLS = {"component": ["laya", "smolvlm", "qwen"], "defect": ["classifier", "laya", "smolvlm", "qwen"],
         "condition": ["classifier", "laya", "smolvlm", "qwen"]}
CASCADE = {"component": ("laya", "smolvlm"), "defect": ("classifier", "smolvlm"), "condition": ("classifier", "smolvlm")}
MISS, FALSE_ALARM = 10, 1


def load():
    truth = {r["image"]: r for r in vision.images(load_items())}
    answers = defaultdict(dict)
    for row in map(json.loads, (results_dir("report3") / "answers.jsonl").open()):
        answers[row["image"]][row["tool"]] = row["answers"]
    images = [i for i, a in answers.items() if all(t in a for t in TOOLS["defect"])]
    return truth, answers, images


def vector(question, value):
    if question == "defect":
        return np.array([1 - value, value])
    names = COMPONENTS if question == "component" else CONDITIONS
    v = np.array([float(value.get(n, 0.)) for n in names])
    return v / v.sum() if v.sum() > 0 else v


def target(question, row):
    if question == "defect":
        return int(row["defect"])
    return (COMPONENTS if question == "component" else CONDITIONS).index(row[question])


def risk(y, p):
    return MISS if y and not p else FALSE_ALARM if p and not y else 0


def scores(question, y, p):
    y, p = np.asarray(y), np.asarray(p)
    if question == "defect":
        return {"risk_per_1000": round(1000 * np.mean([risk(a, b) for a, b in zip(y, p)]), 1),
                "recall": round(float(np.mean(p[y == 1] == 1)), 4), "false_alarm": round(float(np.mean(p[y == 0] == 1)), 4)}
    k = len(COMPONENTS if question == "component" else CONDITIONS)
    f1 = [2 * np.sum((p == c) & (y == c)) / max(1, np.sum(p == c) + np.sum(y == c)) for c in range(k) if np.any(y == c)]
    return {"accuracy": round(float(np.mean(y == p)), 4), "macro_f1": round(float(np.mean(f1)), 4)}


def decide(question, probs, threshold=.5):
    return int(probs[1] >= threshold) if question == "defect" else int(np.argmax(probs))


def tune_threshold(y, p_defect):
    grid = np.unique(np.quantile(p_defect, np.linspace(0, 1, 201)))
    return min(grid, key=lambda t: np.mean([risk(a, int(b >= t)) for a, b in zip(y, p_defect)]))


def tower_folds(rows, k=5):
    towers = sorted({r["tower"] for r in rows})
    return np.array([towers.index(r["tower"]) % k for r in rows])


def main():
    truth, answers, images = load()
    split = {s: [i for i in images if truth[i]["split"] == s] for s in ("val", "test")}
    report = {"images": {s: len(v) for s, v in split.items()}}
    for question, tools in TOOLS.items():
        V = {s: {t: np.array([vector(question, answers[i][t][question]) for i in split[s]]) for t in tools} for s in split}
        y = {s: np.array([target(question, truth[i]) for i in split[s]]) for s in split}
        out = {}
        for t in tools:
            out[f"{t} alone"] = scores(question, y["test"], [decide(question, v) for v in V["test"][t]])
        hits = np.array([[decide(question, V["test"][t][n]) for t in tools] for n in range(len(y["test"]))])
        oracle = [row[np.argmin([risk(yy, p) if question == "defect" else int(p != yy) for p in row])] for row, yy in zip(hits, y["test"])]
        out["oracle best-of-tools (labels)"] = scores(question, y["test"], oracle)
        first, second = CASCADE[question]
        conf = {s: V[s][first].max(1) for s in split}
        def cascade(s, th):
            return [decide(question, V[s][second][n] if conf[s][n] < th else V[s][first][n]) for n in range(len(y[s]))]
        loss = (lambda s, p: np.mean([risk(a, b) for a, b in zip(y[s], p)])) if question == "defect" else (lambda s, p: np.mean(np.array(p) != y[s]))
        th = min((.5, .6, .7, .8, .9, .95, .99), key=lambda t: loss("val", cascade("val", t)))
        out[f"static cascade {first}->{second} (theta {th})"] = scores(question, y["test"], cascade("test", th))
        X = {s: np.hstack([V[s][t] for t in tools]) for s in split}
        fold = tower_folds([truth[i] for i in split["val"]])
        make = lambda: LogisticRegression(max_iter=5000, C=.5)
        if question == "defect":
            cv = np.zeros(len(y["val"]))
            for f in range(5):
                cv[fold == f] = make().fit(X["val"][fold != f], y["val"][fold != f]).predict_proba(X["val"][fold == f])[:, 1]
            th = tune_threshold(y["val"], cv)
            p = make().fit(X["val"], y["val"]).predict_proba(X["test"])[:, 1]
            out[f"learned fusion (threshold {th:.3f})"] = scores(question, y["test"], (p >= th).astype(int))
        else:
            model = make().fit(X["val"], y["val"])
            proba = np.zeros((len(y["test"]), len(COMPONENTS if question == "component" else CONDITIONS)))
            proba[:, model.classes_] = model.predict_proba(X["test"])
            out["learned fusion"] = scores(question, y["test"], proba.argmax(1))
        q = {}
        for t in tools:
            right = np.array([decide(question, v) == yy for v, yy in zip(V["val"][t], y["val"])])
            q[t] = make().fit(X["val"], right).predict_proba(X["test"])[:, 1] if 0 < right.mean() < 1 else np.full(len(y["test"]), right.mean())
        picks = []
        for n in range(len(y["test"])):
            def cost(t):
                p = decide(question, V["test"][t][n])
                return (1 - q[t][n]) * (MISS if question == "defect" and p == 0 else 1)
            picks.append(decide(question, V["test"][min(tools, key=cost)][n]))
        out["learned whose-answer selector"] = scores(question, y["test"], picks)
        report[question] = out
        print(f"\n== {question}")
        for name, s in out.items():
            print(f"  {name:48s} " + "  ".join(f"{k} {v}" for k, v in s.items()))
    write_json(results_dir("report3") / "check.json", report)


if __name__ == "__main__":
    main()
