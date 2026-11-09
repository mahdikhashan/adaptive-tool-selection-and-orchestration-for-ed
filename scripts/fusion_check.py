"""Item-level fusion of tool answers (ResNet, SmolVLM, Qwen, laya) with a label-free threshold.

Fit on val towers (5-fold cross-validation grouped by tower for model choice); score once on test towers.
Thresholds: tuned on val; Bayes 1/11 on val-calibrated probabilities; Bayes after EM prior-shift correction
to the test towers' unknown defect share (Saerens et al. 2002), for all test towers and per region.
"""
import json
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import answer_risk
from fedorch.folds import tower_buckets
from fedorch.items import load_items

LAYA = sys.argv[1] if len(sys.argv) > 1 else "good300_full"
TOOLS = ["classifier", "smolvlm_local", "cloud", "laya"]
BAYES = 1 / 11
cache = load_cache(("classifier", "smolvlm_local", "cloud"))
laya = {p["image"]: p["answers"] for p in map(json.loads, open(f"results/vision/{LAYA}/predictions.jsonl"))}
items = [i for i in load_items() if i["split"] in ("val", "test") and i["image"] in laya]
text = vision.CONDITION
assets = sorted({i["asset"] for i in items})


def defect_option(i):
    return next(o for o in i["options"].values() if o not in ("good", "cannot-judge"))


def p_defect(i, t):
    if t == "laya":
        p = laya[i["image"]]
        if i["task"] == "T2":
            return p["defect"]
        d, g = p["condition"][text[defect_option(i)]], p["condition"][text["good"]]
        return d / (d + g)
    r = cache[i["item_id"]][t]["probs"]
    good, cannot = r.get("good", 0.), r.get("cannot-judge", 0.)
    return max(0., 1 - good - cannot) / max(1e-6, 1 - cannot)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def features(i, rich):
    p = np.array([p_defect(i, t) for t in TOOLS])
    f = list(logit(p)) + [float(i["task"] == "T2")] + [float(i["asset"] == a) for a in assets]
    if rich:
        lp = laya[i["image"]]
        f += [lp["defect"]] + [lp["condition"][text[c]] for c in text] + [lp["component"].get(vision.COMPONENT[i["asset"]], 0.)]
        f += [float((p >= .5).sum())] + [logit(p[0]) * float(i["asset"] == a) for a in assets]
    return f


def risk(rows, preds):
    return 1000 * float(np.mean([answer_risk(p, i["label"]) for i, p in zip(rows, preds)]))


def decide(rows, probs, threshold):
    return [defect_option(i) if p >= threshold else "good" for i, p in zip(rows, probs)]


def counts(rows, preds):
    miss = sum(i["label"] != "good" and p == "good" for i, p in zip(rows, preds))
    false_alarm = sum(i["label"] == "good" and p != "good" for i, p in zip(rows, preds))
    return int(miss), int(false_alarm)


def em_prior(p, prior_train, steps=200):
    prior = prior_train
    for _ in range(steps):
        a = p * prior / prior_train
        b = (1 - p) * (1 - prior) / (1 - prior_train)
        adjusted = a / (a + b)
        prior = float(adjusted.mean())
    return prior, adjusted


def bootstrap(rows, preds, n=1000, seed=0):
    r = np.array([answer_risk(p, i["label"]) for i, p in zip(rows, preds)])
    rng = np.random.default_rng(seed)
    means = [1000 * r[rng.integers(0, len(r), len(r))].mean() for _ in range(n)]
    return [round(float(np.percentile(means, 2.5)), 1), round(float(np.percentile(means, 97.5)), 1)]


def main():
    split = {s: [i for i in items if i["split"] == s] for s in ("val", "test")}
    y = {s: np.array([i["label"] != "good" for i in split[s]]) for s in split}
    towers = sorted({i["tower"] for i in split["val"]})
    fold = np.array([towers.index(i["tower"]) % 5 for i in split["val"]])
    prior_val = float(y["val"].mean())
    choices = []
    for rich in (False, True):
        Xv = np.array([features(i, rich) for i in split["val"]])
        for C in (.03, .1, .3, 1.):
            cv = np.zeros(len(Xv))
            for k in range(5):
                cv[fold == k] = LogisticRegression(max_iter=5000, C=C).fit(Xv[fold != k], y["val"][fold != k]).predict_proba(Xv[fold == k])[:, 1]
            grid = np.unique(np.quantile(cv, np.linspace(0, 1, 401)))
            tuned = min(grid, key=lambda t: risk(split["val"], decide(split["val"], cv, t)))
            choices.append((risk(split["val"], decide(split["val"], cv, tuned)), rich, C, float(tuned)))
            print(f"val CV  rich={rich!s:5}  C={C:<4}  risk {choices[-1][0]:6.1f}  tuned threshold {tuned:.3f}", flush=True)
    _, rich, C, tuned = min(choices)
    Xv = np.array([features(i, rich) for i in split["val"]])
    Xt = np.array([features(i, rich) for i in split["test"]])
    model = LogisticRegression(max_iter=5000, C=C).fit(Xv, y["val"])
    p = model.predict_proba(Xt)[:, 1]
    test = split["test"]
    prior_test, adjusted = em_prior(p, prior_val)
    regions = tower_buckets(test, 5)
    per_region = np.zeros(len(test))
    region_priors = {}
    for r in range(5):
        idx = np.array([n for n, i in enumerate(test) if regions[i["tower"]] == r])
        if len(idx):
            region_priors[r], per_region[idx] = em_prior(p[idx], prior_val)
    best = min(np.unique(np.quantile(p, np.linspace(0, 1, 401))), key=lambda t: risk(test, decide(test, p, t)))
    rows = {"tuned on val (CV)": decide(test, p, tuned),
            "Bayes 1/11, no correction": decide(test, p, BAYES),
            "Bayes 1/11 after EM prior correction, all test": decide(test, adjusted, BAYES),
            "Bayes 1/11 after EM prior correction, per region": decide(test, per_region, BAYES),
            "test-optimal threshold (uses test labels)": decide(test, p, best)}
    single = {t: decide(test, [p_defect(i, t) for i in test], .5) for t in TOOLS}
    oracle = [min([single[t][n] for t in TOOLS], key=lambda a: answer_risk(a, i["label"])) for n, i in enumerate(test)]
    report = {"laya_predictions": LAYA, "chosen": {"rich_features": rich, "C": C, "val_cv_risk": round(min(choices)[0], 1)},
              "val_defect_share": round(prior_val, 4), "test_defect_share_true": round(float(y["test"].mean()), 4),
              "test_defect_share_em": round(prior_test, 4), "region_defect_share_em": {str(k): round(v, 4) for k, v in region_priors.items()},
              "test_auroc": round(float(roc_auc_score(y["test"], p)), 4), "methods": {}}
    print(f"\nchosen on val: rich={rich} C={C} | val defect share {prior_val:.3f} | test true {y['test'].mean():.3f}, EM estimate {prior_test:.3f} | AUROC {report['test_auroc']}")
    for name, preds in {**{f"{t} alone": v for t, v in single.items()}, **rows, "oracle best tool (labels)": oracle}.items():
        m, f = counts(test, preds)
        report["methods"][name] = {"risk_per_1000": round(risk(test, preds), 1), "ci95": bootstrap(test, preds), "misses": m, "false_alarms": f}
        print(f"  {name:52s} risk {risk(test, preds):6.1f} {bootstrap(test, preds)}  misses {m:3d}  false alarms {f:4d}")
    write_json(results_dir("fusion") / f"check_{LAYA}.json", report)


if __name__ == "__main__":
    main()
