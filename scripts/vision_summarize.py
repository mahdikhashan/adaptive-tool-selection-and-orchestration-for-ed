import json
import sys

import numpy as np

from fedorch import vision
from fedorch.common import results_dir, write_json
from fedorch.items import load_items

GOOD = vision.CONDITION["good"]
MISS, FALSE_ALARM = 10.0, 1.0
KEY_FRAMES_PER_SEGMENT = 3.45
JETSON_FACTOR = 9.0
EVLM = {"hardware": "NVIDIA Jetson, 8-core CPU, 16 GB, 1024 CUDA cores",
        "segment_s_table_v": 6.448,
        "segment_s_fig_8a_non_description_intents": [1.4, 2.3],
        "histogram_filter_ms_per_frame_table_iv": 1.9}


def r(x):
    return round(float(x), 4)


def classification(truth, pred, classes):
    truth, pred = np.array(truth), np.array(pred)
    per = {}
    for c in classes:
        tp = np.sum((pred == c) & (truth == c))
        support, predicted = np.sum(truth == c), np.sum(pred == c)
        p = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        per[c] = {"support": int(support), "predicted": int(predicted), "precision": r(p), "recall": r(rec),
                  "f1": r(2 * p * rec / (p + rec)) if p + rec else 0.0}
    present = [c for c in classes if per[c]["support"]]
    return {"n": len(truth), "accuracy": r(np.mean(truth == pred)),
            "balanced_accuracy": r(np.mean([per[c]["recall"] for c in present])),
            "macro_f1": r(np.mean([per[c]["f1"] for c in present])),
            "per_class": per,
            "confusion": {t: {p: int(np.sum((truth == t) & (pred == p))) for p in classes if np.sum((truth == t) & (pred == p))}
                          for t in present}}


def majority(truth, classes):
    top = max(set(truth), key=truth.count)
    m = classification(truth, [top] * len(truth), classes)
    return {"class": top, "accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    ranks = np.empty(len(scores))
    order = np.argsort(scores, kind="mergesort")
    ranks[order] = np.arange(1, len(scores) + 1)
    for value in np.unique(scores):
        tied = scores == value
        ranks[tied] = ranks[tied].mean()
    pos, neg = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg)) if pos and neg else float("nan")


def average_precision(scores, labels):
    labels = np.asarray(labels, bool)[np.argsort(-np.asarray(scores, float), kind="mergesort")]
    hits = np.cumsum(labels)
    return float(np.sum(hits[labels] / (np.flatnonzero(labels) + 1)) / labels.sum()) if labels.sum() else float("nan")


def ece(scores, labels, bins=10):
    scores, labels = np.asarray(scores, float), np.asarray(labels, float)
    index = np.minimum((scores * bins).astype(int), bins - 1)
    return float(sum(abs(scores[index == b].mean() - labels[index == b].mean()) * np.sum(index == b)
                     for b in range(bins) if np.any(index == b)) / len(scores))


def at_threshold(scores, labels, threshold):
    pred, labels = np.asarray(scores) >= threshold, np.asarray(labels, bool)
    tp, fp = np.sum(pred & labels), np.sum(pred & ~labels)
    fn = np.sum(~pred & labels)
    p = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / labels.sum() if labels.sum() else 0.0
    return {"threshold": r(threshold), "precision": r(p), "recall": r(rec), "f1": r(2 * p * rec / (p + rec)) if p + rec else 0.0,
            "false_alarm_rate": r(fp / (~labels).sum()) if (~labels).sum() else 0.0,
            "risk_per_1000": round(1000 * (MISS * fn + FALSE_ALARM * fp) / len(labels), 2)}


def tune(scores, labels):
    grid = np.unique(np.quantile(scores, np.linspace(0, 1, 201)))
    return min(grid, key=lambda t: at_threshold(scores, labels, t)["risk_per_1000"])


def binary(scores, labels, threshold):
    return {"n": len(labels), "prevalence": r(np.mean(labels)), "auroc": r(auroc(scores, labels)),
            "average_precision": r(average_precision(scores, labels)), "ece": r(ece(scores, labels)),
            "at_0.5": at_threshold(scores, labels, 0.5), "at_val_threshold": at_threshold(scores, labels, threshold),
            "always_good_risk_per_1000": round(1000 * MISS * np.sum(labels) / len(labels), 2)}


def evaluate(rows, threshold):
    truth_c = [x["condition"] for x, _ in rows]
    pred_c = [max(p["condition"], key=p["condition"].get) for _, p in rows]
    defect = [x["defect"] for x, _ in rows]
    p_defect = [p["defect"] for _, p in rows]
    condition_defect = [c != GOOD for c in pred_c]
    correct = [t == c for t, c in zip(truth_c, pred_c)]
    components = list(dict.fromkeys(vision.COMPONENT.values()))
    truth_k = [x["component"] for x, _ in rows]
    return {
        "component": dict(classification(truth_k, [max(p["component"], key=p["component"].get) for _, p in rows], components),
                          majority=majority(truth_k, components)),
        "condition": dict(classification(truth_c, pred_c, list(vision.CONDITION.values())),
                          majority=majority(truth_c, list(vision.CONDITION.values()))),
        "defect_noul": binary(p_defect, defect, threshold),
        "defect_from_condition": at_threshold([float(c) for c in condition_defect], defect, 0.5),
        "agreement_noul_vs_condition": r(np.mean([(s >= 0.5) == c for s, c in zip(p_defect, condition_defect)])),
        "judgeable": {"mean_p": r(np.mean([p["judgeable"] for _, p in rows])),
                      "auroc_for_condition_correct": r(auroc([p["judgeable"] for _, p in rows], correct))},
    }


def bootstrap(rows, threshold, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    keys = {"condition_accuracy": [], "condition_macro_f1": [], "condition_balanced_accuracy": [],
            "component_accuracy": [], "defect_auroc": [], "defect_recall_at_val_threshold": []}
    truth_c = np.array([x["condition"] for x, _ in rows])
    pred_c = np.array([max(p["condition"], key=p["condition"].get) for _, p in rows])
    truth_k = np.array([x["component"] for x, _ in rows])
    pred_k = np.array([max(p["component"], key=p["component"].get) for _, p in rows])
    defect = np.array([x["defect"] for x, _ in rows])
    p_defect = np.array([p["defect"] for _, p in rows])
    classes = list(vision.CONDITION.values())
    for _ in range(n):
        i = rng.integers(0, len(rows), len(rows))
        m = classification(truth_c[i], pred_c[i], classes)
        keys["condition_accuracy"].append(m["accuracy"])
        keys["condition_macro_f1"].append(m["macro_f1"])
        keys["condition_balanced_accuracy"].append(m["balanced_accuracy"])
        keys["component_accuracy"].append(np.mean(truth_k[i] == pred_k[i]))
        keys["defect_auroc"].append(auroc(p_defect[i], defect[i]))
        keys["defect_recall_at_val_threshold"].append(at_threshold(p_defect[i], defect[i], threshold)["recall"])
    return {k: [r(np.percentile(v, 2.5)), r(np.percentile(v, 97.5))] for k, v in keys.items()}


def per_component(rows, threshold):
    out = {}
    for component in dict.fromkeys(vision.COMPONENT.values()):
        part = [(x, p) for x, p in rows if x["component"] == component]
        if not part:
            continue
        m = evaluate(part, threshold)
        out[component] = {"n": len(part), "defects": int(sum(x["defect"] for x, _ in part)),
                          "component_recall": m["component"]["per_class"][component]["recall"],
                          "condition_accuracy": m["condition"]["accuracy"],
                          "condition_balanced_accuracy": m["condition"]["balanced_accuracy"],
                          "defect_auroc": m["defect_noul"]["auroc"],
                          "defect_recall_at_val_threshold": m["defect_noul"]["at_val_threshold"]["recall"]}
    return out


def latency(run, predictions):
    out = {"device": run["device"], "gpu": run["gpu"], "precision": run["precision"], "peak_gpu_mb": run["peak_gpu_mb"],
           "full_run_all_questions_ms": vision.stats([p["ms"] for p in predictions]),
           "batch_1_all_questions_ms": run["all_questions_ms"], "batch_1_condition_only_ms": run["condition_only_ms"]}
    for name in ("all_questions_ms", "condition_only_ms"):
        segment = KEY_FRAMES_PER_SEGMENT * run[name]["median"] / 1000
        out[f"segment_s_{name[:-3]}"] = {"measured": round(segment, 3), "jetson_estimate_x9": round(JETSON_FACTOR * segment, 3)}
    out["evlm_reference"] = EVLM
    cpu = results_dir("vision") / "latency_cpu.json"
    if cpu.exists():
        out["cpu"] = json.loads(cpu.read_text())
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "zeroshot"
    truth = {x["image"]: x for x in vision.images(load_items())}
    out = results_dir(f"vision/{run}")
    predictions = [json.loads(line) for line in (out / "predictions.jsonl").open()]
    rows = [(truth[p["image"]], p["answers"]) for p in predictions]
    split = {s: [row for row in rows if row[0]["split"] == s] for s in ("train", "val", "test")}
    threshold = tune([p["defect"] for _, p in split["val"]], [x["defect"] for x, _ in split["val"]])
    summary = {"run": run, "images": len(rows), "truncated": sum(bool(p["truncated"]) for p in predictions),
               "defect_threshold_from_val": r(threshold),
               "test": evaluate(split["test"], threshold),
               "test_bootstrap_95ci": bootstrap(split["test"], threshold),
               "test_per_component": per_component(split["test"], threshold),
               **{s: evaluate(split[s], threshold) for s in ("val", "train") if split[s]},
               "all": evaluate(rows, threshold),
               "latency": latency(json.loads((out / "run.json").read_text()), predictions)}
    write_json(out / "summary.json", summary)
    t = summary["test"]
    print(json.dumps({"component": {k: t["component"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "majority")},
                      "condition": {k: t["condition"][k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "majority")},
                      "defect_noul": {k: t["defect_noul"][k] for k in ("auroc", "average_precision", "ece", "at_val_threshold")},
                      "ci": summary["test_bootstrap_95ci"], "latency": summary["latency"]}, indent=1))


if __name__ == "__main__":
    main()
