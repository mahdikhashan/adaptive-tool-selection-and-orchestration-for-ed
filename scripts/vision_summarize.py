import json
import statistics

import numpy as np

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.items import load_items
from fedorch.vision import ROUTES, answer_risk, labels_of

QUANTILES = [0.0, 0.5, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.0]
CASCADE = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995, 1.01]


def metrics(items, predictions, escalated=None):
    risks = [answer_risk(predictions[i["item_id"]], i["label"]) for i in items]
    defects = [i for i in items if i["label"] != "good"]
    out = {"items": len(items),
           "risk_per_1000": round(1000 * sum(risks) / len(items), 2),
           "accuracy": round(np.mean([predictions[i["item_id"]] == i["label"] for i in items]), 4),
           "defect_recall": round(np.mean([predictions[i["item_id"]] not in ("good", "cannot-judge") for i in defects]), 4)}
    if escalated is not None:
        out["escalation_rate"] = round(np.mean([escalated[i["item_id"]] for i in items]), 4)
    return out


def routed(items, cache, escalate):
    predictions = {i["item_id"]: cache[i["item_id"]][ROUTES[int(escalate[i["item_id"]])]]["prediction"] for i in items}
    return predictions


def tune(val, test, cache, score):
    values = [score[i["item_id"]] for i in val]
    best = None
    for q in QUANTILES:
        threshold = float(np.quantile(values, q)) + (1e-9 if q == 1.0 else 0.0)
        escalate = {i["item_id"]: score[i["item_id"]] >= threshold for i in val}
        risk = metrics(val, routed(val, cache, escalate))["risk_per_1000"]
        if best is None or risk < best[0]:
            best = (risk, threshold, q)
    escalate = {i["item_id"]: score[i["item_id"]] >= best[1] for i in test}
    return dict(metrics(test, routed(test, cache, escalate), escalate), val_risk=best[0], quantile=best[2],
                threshold=round(best[1], 4))


def references(val, test, cache):
    out = {route: metrics(test, {i["item_id"]: cache[i["item_id"]][route]["prediction"] for i in test})
           for route in ROUTES}
    best = None
    for theta in CASCADE:
        escalate = {i["item_id"]: cache[i["item_id"]]["classifier"]["confidence"] < theta for i in val}
        risk = metrics(val, routed(val, cache, escalate))["risk_per_1000"]
        if best is None or risk < best[0]:
            best = (risk, theta)
    escalate = {i["item_id"]: cache[i["item_id"]]["classifier"]["confidence"] < best[1] for i in test}
    out[f"static cascade θ={best[1]}"] = metrics(test, routed(test, cache, escalate), escalate)
    oracle = {i["item_id"]: min(ROUTES, key=lambda r: (answer_risk(cache[i["item_id"]][r]["prediction"], i["label"]),
                                                       ROUTES.index(r))) == ROUTES[1] for i in test}
    out["edge oracle"] = metrics(test, routed(test, cache, oracle), oracle)
    return out


def evaluate(rows, by_id, val, test, cache):
    answer, route = {}, {}
    for row in rows:
        item = by_id[row["item_id"]]
        if row["kind"] == "answer":
            answer[row["item_id"]] = labels_of(item)[int(np.argmax(row["probs"]))]
        else:
            route[row["item_id"]] = row["probs"][1] - row["probs"][0]
    return {"answer_as_tool": metrics(test, answer), "router": tune(val, test, cache, route)}


def mean_sd(results):
    keys = ["risk_per_1000", "accuracy", "defect_recall"]
    return {part: {k: [round(statistics.mean(r[part][k] for r in results), 2),
                       round(statistics.pstdev(r[part][k] for r in results), 2)] for k in keys}
            for part in ("answer_as_tool", "router")}


def main():
    items = load_items(("val", "test"))
    by_id = {i["item_id"]: i for i in items}
    val, test = [i for i in items if i["split"] == "val"], [i for i in items if i["split"] == "test"]
    cache = load_cache(ROUTES)
    summary = {"references (test, no network)": references(val, test, cache), "runs": {}}
    for path in sorted(results_dir("vision").glob("*/scores.jsonl")):
        rows = [json.loads(line) for line in path.open()]
        drones = sorted({row.get("drone", 0) for row in rows})
        results = [evaluate([r for r in rows if r.get("drone", 0) == d], by_id, val, test, cache) for d in drones]
        summary["runs"][path.parent.name] = results[0] if len(results) == 1 else \
            {"per_drone": results, "mean_sd_over_drones": mean_sd(results)}
    write_json(results_dir("vision") / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
