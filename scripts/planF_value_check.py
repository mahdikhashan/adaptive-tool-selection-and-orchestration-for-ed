import argparse
import statistics

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.controller import expected_latency
from fedorch.decision import choice_rule, gap, thresholds
from fedorch.intents import WEIGHTS, budgeted, intent_sets, mission_metrics, read_p
from fedorch.items import load_items
from fedorch.network import lte_mbps
from fedorch.replay import mission, trace_for
from fedorch.simulate import run_mission

MULTIPLIER = WEIGHTS["critical_miss"] / WEIGHTS["other_miss"]


def parse_and_weight(p, cache, intent, threshold):
    def policy(item, history, state):
        if not history:
            return "classifier"
        if len(history) > 1:
            return None
        key = item["label"] if intent["kind"] == "label" else item["asset"]
        weight = MULTIPLIER if key in intent["focus"] else 1.0
        best, best_gap = None, threshold
        for tool in ("smolvlm_local", "cloud"):
            value = gap(p[item["item_id"]], tool) * weight
            if expected_latency(tool, cache[item["item_id"]][tool], state) is not None and value >= best_gap:
                best, best_gap = tool, value
        return best
    return policy


def main():
    parser = argparse.ArgumentParser(description="Plan F: upper estimate of the value of the intent (blind MLP with a parse-and-weight rule).")
    parser.add_argument("--kind", default="component")
    args = parser.parse_args()
    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    intents = intent_sets(args.kind)
    val_trace = trace_for(val, "weak", 1000, base)
    traces = [trace_for(test, "weak", seed, base) for seed in range(5)]
    blind = read_p(results_dir("planF") / f"p_{args.kind}_mlp_blind.jsonl.gz")
    reference = blind[intents["val"][0]["id"]]
    grid = sorted(set(g * m for g in thresholds(reference, val_ids) for m in (1.0, MULTIPLIER)))
    rules = {"blind": lambda p, intent, d: choice_rule(p, cache, d),
             "parse-and-weight": lambda p, intent, d: parse_and_weight(p, cache, intent, d)}
    results = {}
    for budget in (None, 0.02):
        for name, rule in rules.items():
            val_risk = {d: statistics.mean(mission_metrics(run_mission(val, cache, val_trace, budgeted(rule(reference, i, d), budget)), i)["risk_per_1000"]
                                           for i in intents["val"]) for d in grid}
            d = min(val_risk, key=val_risk.get)
            row = {}
            for group in ("new_wording", "new_focus"):
                risks = [statistics.mean(mission_metrics(run_mission(test, cache, t, budgeted(rule(blind[i["id"]], i, d), budget)), i)["risk_per_1000"]
                                         for t in traces) for i in intents["test"] if i["group"] == group]
                row[group] = round(statistics.mean(risks), 2)
            results[f"budget {budget} / {name}"] = row
            print(f"budget {budget}: {name:18s} {row}")
    write_json(results_dir("planF") / f"value_check_{args.kind}.json", results)


if __name__ == "__main__":
    main()
