import argparse
import statistics

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import choice_rule, thresholds
from fedorch.intents import budgeted, intent_sets, mission_metrics, read_p
from fedorch.items import load_items
from fedorch.network import lte_mbps
from fedorch.policies import THRESHOLDS, always, cascade, oracle, to_local
from fedorch.replay import mission, trace_for
from fedorch.simulate import run_mission

REGIME = "weak"


def score(make, items, cache, traces, intent):
    runs = [mission_metrics(run_mission(items, cache, trace, make()), intent) for trace in traces]
    return {k: statistics.mean(r[k] for r in runs) if runs[0][k] is not None else None for k in runs[0]}


def tune(make, grid, items, cache, trace, intents):
    risk = {value: statistics.mean(mission_metrics(run_mission(items, cache, trace, make(value, intent)), intent)["risk_per_1000"]
                                   for intent in intents) for value in grid}
    return min(risk, key=risk.get)


def main():
    parser = argparse.ArgumentParser(description="Plan F: replay the intent-aware controllers on the test intents.")
    parser.add_argument("--kind", default="label", help="intent focus: label or component")
    parser.add_argument("--budget", type=float, default=None, help="maximum share of escalated frames in a mission")
    parser.add_argument("--per-intent-threshold", action="store_true", help="also tune the blind MLP threshold for each test intent")
    args = parser.parse_args()

    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    intents = intent_sets(args.kind)
    val_trace = trace_for(val, REGIME, 1000, base)
    traces = [trace_for(test, REGIME, seed, base) for seed in range(5)]
    budget = args.budget

    makers = {"classifier only": lambda intent: lambda: always("classifier"),
              "oracle": lambda intent: lambda: budgeted(oracle(cache), budget)}
    theta = tune(lambda t, intent: budgeted(cascade(t, to_local), budget), THRESHOLDS, val, cache, val_trace, intents["val"])
    makers[f"static cascade θ={theta}"] = lambda intent: lambda: budgeted(cascade(theta, to_local), budget)
    for path in sorted(results_dir("planF").glob(f"p_{args.kind}_*.jsonl.gz")):
        name, table = path.name.removeprefix(f"p_{args.kind}_").removesuffix(".jsonl.gz"), read_p(path)
        pooled = {f"{intent['id']}|{i}": table[intent["id"]][i] for intent in intents["val"] for i in val_ids}
        grid = thresholds(pooled, list(pooled))
        delta = tune(lambda d, intent, table=table: budgeted(choice_rule(table[intent["id"]], cache, d), budget),
                     grid, val, cache, val_trace, intents["val"])
        makers[f"choice + {name}"] = lambda intent, table=table, delta=delta: lambda: budgeted(choice_rule(table[intent["id"]], cache, delta), budget)
        if args.per_intent_threshold and name == "mlp_blind":
            reference = table[intents["val"][0]["id"]]
            per = {intent["id"]: tune(lambda d, _: budgeted(choice_rule(reference, cache, d), budget), grid, val, cache, val_trace, [intent])
                   for intent in intents["test"]}
            makers["choice + mlp_blind, threshold per intent"] = (
                lambda intent, table=table, per=per: lambda: budgeted(choice_rule(table[intent["id"]], cache, per[intent["id"]]), budget))

    results = {"kind": args.kind, "budget": budget, "regime": REGIME, "intents": intents["test"], "policies": {}}
    for name, make in makers.items():
        per_intent = {intent["id"]: score(make(intent), test, cache, traces, intent) for intent in intents["test"]}
        summary = {}
        for group in ["new_wording", "new_focus"]:
            rows = [per_intent[i["id"]] for i in intents["test"] if i["group"] == group]
            summary[group] = {k: round(statistics.mean(r[k] for r in rows if r[k] is not None), 4)
                              for k in ["risk_per_1000", "critical_recall", "escalation_rate"]}
        results["policies"][name] = {"summary": summary, "per_intent": per_intent}
        print(f"{name:42s} new wording risk {summary['new_wording']['risk_per_1000']:6.1f} recall {summary['new_wording']['critical_recall']:.3f} "
              f"esc {summary['new_wording']['escalation_rate']:.3f} | new focus risk {summary['new_focus']['risk_per_1000']:6.1f} "
              f"recall {summary['new_focus']['critical_recall']:.3f}")
    suffix = f"_budget{budget}" if budget is not None else ""
    write_json(results_dir("planF") / f"metrics_{args.kind}{suffix}.json", results)


if __name__ == "__main__":
    main()
