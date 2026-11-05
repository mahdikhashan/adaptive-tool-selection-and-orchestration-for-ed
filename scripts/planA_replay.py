import re
import statistics

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import choice_rule, read_p, thresholds
from fedorch.items import load_items
from fedorch.network import lte_mbps
from fedorch.policies import THRESHOLDS, cascade, to_local
from fedorch.replay import evaluate, mission, trace_for, tune

REGIME = "weak"
NAME = re.compile(r"(?P<model>[a-z]+)_T[\d.]+_O\d+(_f(?P<fraction>[\d.]+)_s(?P<seed>\d+))?")


def main():
    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    traces = [trace_for(test, REGIME, seed, base) for seed in range(5)]
    threshold, _ = tune(lambda t: cascade(t, to_local), THRESHOLDS, val, cache, REGIME, base)
    results = {"regime": REGIME, "static_cascade": evaluate(cascade(threshold, to_local), test, cache, traces), "runs": {}}

    for path in sorted(results_dir("planA").glob("p_*.jsonl")):
        name = path.stem.removeprefix("p_")
        match = NAME.fullmatch(name)
        p = read_p(path)
        margin, _ = tune(lambda m: choice_rule(p, cache, m), thresholds(p, val_ids), val, cache, REGIME, base)
        result = evaluate(choice_rule(p, cache, margin), test, cache, traces)
        results["runs"][name] = {"model": match["model"], "fraction": float(match["fraction"] or 1.0),
                                 "seed": int(match["seed"] or 0), **result}
        print(f"{name:32s} risk {result['risk_per_1000']:7.1f}  escalation {result['escalation_rate']:.3f}")

    curves = {}
    for run in results["runs"].values():
        curves.setdefault(run["model"], {}).setdefault(run["fraction"], []).append(run["risk_per_1000"])
    results["curves"] = {model: {str(f): {"mean": round(statistics.mean(v), 2), "std": round(statistics.pstdev(v), 2), "seeds": len(v)}
                                 for f, v in sorted(by_fraction.items())} for model, by_fraction in curves.items()}
    print(results["curves"])
    write_json(results_dir("planA") / "metrics.json", results)


if __name__ == "__main__":
    main()
