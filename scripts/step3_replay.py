import argparse
import statistics

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.costs import COST, SOURCES
from fedorch.items import load_items
from fedorch.network import REGIMES, bandwidth_trace, lte_mbps
from fedorch.policies import THRESHOLDS, baselines
from fedorch.simulate import RISK, metrics, run_mission


def mission(split, items):
    return sorted((i for i in items if i["split"] == split), key=lambda i: (i["tower"], i["parent_id"], i["item_id"]))


def trace_for(items, regime, seed, base):
    return bandwidth_trace(regime, int(len(items) * COST["frame_interval_s"]) + 60, seed, base)


def tune(make, val, cache, regime, base):
    trace = trace_for(val, regime, 1000, base)
    scores = {t: metrics(run_mission(val, cache, trace, make(t)))["risk_per_1000"] for t in THRESHOLDS}
    return min(scores, key=scores.get), scores


def mean_over_seeds(runs):
    return {k: round(statistics.mean(r[k] for r in runs), 4) for k in runs[0]}


def main():
    parser = argparse.ArgumentParser(description="Step 3: replay the tool cache over network traces and score the baseline policies.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jetson-factor", type=float, default=COST["jetson_factor"], help="GV100 time x factor = Jetson time")
    parser.add_argument("--out", default="metrics.json")
    args = parser.parse_args()
    COST["jetson_factor"] = args.jetson_factor

    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    fixed, tuned = baselines(cache)
    results = {"cost": COST, "sources": SOURCES, "risk_weights": RISK, "regimes": REGIMES, "test_items": len(test), "policies": {}}

    for regime in REGIMES:
        traces = [trace_for(test, regime, seed, base) for seed in range(args.seeds)]
        policies = dict(fixed)
        for name, make in tuned.items():
            threshold, scores = tune(make, val, cache, regime, base)
            policies[f"{name} θ={threshold}"] = make(threshold)
            results.setdefault("tuning", {})[f"{regime}/{name}"] = {"chosen": threshold, "val_risk": scores}
        for name, policy in policies.items():
            runs = [metrics(run_mission(test, cache, trace, policy)) for trace in traces]
            results["policies"].setdefault(name.split(" θ=")[0], {})[regime] = {"name": name, **mean_over_seeds(runs)}
            print(f"{regime:5s} {name:42s} risk {runs[0]['risk_per_1000']:7.1f}  recall {runs[0]['defect_recall']:.3f}  late {runs[0]['late_rate']:.3f}")

    write_json(results_dir("step3") / args.out, results)


if __name__ == "__main__":
    main()
