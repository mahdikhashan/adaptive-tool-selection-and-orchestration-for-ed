import argparse

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.costs import COST, SOURCES
from fedorch.items import load_items
from fedorch.network import REGIMES, lte_mbps
from fedorch.policies import THRESHOLDS, baselines
from fedorch.replay import evaluate, mission, trace_for, tune
from fedorch.simulate import RISK


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
            threshold, scores = tune(make, THRESHOLDS, val, cache, regime, base)
            policies[f"{name} θ={threshold}"] = make(threshold)
            results.setdefault("tuning", {})[f"{regime}/{name}"] = {"chosen": threshold, "val_risk": scores}
        for name, policy in policies.items():
            result = evaluate(policy, test, cache, traces)
            results["policies"].setdefault(name.split(" θ=")[0], {})[regime] = {"name": name, **result}
            print(f"{regime:5s} {name:42s} risk {result['risk_per_1000']:7.1f}  recall {result['defect_recall']:.3f}  late {result['late_rate']:.3f}")

    write_json(results_dir("step3") / args.out, results)


if __name__ == "__main__":
    main()
