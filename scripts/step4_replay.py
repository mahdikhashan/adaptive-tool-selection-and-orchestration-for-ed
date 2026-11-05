import argparse

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.controller import LAMBDAS, risk_rule, true_q
from fedorch.costs import COST
from fedorch.features import read_q
from fedorch.items import load_items
from fedorch.network import REGIMES, lte_mbps
from fedorch.policies import THRESHOLDS, always, cascade, oracle, to_local
from fedorch.replay import evaluate, mission, trace_for, tune
from fedorch.simulate import RISK


def main():
    parser = argparse.ArgumentParser(description="Step 4: score the learned controllers (risk rule on q) against the static cascade.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--jetson-factor", type=float, default=COST["jetson_factor"])
    args = parser.parse_args()
    COST["jetson_factor"] = args.jetson_factor

    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    estimators = {path.stem.removeprefix("q_"): read_q(path) for path in sorted(results_dir("step4").glob("q_*.jsonl"))}
    estimators["true (upper bound)"] = true_q(cache)
    results = {"cost": COST, "risk_weights": RISK, "estimators": list(estimators), "policies": {}, "tuning": {}}

    for regime in REGIMES:
        traces = [trace_for(test, regime, seed, base) for seed in range(args.seeds)]
        threshold, _ = tune(lambda t: cascade(t, to_local), THRESHOLDS, val, cache, regime, base)
        policies = {"classifier only": always("classifier"), f"static cascade θ={threshold}": cascade(threshold, to_local),
                    "oracle": oracle(cache)}
        for name, q in estimators.items():
            lam, scores = tune(lambda value: risk_rule(q, cache, value), LAMBDAS, val, cache, regime, base)
            policies[f"risk rule + q_{name} λ={lam}"] = risk_rule(q, cache, lam)
            results["tuning"][f"{regime}/{name}"] = {"chosen": lam, "val_risk": scores}
        for name, policy in policies.items():
            result = evaluate(policy, test, cache, traces)
            key = name.split(" θ=")[0].split(" λ=")[0]
            results["policies"].setdefault(key, {})[regime] = {"name": name, **result}
            print(f"{regime:5s} {name:42s} risk {result['risk_per_1000']:7.1f}  recall {result['defect_recall']:.3f}  "
                  f"FA {result['false_alarm_rate']:.3f}  esc {result['escalation_rate']:.3f}  uplink {result['uplink_mb_per_1000']:.2f}")

    write_json(results_dir("step4") / "metrics.json", results)


if __name__ == "__main__":
    main()
