from fedorch.cache import load_cache
from fedorch.calibration import calibrate
from fedorch.common import results_dir, write_json
from fedorch.controller import LAMBDAS, risk_rule
from fedorch.decision import MARGINS, choice_rule, read_p
from fedorch.features import read_q
from fedorch.items import load_items
from fedorch.network import REGIMES, lte_mbps
from fedorch.policies import THRESHOLDS, cascade, oracle, to_local
from fedorch.replay import evaluate, mission, trace_for, tune
from fedorch.simulate import RISK


def main():
    items, cache, base = load_items(), load_cache(), lte_mbps()
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    q_estimators = {"q_mlp (step 4 features)": results_dir("step4") / "q_mlp.jsonl", "q_mlp_v2": results_dir("step4b") / "q_mlp_v2.jsonl"}
    q_estimators = {name: calibrate(read_q(path), cache, val_ids)[0] for name, path in q_estimators.items() if path.exists()}
    p_estimators = {path.stem: read_p(path) for path in sorted(results_dir("step4b").glob("p_*.jsonl"))}
    results = {"risk_weights": RISK, "policies": {}, "tuning": {}}

    for regime in REGIMES:
        traces = [trace_for(test, regime, seed, base) for seed in range(5)]
        threshold, _ = tune(lambda t: cascade(t, to_local), THRESHOLDS, val, cache, regime, base)
        policies = {f"static cascade θ={threshold}": cascade(threshold, to_local), "oracle": oracle(cache)}
        for name, q in q_estimators.items():
            lam, _ = tune(lambda v: risk_rule(q, cache, v), LAMBDAS, val, cache, regime, base)
            policies[f"risk rule + {name} λ={lam}"] = risk_rule(q, cache, lam)
        for name, p in p_estimators.items():
            margin, scores = tune(lambda m: choice_rule(p, cache, m), MARGINS, val, cache, regime, base)
            policies[f"choice + {name} δ={margin}"] = choice_rule(p, cache, margin)
            results["tuning"][f"{regime}/{name}"] = {"chosen": margin, "val_risk": scores}
        for name, policy in policies.items():
            result = evaluate(policy, test, cache, traces)
            key = name.split(" θ=")[0].split(" λ=")[0].split(" δ=")[0]
            results["policies"].setdefault(key, {})[regime] = {"name": name, **result}
            print(f"{regime:5s} {name:44s} risk {result['risk_per_1000']:7.1f}  recall {result['defect_recall']:.3f}  "
                  f"FA {result['false_alarm_rate']:.3f}  esc {result['escalation_rate']:.3f}  uplink {result['uplink_mb_per_1000']:.2f}")

    write_json(results_dir("step4b") / "metrics.json", results)


if __name__ == "__main__":
    main()
