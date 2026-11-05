import json
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLUMNS = ["risk_per_1000", "defect_recall", "false_alarm_rate", "escalation_rate", "uplink_mb_per_1000"]


def load(path):
    return json.loads((RESULTS / path).read_text())


def policy_table(path, regime="weak"):
    print(f"[{path}] regime={regime}")
    print("policy | " + " | ".join(COLUMNS))
    for name, by_regime in load(path)["policies"].items():
        row = by_regime.get(regime, by_regime)
        print(name + " | " + " | ".join(f"{row.get(c, float('nan')):.4f}" for c in COLUMNS))
    print()


def step1():
    for name in ["env", "disk", "models", "data", "trace", "smoke"]:
        d = load(f"step1/{name}.json")
        print(f"step1/{name}: ok={d.get('ok')}")
    smoke = load("step1/smoke.json")
    print(f"nano_jev_median_ms={smoke['nano_jev']['median_ms']} laya_median_ms={smoke['laya']['median_ms']}")


def step2():
    print("tool | items | accuracy | T2 | T3 | test | median_compute_ms | mean_payload_bytes")
    for tool, d in load("step2/summary.json").items():
        if "accuracy" not in d:
            print(tool, json.dumps(d))
            continue
        print(f"{tool} | {d['items']} | {d['accuracy']:.4f} | {d['accuracy_by_task']['T2']:.4f} | "
              f"{d['accuracy_by_task']['T3']:.4f} | {d['accuracy_by_split']['test']:.4f} | "
              f"{d['median_compute_ms']} | {d['mean_payload_bytes']}")


def step3():
    for regime in ["good", "weak", "poor"]:
        policy_table("step3/metrics.json", regime)
    for name in ["metrics_jetson5.json", "metrics_jetson35.json"]:
        policy_table(f"step3/{name}")


def step4():
    for regime in ["good", "weak", "poor"]:
        policy_table("step4/metrics.json", regime)
    print("quality_test", json.dumps(load("step4/metrics.json")["quality_test"]))


def step4b():
    for regime in ["good", "weak", "poor"]:
        policy_table("step4b/metrics.json", regime)
    print("selected_on_val", json.dumps(load("step4b/metrics.json")["selected_on_val"]))


def planA():
    d = load("planA/metrics.json")
    print(f"static_cascade risk_per_1000={d['static_cascade']['risk_per_1000']:.2f}")
    print("model | fraction | mean | std | seeds")
    for model, curve in d["curves"].items():
        for fraction, c in curve.items():
            print(f"{model} | {fraction} | {c['mean']:.2f} | {c['std']:.2f} | {c['seeds']}")


def planF():
    for name in ["metrics_label.json", "metrics_component.json", "metrics_component_budget0.05.json"]:
        print(f"[planF/{name}]")
        for policy, d in load(f"planF/{name}")["policies"].items():
            print(policy, json.dumps(d["summary"]))
        print()
    for kind in ["label", "component"]:
        print(f"value_check_{kind}", json.dumps(load(f"planF/value_check_{kind}.json")))
    parse = load("planF/laya_zero_shot_parse.json")
    print("laya_zero_shot_parse", json.dumps({k: {m: v for m, v in d.items() if not isinstance(v, list)}
                                               for k, d in parse.items()})[:2000])


STAGES = {"step1": step1, "step2": step2, "step3": step3, "step4": step4, "step4b": step4b, "planA": planA, "planF": planF}

if __name__ == "__main__":
    names = sys.argv[1:] or list(STAGES)
    for name in names:
        print(f"===== {name} =====")
        STAGES[name]()
        print()
