import importlib.util
import statistics
import sys
from pathlib import Path

import torch

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import ACTIONS, choice_rule, load_evidence, numeric_v2, soft_target, thresholds, training_items
from fedorch.fedcontrol import federated_norm, fedavg, regions, update_kb
from fedorch.items import load_items
from fedorch.network import lte_mbps
from fedorch.policies import THRESHOLDS, cascade, to_local
from fedorch.replay import evaluate, mission, trace_for, tune

REGIONS = 5
SEEDS = [0, 1, 2]
TEMPERATURE, OVERSAMPLE = 1.0, 5
REGIME = sys.argv[1] if len(sys.argv) > 1 else "weak"

spec = importlib.util.spec_from_file_location("step4b_mlp", Path(__file__).with_name("step4b_train_mlp.py"))
step4b_mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4b_mlp)


def main():
    items, cache, evidence, base = load_items(), load_cache(), load_evidence(), lte_mbps()
    assets = sorted({i["asset"] for i in items})
    split = {s: [i for i in items if i["split"] == s] for s in ["train", "val", "test"]}

    def features(rows):
        return torch.tensor([numeric_v2(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]], assets) for i in rows])

    def targets(rows):
        return torch.tensor([[soft_target(i, cache[i["item_id"]], TEMPERATURE)[a] for a in ACTIONS] for i in rows])

    loss_fn = step4b_mlp.soft_cross_entropy
    raw = {s: features(split[s]) for s in ["val", "test"]}
    y_val = targets(split["val"])
    parts = regions(split["train"], REGIONS)
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    traces = [trace_for(test, REGIME, seed, base) for seed in range(5)]

    def probabilities(model, mean, std):
        with torch.no_grad():
            return {i["item_id"]: dict(zip(ACTIONS, map(float, torch.softmax(model((raw[s] - mean) / std), -1)[n])))
                    for s in ["val", "test"] for n, i in enumerate(split[s])}

    def replay(p):
        margin, _ = tune(lambda m: choice_rule(p, cache, m), thresholds(p, val_ids), val, cache, REGIME, base)
        return evaluate(choice_rule(p, cache, margin), test, cache, traces)

    threshold, _ = tune(lambda t: cascade(t, to_local), THRESHOLDS, val, cache, REGIME, base)
    results = {"regime": REGIME, "regions": [{"frames": len(r), "towers": len({i["tower"] for i in r}),
                                              "defects": sum(i["label"] != "good" for i in r)} for r in parts],
               "static_cascade": evaluate(cascade(threshold, to_local), test, cache, traces), "runs": {}}

    for seed in SEEDS:
        torch.manual_seed(seed)
        local = [training_items(r, cache, OVERSAMPLE) for r in parts]
        x_local, y_local = [features(r) for r in local], [targets(r) for r in local]
        mean, std = federated_norm(x_local)

        def record(name, model, mean, std, extra):
            result = replay(probabilities(model, mean, std))
            results["runs"][f"{name}_s{seed}"] = {"method": name, "seed": seed, **extra, **result}
            print(f"{name:12s} s{seed} risk {result['risk_per_1000']:7.1f} escalation {result['escalation_rate']:.3f}", flush=True)

        model = step4b_mlp.mlp(x_local[0].shape[1], len(ACTIONS))
        x_all, y_all = torch.cat(x_local), torch.cat(y_local)
        step4b_mlp.fit(model, loss_fn, (x_all - mean) / std, y_all, (raw["val"] - mean) / std, y_val)
        record("central", model, mean, std, {"update_kb": 0.0})

        for r, (x, y) in enumerate(zip(x_local, y_local)):
            m, s = x.mean(0), x.std(0).clamp(min=1e-6)
            model = step4b_mlp.mlp(x.shape[1], len(ACTIONS))
            step4b_mlp.fit(model, loss_fn, (x - m) / s, y, (raw["val"] - m) / s, y_val)
            record(f"local_r{r}", model, m, s, {"region": r})

        for name, prox in [("fedavg", 0.0), ("fedprox", 0.01)]:
            torch.manual_seed(seed)
            model = step4b_mlp.mlp(x_local[0].shape[1], len(ACTIONS))
            clients = [((x - mean) / std, y) for x, y in zip(x_local, y_local)]
            info = fedavg(model, loss_fn, clients, (raw["val"] - mean) / std, y_val, prox=prox)
            record(name, model, mean, std, {"update_kb": update_kb(model), **info,
                                            "total_uplink_kb": update_kb(model) * REGIONS * info["rounds_run"]})

    summary = {}
    for run in results["runs"].values():
        key = "local" if run["method"].startswith("local_") else run["method"]
        summary.setdefault(key, []).append(run["risk_per_1000"])
    results["summary"] = {k: {"mean": round(statistics.mean(v), 2), "std": round(statistics.pstdev(v), 2), "n": len(v)}
                          for k, v in summary.items()}
    print(results["summary"], "cascade", results["static_cascade"]["risk_per_1000"])
    write_json(results_dir("fc1") / f"metrics_{REGIME}.json", results)


if __name__ == "__main__":
    main()
