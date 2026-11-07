import argparse
import copy
import importlib.util
import random
from pathlib import Path

import torch

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import ACTIONS, action_risks, choice_rule, load_evidence, numeric_v2, thresholds
from fedorch.fedcontrol import average, federated_norm, local_steps, regions, update_kb
from fedorch.feedback import (LEARNERS, Confusion, behaviour, reveal, target_for, truth_propensity, weight_for,
                              weighted_soft_cross_entropy)
from fedorch.items import load_items
from fedorch.network import lte_mbps
from fedorch.policies import THRESHOLDS, cascade, to_local
from fedorch.replay import evaluate, mission, trace_for, tune

REGIONS, ROUNDS, SUB_ROUNDS, STEPS = 5, 10, 20, 10
TEMPERATURE, EPSILON, REVIEW = 1.0, 0.05, 0.5

spec = importlib.util.spec_from_file_location("step4b_mlp", Path(__file__).with_name("step4b_train_mlp.py"))
step4b_mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4b_mlp)


def loss_fn(logits, target_weight):
    return weighted_soft_cross_entropy(logits, *target_weight)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--regime", default="weak")
    parser.add_argument("--learners", default=",".join(LEARNERS))
    parser.add_argument("--audits", default="0,0.01,0.05")
    parser.add_argument("--modes", default="fed")
    parser.add_argument("--tag", default="main")
    args = parser.parse_args()

    items, cache, evidence, base = load_items(), load_cache(), load_evidence(), lte_mbps()
    assets = sorted({i["asset"] for i in items})
    order = lambda rows: sorted(rows, key=lambda i: (i["tower"], i["parent_id"], i["item_id"]))
    streams = [order(r) for r in regions([i for i in items if i["split"] == "train"], REGIONS)]
    val, test = mission("val", items), mission("test", items)
    val_ids = [i["item_id"] for i in val]
    traces = [trace_for(test, args.regime, s, base) for s in range(5)]

    def features(rows):
        return torch.tensor([numeric_v2(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]], assets) for i in rows])

    x_stream = [features(s) for s in streams]
    mean, std = federated_norm(x_stream)
    x_stream = [(x - mean) / std for x in x_stream]
    x_eval = {"val": (features(val) - mean) / std, "test": (features(test) - mean) / std}
    threshold, _ = tune(lambda t: cascade(t, to_local), THRESHOLDS, val, cache, args.regime, base)
    width = x_stream[0].shape[1]

    def replay(model):
        with torch.no_grad():
            p = {i["item_id"]: dict(zip(ACTIONS, map(float, torch.softmax(model(x_eval[s]), -1)[n])))
                 for s, rows in [("val", val), ("test", test)] for n, i in enumerate(rows)}
        margin, _ = tune(lambda m: choice_rule(p, cache, m), thresholds(p, val_ids), val, cache, args.regime, base)
        return evaluate(choice_rule(p, cache, margin), test, cache, traces)

    out = results_dir("fc2") / f"metrics_{args.tag}_{args.regime}_s{args.seed}.json"
    results = {"seed": args.seed, "regime": args.regime, "epsilon": EPSILON, "review": REVIEW, "rounds": ROUNDS,
               "cascade_threshold": threshold, "runs": {}}
    for mode in args.modes.split(","):
        for learner in args.learners.split(","):
            for audit in map(float, args.audits.split(",")):
                rng = random.Random(args.seed)
                torch.manual_seed(args.seed)
                models = [step4b_mlp.mlp(width, len(ACTIONS))] * REGIONS if mode == "fed" else \
                    [step4b_mlp.mlp(width, len(ACTIONS)) for _ in range(REGIONS)]
                buffers = [([], [], []) for _ in range(REGIONS)]
                confusion = Confusion()
                stats = {"frames": 0, "stream_risk": 0.0, "truth_labels": 0, "cloud_labels": 0, "cloud_uploads": 0,
                         "audit_uploads": 0, "upload_mb": 0.0}
                for k in range(ROUNDS):
                    for r, stream in enumerate(streams):
                        size = len(stream) // ROUNDS
                        chunk = range(k * size, (k + 1) * size if k < ROUNDS - 1 else len(stream))
                        with torch.no_grad():
                            p_model = torch.softmax(models[r](x_stream[r][list(chunk)]), -1)
                        for j, n in enumerate(chunk):
                            item, rows = stream[n], cache[stream[n]["item_id"]]
                            fallback = "smolvlm_local" if rows["classifier"]["confidence"] < threshold else "accept"
                            greedy = ACTIONS[int(p_model[j].argmax())] if k > 0 else None
                            probs = behaviour(greedy, fallback, EPSILON)
                            action = rng.choices(ACTIONS, probs)[0]
                            kind = reveal(action, rng, audit, REVIEW)
                            stats["frames"] += 1
                            stats["stream_risk"] += action_risks(item, rows)[action]
                            uploaded = action == "cloud" or kind == "truth"
                            stats["cloud_uploads"] += action == "cloud"
                            stats["audit_uploads"] += action != "cloud" and kind == "truth"
                            stats["upload_mb"] += rows["cloud"]["payload_bytes"] / 1e6 if uploaded else 0.0
                            if kind == "truth":
                                confusion.add(item["label"], rows["cloud"]["prediction"])
                            target = target_for(learner, item, rows, kind, confusion, TEMPERATURE)
                            if target is None:
                                continue
                            stats["truth_labels" if kind == "truth" or learner == "full" else "cloud_labels"] += 1
                            prop = truth_propensity(probs[ACTIONS.index("cloud")], audit, REVIEW)
                            buffers[r][0].append(n)
                            buffers[r][1].append([target[a] for a in ACTIONS])
                            buffers[r][2].append(weight_for(learner, kind, prop))
                    data = [(x_stream[r][b[0]], (torch.tensor(b[1]), torch.tensor(b[2])))
                            for r, b in enumerate(buffers) if b[0]]
                    if not data:
                        continue
                    if mode == "fed":
                        model = models[0]
                        for _ in range(SUB_ROUNDS):
                            states = [local_steps(copy.deepcopy(model), loss_fn, x, y, STEPS).state_dict() for x, y in data]
                            model.load_state_dict(average(states, [len(x) for x, _ in data]))
                        models = [model] * REGIONS
                    else:
                        for r, b in enumerate(buffers):
                            if b[0]:
                                local_steps(models[r], loss_fn, x_stream[r][b[0]], (torch.tensor(b[1]), torch.tensor(b[2])),
                                            SUB_ROUNDS * STEPS)
                for m in models:
                    m.eval()
                evals = [replay(models[0])] if mode == "fed" else [replay(m) for m in models]
                result = {k: sum(e[k] for e in evals) / len(evals) for k in evals[0]}
                stats["stream_risk_per_1000"] = 1000 * stats.pop("stream_risk") / stats["frames"]
                name = f"{mode}_{learner}_a{audit}"
                results["runs"][name] = {"mode": mode, "learner": learner, "audit": audit, "update_kb": update_kb(models[0]),
                                         "uplink_kb_model": update_kb(models[0]) * REGIONS * ROUNDS * SUB_ROUNDS if mode == "fed" else 0,
                                         **stats, **result,
                                         "per_region_risk": [round(e["risk_per_1000"], 2) for e in evals] if mode != "fed" else None}
                print(f"{name:28s} s{args.seed} test risk {result['risk_per_1000']:7.1f} esc {result['escalation_rate']:.3f} "
                      f"truth {stats['truth_labels']} cloud {stats['cloud_labels']} stream {stats['stream_risk_per_1000']:.1f}", flush=True)
                write_json(out, results)


if __name__ == "__main__":
    main()
