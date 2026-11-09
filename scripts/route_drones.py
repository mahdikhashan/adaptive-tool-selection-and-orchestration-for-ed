"""PPO tool selection on heterogeneous drones. Each drone has its own component group (or tower group)
and its own LTE regime. Train on val items (out-of-fold fusion), test on test items of the same drone.
Methods: ResNet only, all tools that fit, fixed cascade, cascade tuned on val (one global rule, one rule
per drone), oracle subset (labels), central PPO, local PPO, FedAvg PPO, FedAvg PPO + local fine-tune."""
import argparse
import itertools
import json

import numpy as np
import torch

from fedorch.common import results_dir
from fedorch.folds import tower_buckets
from fedorch.route_env import BAYES, Data, Episode, Fusion, em_prior, out_of_fold
from route_ppo import Actor, Critic, Env, all_that_fit, cascade, collect, greedy, oracle, run_policy, summarize, update

GROUPS = {"glass": ("glass",), "polymer": ("polymer",), "hardware": ("damper", "spacer", "plate"),
          "yoke": ("yoke",), "rod-grip": ("lightning", "vari-grip")}
ORDERS = list(itertools.permutations((1, 2, 3)))
MARGINS = (0., .02, .05, .1, .2, .3, .45)


def rule(order, margin, k):
    def choose(ep):
        if abs(ep.p_now() - BAYES) > margin or len(ep.called) >= k:
            return 0
        m = ep.mask()
        return next((a for a in order if m[a]), 0)
    return choose


def drone_of(items, split):
    if split == "towers":
        buckets = {s: tower_buckets([i for i in items if i["split"] == s], len(GROUPS)) for s in ("val", "test")}
        return lambda i: buckets[i["split"]][i["tower"]]
    names = list(GROUPS)
    return lambda i: next(n for n, g in enumerate(names) if i["asset"].startswith(GROUPS[g]))


def reward(out, lam, budget):
    return -float(np.mean([o["risk"] + lam * o["time"] / budget for o in out]))


def fleet(per_drone, sizes):
    keys = per_drone[0].keys()
    total = sum(sizes)
    return {k: round(sum(p[k] * s for p, s in zip(per_drone, sizes)) / total, 3) for k in keys}


def evaluate(envs, rows, budgets, choosers):
    out = {}
    for b in budgets:
        drones = [summarize(run_policy(envs[d](), rows[d], b, choosers[d] if isinstance(choosers, list) else choosers)) for d in range(len(rows))]
        out[str(b)] = {"fleet": fleet(drones, [len(r) for r in rows]), "drones": drones}
    return out


def per_budget(envs, rows, budgets, pick):
    out = {}
    for b in budgets:
        drones = [summarize(run_policy(envs[d](), rows[d], b, pick(d, b))) for d in range(len(rows))]
        out[str(b)] = {"fleet": fleet(drones, [len(r) for r in rows]), "drones": drones}
    return out


def tune(envs, rows, budgets, lam):
    configs = [(o, m, k) for o in ORDERS for m in MARGINS for k in (1, 2, 3)]
    score = {(d, b, c): reward(run_policy(envs[d](), rows[d], b, rule(*c)), lam, b) for d in range(len(rows)) for b in budgets for c in configs}
    local = {(d, b): max(configs, key=lambda c: score[d, b, c]) for d in range(len(rows)) for b in budgets}
    shared = {b: max(configs, key=lambda c: sum(score[d, b, c] * len(rows[d]) for d in range(len(rows)))) for b in budgets}
    return local, shared


def collect_fleet(envs, actor, critics, rows, rngs, episodes, sizes):
    batches = []
    for d in range(len(rows)):
        batches.append(collect(envs[d], actor, critics[d], rows[d], rngs[d], max(16, episodes * sizes[d] // sum(sizes))))
    return batches


def show(name, res):
    print(f"{name:34s}", " ".join(f"{b}: {v['fleet']['risk_per_1000']:6.1f}/{v['fleet']['mean_time_s']:.2f}s" for b, v in res.items()), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("components", "towers"), default="components")
    parser.add_argument("--regimes", default="poor,good,weak,poor,good")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[.25, .75, 1.5, 3.])
    parser.add_argument("--lambda-time", type=float, default=.01)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--finetune-rounds", type=int, default=10)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    args = parser.parse_args()
    regimes, lam, budgets = args.regimes.split(","), args.lambda_time, args.budgets
    n = len(GROUPS)
    data = Data()
    owner = drone_of(data.items, args.split)
    val = [[i for i in data.items if i["split"] == "val" and owner(i) == d] for d in range(n)]
    test = [[i for i in data.items if i["split"] == "test" and owner(i) == d] for d in range(n)]
    all_val = [i for r in val for i in r]
    oof = out_of_fold(data, all_val)
    table = Fusion(data).fit(all_val).table([i for r in test for i in r])
    pooled = float(np.mean([i["label"] != "good" for i in all_val]))
    fleet_em = em_prior(np.array([table[i["item_id"]][("laya", "smolvlm_local", "cloud")] for r in test for i in r]), pooled)
    prior_val, prior_test = [pooled] * n, [fleet_em] * n
    sizes = [len(r) for r in val]
    print(f"split {args.split} | regimes {regimes} | lambda {lam} | defect share val {pooled:.3f}, test fleet EM {fleet_em:.3f}")
    for d in range(n):
        print(f"  drone {d} ({list(GROUPS)[d] if args.split == 'components' else 'towers'}): val {len(val[d])} test {len(test[d])} "
              f"defect share val {np.mean([i['label'] != 'good' for i in val[d]]):.3f} test {np.mean([i['label'] != 'good' for i in test[d]]):.3f}", flush=True)
    train_envs = [lambda d=d, s=0: Env(data, val[d], oof, prior_val[d], prior_val[d], budgets, regimes[d], 500 + d + s, lam) for d in range(n)]
    test_envs = [lambda d=d: Env(data, test[d], table, prior_val[d], prior_test[d], budgets, regimes[d], 10_000 + d, lam) for d in range(n)]
    report = {"split": args.split, "regimes": regimes, "budgets": budgets, "lambda_time": lam,
              "prior_val": pooled, "prior_test_fleet_em": fleet_em, "drones": [{"val": len(val[d]), "test": len(test[d])} for d in range(n)],
              "methods": {}}
    path = results_dir("route") / f"drones_{args.split}_lambda{lam}.json"
    m = report["methods"]
    for name, choose in (("ResNet only", lambda ep: 0), ("all tools that fit", all_that_fit), ("fixed cascade", cascade)):
        m[name] = evaluate(test_envs, test, budgets, choose)
        show(name, m[name])
    local, shared = tune(train_envs, val, budgets, lam)
    m["cascade tuned, one rule"] = per_budget(test_envs, test, budgets, lambda d, b: rule(*shared[b]))
    m["cascade tuned per drone"] = per_budget(test_envs, test, budgets, lambda d, b: rule(*local[d, b]))
    report["tuned_rules"] = {"one": {str(b): shared[b] for b in budgets}, "per_drone": {f"{d}@{b}": local[d, b] for d in range(n) for b in budgets}}
    show("cascade tuned, one rule", m["cascade tuned, one rule"])
    show("cascade tuned per drone", m["cascade tuned per drone"])
    m["oracle subset (labels)"] = {}
    for b in budgets:
        drones = [summarize(oracle(test_envs[d](), test[d], b)) for d in range(n)]
        m["oracle subset (labels)"][str(b)] = {"fleet": fleet(drones, [len(r) for r in test]), "drones": drones}
    show("oracle subset (labels)", m["oracle subset (labels)"])
    path.write_text(json.dumps(report, indent=2))
    dim = len(Episode(data, oof, all_val[0], 1., 1., prior_val[0], prior_val[0]).features())
    for seed in args.seeds:
        envs = [train_envs[d](s=100 * seed) for d in range(n)]
        rngs = [np.random.default_rng(1000 * seed + d) for d in range(n)]
        torch.manual_seed(seed)
        actor, critic = Actor(dim), Critic(dim)
        for _ in range(args.rounds):
            update(actor, critic, [ep for b in collect_fleet(envs, actor, [critic] * n, val, rngs, args.episodes, sizes) for ep in b])
        m[f"central PPO seed {seed}"] = evaluate(test_envs, test, budgets, greedy(actor))
        show(f"central PPO seed {seed}", m[f"central PPO seed {seed}"])
        torch.manual_seed(seed)
        locals_ = []
        for d in range(n):
            a, c = Actor(dim), Critic(dim)
            for _ in range(args.rounds):
                update(a, c, collect(envs[d], a, c, val[d], rngs[d], max(16, args.episodes * sizes[d] // sum(sizes))))
            locals_.append(a)
        m[f"local PPO seed {seed}"] = evaluate(test_envs, test, budgets, [greedy(a) for a in locals_])
        show(f"local PPO seed {seed}", m[f"local PPO seed {seed}"])
        torch.manual_seed(seed)
        global_actor, critics = Actor(dim), [Critic(dim) for _ in range(n)]
        for _ in range(args.rounds):
            states = []
            for d in range(n):
                a = Actor(dim)
                a.load_state_dict(global_actor.state_dict())
                update(a, critics[d], collect(envs[d], a, critics[d], val[d], rngs[d], max(16, args.episodes * sizes[d] // sum(sizes))))
                states.append(a.state_dict())
            global_actor.load_state_dict({k: sum(s[k] * w for s, w in zip(states, sizes)) / sum(sizes) for k in states[0]})
        m[f"FedAvg PPO seed {seed}"] = evaluate(test_envs, test, budgets, greedy(global_actor))
        show(f"FedAvg PPO seed {seed}", m[f"FedAvg PPO seed {seed}"])
        tuned = []
        for d in range(n):
            a = Actor(dim)
            a.load_state_dict(global_actor.state_dict())
            for _ in range(args.finetune_rounds):
                update(a, critics[d], collect(envs[d], a, critics[d], val[d], rngs[d], max(16, args.episodes * sizes[d] // sum(sizes))))
            tuned.append(a)
        m[f"FedAvg + local fine-tune seed {seed}"] = evaluate(test_envs, test, budgets, [greedy(a) for a in tuned])
        show(f"FedAvg + local fine-tune seed {seed}", m[f"FedAvg + local fine-tune seed {seed}"])
        report["actor_parameters"] = sum(p.numel() for p in global_actor.parameters())
        path.write_text(json.dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
