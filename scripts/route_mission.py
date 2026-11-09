"""PPO tool selection over mission segments: one time budget and one deferral cap per segment, network
outages that persist in mission time. Train on val towers (out-of-fold per-subset fusion), test on test towers.
Rules: greedy (spend time and deferrals as items come) and paced (time per remaining item, deferral threshold),
both tuned on val. Learners: central PPO and FedAvg PPO over five tower-group drones."""
import argparse
import itertools
import json

import numpy as np
import torch

from fedorch.common import results_dir
from fedorch.folds import tower_buckets
from fedorch.mission_env import Segment, fixed_segments, random_segment, tower_runs
from fedorch.route_env import Data, Fusion, Network, em_prior, out_of_fold
from route_ppo import Actor, Critic, update

TOOL_ORDER = (1, 2, 3)
PACES = (.5, 1., 2., 4., np.inf)
TAUS = (.5, .6, .7, .8, .9)


def tool_time(seg, a):
    return seg.ep.cost(["laya", "smolvlm_local", "cloud"][a - 1])


def paced(pace, tau):
    def choose(seg):
        ep, m = seg.ep, seg.mask()
        if ep.report_risk() <= .5:
            return 0
        left = len(seg.items) - seg.n
        allowance = pace * max(0., seg.budget - seg.elapsed - seg.reserve()) / left
        spent = ep.elapsed - ep.t["classifier"]
        tool = next((a for a in TOOL_ORDER if m[a] and spent + tool_time(seg, a) <= allowance), 0)
        if tool:
            return tool
        return 4 if m[4] and ep.report_risk() > tau else 0
    return choose


def run(segments, make, choose):
    logs, used = [], []
    for n, items in enumerate(segments):
        seg = make(items, n)
        done = False
        while not done:
            _, done, _ = seg.step(choose(seg))
        logs += seg.log
        used.append(seg.elapsed / seg.budget)
    return {"risk_per_1000": round(1000 * float(np.mean([o["risk"] for o in logs])), 1),
            "mean_time_s": round(float(np.mean([o["time"] for o in logs])), 3),
            "deferred": round(float(np.mean([o["deferred"] for o in logs])), 4),
            "budget_used": round(float(np.mean(used)), 3),
            **{f"calls_{t}": round(float(np.mean([t in o["called"] for o in logs])), 3) for t in ("laya", "smolvlm_local", "cloud")}}


def greedy_policy(actor):
    def choose(seg):
        with torch.no_grad():
            logits = actor(torch.from_numpy(seg.features())).masked_fill(~torch.from_numpy(seg.mask()), -1e9)
        return int(logits.argmax())
    return choose


def collect(make, actor, critic, runs, rng, n_segments, length, budgets):
    batch = []
    for _ in range(n_segments):
        seg = make(random_segment(runs, length, rng), budgets[rng.integers(len(budgets))], rng)
        steps, done = [], False
        while not done:
            x, m = torch.from_numpy(seg.features()), torch.from_numpy(seg.mask())
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(x).masked_fill(~m, -1e9))
                a = int(dist.sample())
                steps.append(dict(x=x, m=m, a=a, logp=float(dist.log_prob(torch.tensor(a))), v=float(critic(x))))
            reward, done, _ = seg.step(a)
            steps[-1]["r"] = reward
        batch.append(steps)
    return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[.3, .75, 1.5])
    parser.add_argument("--cap", type=float, default=.05, help="maximum share of items deferred per segment")
    parser.add_argument("--length", type=int, default=40, help="items per segment")
    parser.add_argument("--regime", default="weak")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--segments", type=int, default=256, help="training segments per round")
    parser.add_argument("--gamma", type=float, default=1.)
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--gae", type=float, default=None, help="GAE lambda; default: Monte-Carlo returns")
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    parser.add_argument("--fed", action="store_true", help="also train FedAvg PPO over five tower-group drones")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    data = Data()
    val = [i for i in data.items if i["split"] == "val"]
    test = [i for i in data.items if i["split"] == "test"]
    prior_val = float(np.mean([i["label"] != "good" for i in val]))
    oof = out_of_fold(data, val, per_subset=True)
    table = Fusion(data, per_subset=True).fit(val).table(test)
    prior_test = em_prior(np.array([table[i["item_id"]][("laya", "smolvlm_local", "cloud")] for i in test]), prior_val)
    train_trace, test_trace = Network(args.regime, 1).trace, Network(args.regime, 10_000).trace
    val_segments, test_segments = fixed_segments(val, args.length), fixed_segments(test, args.length)
    starts = np.random.default_rng(10_000).integers(len(test_trace), size=len(test_segments))
    val_starts = np.random.default_rng(1).integers(len(train_trace), size=len(val_segments))

    def test_make(b):
        return lambda items, n: Segment(data, table, items, b, args.cap, test_trace, starts[n], prior_val, prior_test)

    def val_make(b):
        return lambda items, n: Segment(data, oof, items, b, args.cap, train_trace, val_starts[n], prior_val, prior_val)

    def train_make(items, b, rng):
        return Segment(data, oof, items, b, args.cap, train_trace, rng.integers(len(train_trace)), prior_val, prior_val)

    report = {"budgets_per_item": args.budgets, "cap": args.cap, "length": args.length, "regime": args.regime,
              "test_segments": len(test_segments), "methods": {}}
    m = report["methods"]
    path = results_dir("route") / f"mission{args.tag}.json"

    def show(name):
        print(f"{name:34s}", " ".join(f"{b}: {v['risk_per_1000']:6.1f} t {v['mean_time_s']:.2f} d {v['deferred']:.3f}" for b, v in m[name].items()), flush=True)

    rules = {"ResNet only": lambda seg: 0,
             "greedy: defer if uncertain": lambda seg: 4 if seg.mask()[4] and seg.ep.report_risk() > .5 else 0,
             "greedy cascade with defer": paced(np.inf, .5)}
    for name, choose in rules.items():
        m[name] = {str(b): run(test_segments, test_make(b), choose) for b in args.budgets}
        show(name)
    tuned = {}
    for b in args.budgets:
        tuned[b] = min(itertools.product(PACES, TAUS), key=lambda c: run(val_segments, val_make(b), paced(*c))["risk_per_1000"])
    m["paced cascade, tuned on val"] = {str(b): run(test_segments, test_make(b), paced(*tuned[b])) for b in args.budgets}
    report["tuned_rule"] = {str(b): [float(c) for c in tuned[b]] for b in args.budgets}
    show("paced cascade, tuned on val")
    path.write_text(json.dumps(report, indent=2, default=float))
    dim = len(Segment(data, oof, val[:2], 1., args.cap, train_trace, 0, prior_val, prior_val).features())
    runs = tower_runs(val)
    buckets = tower_buckets(val, 5)
    drone_runs = [[r for r in runs if buckets[r[0]["tower"]] == d] for d in range(5)]
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        actor, critic = Actor(dim), Critic(dim)
        for r in range(args.rounds):
            update(actor, critic, collect(train_make, actor, critic, runs, rng, args.segments, args.length, args.budgets),
                   entropy=args.entropy, gamma=args.gamma, gae=args.gae)
        m[f"central PPO seed {seed}"] = {str(b): run(test_segments, test_make(b), greedy_policy(actor)) for b in args.budgets}
        show(f"central PPO seed {seed}")
        path.write_text(json.dumps(report, indent=2, default=float))
        if not args.fed:
            continue
        torch.manual_seed(seed)
        global_actor, critics = Actor(dim), [Critic(dim) for _ in range(5)]
        rngs = [np.random.default_rng(100 * seed + d) for d in range(5)]
        sizes = [sum(len(x) for x in dr) for dr in drone_runs]
        for r in range(args.rounds):
            states = []
            for d in range(5):
                local = Actor(dim)
                local.load_state_dict(global_actor.state_dict())
                update(local, critics[d], collect(train_make, local, critics[d], drone_runs[d], rngs[d],
                                                  max(8, args.segments * sizes[d] // sum(sizes)), args.length, args.budgets),
                       entropy=args.entropy, gamma=args.gamma, gae=args.gae)
                states.append(local.state_dict())
            global_actor.load_state_dict({k: sum(s[k] * w for s, w in zip(states, sizes)) / sum(sizes) for k in states[0]})
        m[f"FedAvg PPO seed {seed}"] = {str(b): run(test_segments, test_make(b), greedy_policy(global_actor)) for b in args.budgets}
        show(f"FedAvg PPO seed {seed}")
        path.write_text(json.dumps(report, indent=2, default=float))
    print("done", flush=True)


if __name__ == "__main__":
    main()
