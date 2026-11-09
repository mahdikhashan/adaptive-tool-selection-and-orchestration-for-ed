"""PPO tool selection with fused reports. Train on val towers (out-of-fold fusion), test on test towers.
Methods: ResNet only, all tools that fit, uncertainty cascade, oracle subset (labels), central PPO,
Flower FedAvg PPO over five tower-split drones. CPU only."""
import argparse
import json
import math
import time

import numpy as np
import torch
from torch import nn

from fedorch.common import results_dir
from fedorch.decision import answer_risk
from fedorch.folds import tower_buckets
from fedorch.route_env import ACTIONS, BAYES, EXTRA, SUBSETS, Data, Episode, Fusion, Network, corrected, em_prior, out_of_fold


class Actor(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, len(ACTIONS)))

    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class Env:
    def __init__(self, data, rows, table, prior_train, prior_new, budgets, regime, seed, lambda_time, defer=False):
        self.data, self.rows, self.table, self.defer = data, rows, table, defer
        self.prior_train, self.prior_new, self.budgets, self.lambda_time = prior_train, prior_new, budgets, lambda_time
        self.net = Network(regime, seed)

    def episode(self, item, budget, mbps=None):
        mbps = self.net.sample() if mbps is None else mbps
        return Episode(self.data, self.table, item, budget, mbps, self.prior_train, self.prior_new, self.lambda_time, self.defer)


def run_policy(env, items, budget, choose):
    out = []
    for item in items:
        ep = env.episode(item, budget)
        while True:
            action = choose(ep)
            _, done, risk = ep.step(action)
            if done:
                out.append({"risk": risk, "time": ep.elapsed, "called": ep.called, "bytes": ep.bytes, "deferred": action == 4})
                break
    return out


def summarize(out):
    return {"risk_per_1000": round(1000 * float(np.mean([o["risk"] for o in out])), 1),
            "mean_time_s": round(float(np.mean([o["time"] for o in out])), 3),
            "deferred": round(float(np.mean([o.get("deferred", False) for o in out])), 3),
            "uplink_kb_per_item": round(float(np.mean([o["bytes"] for o in out])) / 1000, 2),
            **{f"calls_{t}": round(float(np.mean([t in o["called"] for o in out])), 3) for t in EXTRA}}


def all_that_fit(ep):
    m = ep.mask()
    return next((a for a in (1, 2, 3) if m[a]), 0)


def cascade(ep):
    if abs(ep.p_now() - BAYES) > .1 or len(ep.called) >= 2:
        return 0
    return all_that_fit(ep)


def all_then_defer(ep):
    a = all_that_fit(ep)
    return a if a else (4 if ep.report_risk() > .5 else 0)


def cascade_defer(ep):
    if ep.report_risk() <= .5:
        return 0
    return all_that_fit(ep) or 4


def oracle_choice(env, item, budget, mbps):
    best = None
    for subset in SUBSETS:
        ep = env.episode(item, budget, mbps)
        ok = True
        for tool in subset:
            if not ep.mask()[1 + EXTRA.index(tool)]:
                ok = False
                break
            ep.step(1 + EXTRA.index(tool))
        if not ok:
            continue
        reward, _, _ = Episode.step(ep, 0)
        if best is None or reward > best[0]:
            best = (reward, subset)
    return best[1]


def oracle(env, items, budget):
    out = []
    for item in items:
        mbps = env.net.sample()
        subset = oracle_choice(env, item, budget, mbps)
        ep = env.episode(item, budget, mbps)
        for tool in subset:
            ep.step(1 + EXTRA.index(tool))
        _, _, risk = ep.step(0)
        out.append({"risk": risk, "time": ep.elapsed, "called": ep.called, "bytes": ep.bytes})
    return out


def collect(env, actor, critic, items, rng, episodes):
    batch = []
    for n in rng.integers(len(items), size=episodes):
        ep = env.episode(items[n], env.budgets[rng.integers(len(env.budgets))])
        steps = []
        while True:
            x = torch.from_numpy(ep.features())
            m = torch.from_numpy(ep.mask())
            with torch.no_grad():
                dist = torch.distributions.Categorical(logits=actor(x).masked_fill(~m, -1e9))
                a = int(dist.sample())
                steps.append(dict(x=x, m=m, a=a, logp=float(dist.log_prob(torch.tensor(a))), v=float(critic(x))))
            reward, done, _ = ep.step(a)
            steps[-1]["r"] = reward
            if done:
                break
        batch.append(steps)
    return batch


def update(actor, critic, batch, lr=3e-4, epochs=4, clip=.2, entropy=.01, gamma=1., gae=None, anchor=0.):
    steps, ret, advs = [], [], []
    for ep in batch:
        g, a, next_v, tail, tail_a = 0., 0., 0., [], []
        for s in reversed(ep):
            g = s["r"] + gamma * g
            if gae is not None:
                a = s["r"] + gamma * next_v - s["v"] + gamma * gae * a
                next_v = s["v"]
            tail.append(g)
            tail_a.append(a)
        ret.extend(reversed(tail))
        advs.extend(reversed(tail_a))
        steps.extend(ep)
    values = torch.tensor([s["v"] for s in steps])
    if gae is None:
        ret = torch.tensor(ret, dtype=torch.float32)
        adv = ret - values
    else:
        adv = torch.tensor(advs, dtype=torch.float32)
        ret = adv + values
    adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
    x = torch.stack([s["x"] for s in steps])
    m = torch.stack([s["m"] for s in steps])
    a = torch.tensor([s["a"] for s in steps])
    old = torch.tensor([s["logp"] for s in steps])
    teacher = torch.tensor([s.get("t", 0) for s in steps])
    opt_a, opt_c = torch.optim.Adam(actor.parameters(), lr=lr), torch.optim.Adam(critic.parameters(), lr=1e-3)
    for _ in range(epochs):
        for ids in torch.randperm(len(steps)).split(512):
            dist = torch.distributions.Categorical(logits=actor(x[ids]).masked_fill(~m[ids], -1e9))
            ratio = (dist.log_prob(a[ids]) - old[ids]).exp()
            loss = -torch.minimum(ratio * adv[ids], ratio.clamp(1 - clip, 1 + clip) * adv[ids]).mean() - entropy * dist.entropy().mean()
            if anchor:
                loss = loss - anchor * dist.log_prob(teacher[ids]).mean()
            opt_a.zero_grad(); loss.backward(); opt_a.step()
            vloss = ((critic(x[ids]) - ret[ids]) ** 2).mean()
            opt_c.zero_grad(); vloss.backward(); opt_c.step()


def greedy(actor):
    def choose(ep):
        with torch.no_grad():
            logits = actor(torch.from_numpy(ep.features())).masked_fill(~torch.from_numpy(ep.mask()), -1e9)
        return int(logits.argmax())
    return choose


def evaluate(env, items, budgets, choose):
    return {str(b): summarize(run_policy(env, items, b, choose)) for b in budgets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[.25, .75, 1.5, 3.])
    parser.add_argument("--regime", default="weak")
    parser.add_argument("--lambda-time", type=float, default=.2)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=2048)
    parser.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=[0, 1, 2])
    parser.add_argument("--fed-rounds", type=int, default=30)
    parser.add_argument("--local-cycles", type=int, default=1)
    parser.add_argument("--oracle-only", action="store_true", help="recompute the oracle into an existing summary")
    parser.add_argument("--defer", action="store_true", help="add the DEFER action (operator review, risk 0.5)")
    parser.add_argument("--per-subset", action="store_true", help="fit one fusion model per tool subset")
    parser.add_argument("--tag", default="")
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    data = Data()
    val = [i for i in data.items if i["split"] == "val"]
    test = [i for i in data.items if i["split"] == "test"]
    prior_val = float(np.mean([i["label"] != "good" for i in val]))
    oof = out_of_fold(data, val, per_subset=args.per_subset)
    fusion = Fusion(data, per_subset=args.per_subset).fit(val)
    table = fusion.table(test)
    prior_test = em_prior(np.array([table[i["item_id"]][("laya", "smolvlm_local", "cloud")] for i in test]), prior_val)
    print(f"val {len(val)} items, test {len(test)} items | defect share val {prior_val:.3f}, test EM {prior_test:.3f}", flush=True)
    report = {"budgets": args.budgets, "regime": args.regime, "lambda_time": args.lambda_time, "prior_val": prior_val,
              "prior_test_em": prior_test, "methods": {}}
    test_env = lambda seed: Env(data, test, table, prior_val, prior_test, args.budgets, args.regime, 10_000 + seed, args.lambda_time, args.defer)
    env0 = test_env(0)
    baselines = [("ResNet only", lambda ep: 0), ("all tools that fit", all_that_fit), ("uncertainty cascade", cascade)]
    if args.defer:
        baselines += [("ResNet, defer if uncertain", lambda ep: 4 if ep.report_risk() > .5 else 0),
                      ("all tools, defer if uncertain", all_then_defer), ("cascade with defer", cascade_defer)]
    for name, choose in baselines:
        report["methods"][name] = evaluate(test_env(0), test, args.budgets, choose)
        print(name, json.dumps({b: v["risk_per_1000"] for b, v in report["methods"][name].items()}), flush=True)
    report["methods"]["oracle subset (labels)"] = {str(b): summarize(oracle(test_env(0), test, b)) for b in args.budgets}
    print("oracle", json.dumps({b: (v["risk_per_1000"], v["mean_time_s"]) for b, v in report["methods"]["oracle subset (labels)"].items()}), flush=True)
    path = results_dir("route") / f"summary_lambda{args.lambda_time}{args.tag}.json"
    if args.oracle_only:
        saved = json.loads(path.read_text())
        saved["methods"]["oracle subset (labels)"] = report["methods"]["oracle subset (labels)"]
        path.write_text(json.dumps(saved, indent=2))
        return
    train_env = lambda seed: Env(data, val, oof, prior_val, prior_val, args.budgets, args.regime, seed, args.lambda_time, args.defer)
    dim = len(Episode(data, oof, val[0], 1., 1., prior_val, prior_val).features())
    for seed in args.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        actor, critic, env = Actor(dim), Critic(dim), train_env(seed)
        curve = []
        for r in range(args.rounds):
            update(actor, critic, collect(env, actor, critic, val, rng, args.episodes), lr=args.lr, entropy=args.entropy)
            if (r + 1) % 10 == 0:
                curve.append({"round": r + 1, "test": evaluate(test_env(seed), test, args.budgets, greedy(actor))})
                print(f"central seed {seed} round {r + 1}", json.dumps({b: (v["risk_per_1000"], v["mean_time_s"]) for b, v in curve[-1]["test"].items()}), flush=True)
        report["methods"][f"central PPO seed {seed}"] = curve
        path.write_text(json.dumps(report, indent=2))
        if not args.fed_rounds:
            continue
        buckets = tower_buckets(val, 5)
        drones = [[i for i in val if buckets[i["tower"]] == d] for d in range(5)]
        torch.manual_seed(seed)
        global_actor, critics = Actor(dim), [Critic(dim) for _ in range(5)]
        envs = [train_env(seed * 100 + d) for d in range(5)]
        rngs = [np.random.default_rng(seed * 100 + d) for d in range(5)]
        curve = []
        for r in range(args.fed_rounds):
            states, weights = [], []
            for d in range(5):
                local = Actor(dim)
                local.load_state_dict(global_actor.state_dict())
                for _ in range(args.local_cycles):
                    update(local, critics[d], collect(envs[d], local, critics[d], drones[d], rngs[d], args.episodes // 5))
                states.append(local.state_dict())
                weights.append(len(drones[d]))
            global_actor.load_state_dict({k: sum(s[k] * w for s, w in zip(states, weights)) / sum(weights) for k in states[0]})
            if (r + 1) % 10 == 0:
                curve.append({"round": r + 1, "test": evaluate(test_env(seed), test, args.budgets, greedy(global_actor))})
                print(f"fedavg seed {seed} round {r + 1}", json.dumps({b: (v["risk_per_1000"], v["mean_time_s"]) for b, v in curve[-1]["test"].items()}), flush=True)
        report["methods"][f"FedAvg PPO seed {seed}"] = curve
        report["actor_parameters"] = sum(p.numel() for p in global_actor.parameters())
        path.write_text(json.dumps(report, indent=2))
    print("done", flush=True)


if __name__ == "__main__":
    main()
