"""Train base laya as the agent with PPO on the agent cache; evaluate on test images.

Modes: zeroshot (base laya, writes the report at once), report_only (PPO, no tools: learns to answer),
central (PPO with tools, all val images), fed (Flower FedAvg/FedProx over five tower-split drones).
"""
import argparse
import json
import time

import numpy as np
import torch

from fedorch.common import results_dir, runs_dir
from fedorch.laya_agent import Cache, Critic, Policy, drones, evaluate, params, rollout, set_params, update


def train_local(policy, critic, cache, indices, args, rounds, tools=True, rng=None, prox=0.):
    rng = rng or np.random.default_rng(args.seed)
    for _ in range(rounds):
        anchor = [p.detach().clone() for p in policy.parameters()]
        picks = rng.choice(indices, size=args.episodes, replace=True)
        batch, _ = rollout(policy, critic, cache, picks, args.budgets, rng, tools=tools)
        info = update(policy, critic, batch, prox=prox, anchor=anchor)
    return info


def run_fed(cache, train, test, args, out):
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg, FedProx
    from flwr.simulation import run_simulation
    parts = drones(cache, train)
    base = Policy(cache.scorer, split=not args.shared_scorer)
    history = []

    class Drone(NumPyClient):
        def __init__(self, d):
            self.d = d

        def fit(self, parameters, config):
            policy = Policy(cache.scorer, split=not args.shared_scorer)
            set_params(policy, parameters)
            critic_path = out / f"critic-drone{self.d}.pt"
            critic = Critic()
            if critic_path.exists():
                critic.load_state_dict(torch.load(critic_path))
            rng = np.random.default_rng(args.seed + 1000 * self.d + int(config.get("round", 0)))
            info = train_local(policy, critic, cache, parts[self.d], args, args.local_rounds, rng=rng,
                               prox=float(config.get("proximal_mu", 0.)))
            torch.save(critic.state_dict(), critic_path)
            return params(policy), len(parts[self.d]), {"drone": self.d, **{k: float(v) for k, v in info.items()}}

    def evaluate_fn(server_round, arrays, config):
        if server_round == 0:
            return None
        policy = Policy(cache.scorer, split=not args.shared_scorer)
        set_params(policy, arrays)
        report = evaluate(policy, cache, test, args.budgets)
        history.append({"round": server_round, "test": report})
        print(json.dumps({"fed_round": server_round, "test": report}), flush=True)
        return 0., {}

    class Clients:
        def __call__(self, context):
            return Drone(int(context.node_config["partition-id"])).to_client()

    strategy_type = FedProx if args.strategy == "fedprox" else FedAvg
    extra = {"proximal_mu": args.proximal_mu} if args.strategy == "fedprox" else {}
    strategy = strategy_type(fraction_fit=1., fraction_evaluate=0., min_fit_clients=5, min_available_clients=5,
                             initial_parameters=ndarrays_to_parameters(params(base)), evaluate_fn=evaluate_fn,
                             on_fit_config_fn=lambda r: {"round": r}, **extra)

    class Server:
        def __call__(self, context):
            return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=args.rounds))

    run_simulation(server_app=ServerApp(server_fn=Server()), client_app=ClientApp(client_fn=Clients()), num_supernodes=5,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0}})
    update_bytes = sum(a.nbytes for a in params(base))
    return {"history": history, "drone_images": [len(p) for p in parts],
            "uplink_bytes_per_drone_round_fp32": update_bytes, "uplink_total_mb_fp32": round(update_bytes * 5 * args.rounds / 1e6, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="zeroshot,report_only,central,fed")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[1., 3., 6.])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-rounds", type=int, default=2, help="PPO rollout/update cycles per drone per Flower round")
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--proximal-mu", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shared-scorer", action="store_true", help="one scorer for action and report (first run)")
    parser.add_argument("--cache", default=str(runs_dir("agent_cache")))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    from pathlib import Path
    cache = Cache(Path(args.cache))
    train = [i for i, m in enumerate(cache.meta) if m["split"] == "val"]
    test = [i for i, m in enumerate(cache.meta) if m["split"] == "test"]
    out = results_dir(f"agent_ppo/{'shared' if args.shared_scorer else 'split'}_seed{args.seed}")
    summary = {"train_images": len(train), "test_images": len(test), "budgets": args.budgets, "rounds": args.rounds}
    for mode in args.modes.split(","):
        start = time.time()
        if mode == "zeroshot":
            summary[mode] = evaluate(Policy(cache.scorer, split=not args.shared_scorer), cache, test, args.budgets, tools=False)
        elif mode in ("report_only", "central"):
            policy, critic, rng = Policy(cache.scorer, split=not args.shared_scorer), Critic(), np.random.default_rng(args.seed)
            curve = []
            for r in range(args.rounds):
                train_local(policy, critic, cache, train, args, args.local_rounds * 5, tools=mode == "central", rng=rng)
                curve.append({"round": r + 1, "test": evaluate(policy, cache, test, args.budgets, tools=mode == "central")})
                print(json.dumps({mode: r + 1, "test": curve[-1]["test"]}), flush=True)
            summary[mode] = curve
        elif mode == "fed":
            summary[mode] = run_fed(cache, train, test, args, out)
        summary[f"{mode}_seconds"] = round(time.time() - start, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done, wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
