"""Option 2: PPO trains base laya's head transformer, type embedding and two scorers (action, report)
on cached backbone outputs. Modes: zeroshot, report_only (no tools), central (tools), fed (Flower
FedAvg/FedProx over five tower-split drones, int8 update deltas). Train on val images, test on test images.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from fedorch.common import results_dir, runs_dir
from fedorch.laya_agent import drones
from fedorch.laya_head_agent import Critic, HeadCache, HeadPolicy, evaluate, rollout, update

CACHE = {}


def cache(args):
    if "c" not in CACHE:
        CACHE["c"] = HeadCache(Path(args.source), Path(args.cache))
    return CACHE["c"]


def train_cycles(policy, critic, c, indices, args, cycles, tools=True, rng=None, prox=0.):
    device = args.device
    rng = rng or np.random.default_rng(args.seed)
    for _ in range(cycles):
        anchor = [p.detach().clone() for p in policy.parameters()]
        episodes, _ = rollout(policy, critic, c, rng.choice(indices, size=args.episodes), args.budgets, rng, tools=tools, device=device)
        info = update(policy, critic, episodes, c, prox=prox, anchor=anchor, device=device, lr=args.lr)
    return info


def arrays(policy):
    return [p.detach().cpu().numpy().astype(np.float32) for p in policy.state_dict().values()]


def load_arrays(policy, values):
    policy.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in zip(policy.state_dict(), values)})


def int8_round_trip(new, old):
    out, size = [], 0
    for a, b in zip(new, old):
        delta = a - b
        scale = float(np.abs(delta).max()) / 127 or 1.
        q = np.clip(np.round(delta / scale), -127, 127).astype(np.int8)
        out.append(b + q.astype(np.float32) * scale)
        size += q.nbytes + 4
    return out, size


def run_fed(c, train, test, args, out):
    from flwr.client import ClientApp, NumPyClient
    from flwr.common import ndarrays_to_parameters
    from flwr.server import ServerApp, ServerAppComponents, ServerConfig
    from flwr.server.strategy import FedAvg, FedProx
    from flwr.simulation import run_simulation
    parts = drones(c, train)
    history, uplink = [], []

    class Drone(NumPyClient):
        def __init__(self, d):
            self.d = d

        def fit(self, parameters, config):
            local = cache(args)
            policy = HeadPolicy(local.base).to(args.device)
            load_arrays(policy, parameters)
            critic = Critic().to(args.device)
            path = out / f"critic-drone{self.d}.pt"
            if path.exists():
                critic.load_state_dict(torch.load(path))
            rng = np.random.default_rng(args.seed + 1000 * self.d + int(config.get("round", 0)))
            info = train_cycles(policy, critic, local, parts[self.d], args, args.local_cycles, rng=rng,
                                prox=float(config.get("proximal_mu", 0.)))
            torch.save(critic.state_dict(), path)
            values, size = int8_round_trip(arrays(policy), parameters)
            return values, len(parts[self.d]), {"drone": self.d, "int8_bytes": size, **{k: float(v) for k, v in info.items()}}

    def evaluate_fn(server_round, values, config):
        if server_round == 0:
            return None
        policy = HeadPolicy(c.base).to(args.device)
        load_arrays(policy, values)
        report = evaluate(policy, c, test, args.budgets, device=args.device)
        history.append({"round": server_round, "test": report})
        print(json.dumps({"fed_round": server_round, "test": report}), flush=True)
        return 0., {}

    def fit_metrics(metrics):
        uplink.append(sum(m["int8_bytes"] for _, m in metrics))
        return {}

    class Clients:
        def __call__(self, context):
            return Drone(int(context.node_config["partition-id"])).to_client()

    strategy_type = FedProx if args.strategy == "fedprox" else FedAvg
    extra = {"proximal_mu": args.proximal_mu} if args.strategy == "fedprox" else {}
    strategy = strategy_type(fraction_fit=1., fraction_evaluate=0., min_fit_clients=5, min_available_clients=5,
                             initial_parameters=ndarrays_to_parameters(arrays(HeadPolicy(c.base))), evaluate_fn=evaluate_fn,
                             on_fit_config_fn=lambda r: {"round": r}, fit_metrics_aggregation_fn=fit_metrics, **extra)

    class Server:
        def __call__(self, context):
            return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=args.rounds))

    run_simulation(server_app=ServerApp(server_fn=Server()), client_app=ClientApp(client_fn=Clients()), num_supernodes=5,
                   backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 1 if args.device == "cuda" else 0}})
    return {"history": history, "drone_images": [len(p) for p in parts], "uplink_int8_bytes_per_round": uplink,
            "uplink_int8_total_mb": round(sum(uplink) / 1e6, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", default="zeroshot,report_only,central,fed")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[1., 3., 6.])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--central-cycles", type=int, default=5, help="rollout/update cycles per round, central modes")
    parser.add_argument("--local-cycles", type=int, default=1, help="rollout/update cycles per drone per Flower round")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default="fedavg")
    parser.add_argument("--proximal-mu", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--source", default=str(runs_dir("agent_cache")))
    parser.add_argument("--cache", default=str(runs_dir("agent_cache_h")))
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    c = cache(args)
    train = [i for i, m in enumerate(c.meta) if m["split"] == "val"]
    test = [i for i, m in enumerate(c.meta) if m["split"] == "test"]
    out = results_dir(f"agent_ppo_head/seed{args.seed}")
    summary = {"train_images": len(train), "test_images": len(test), "budgets": args.budgets, "rounds": args.rounds,
               "trainable_parameters": sum(p.numel() for p in HeadPolicy(c.base).parameters())}
    for mode in args.modes.split(","):
        start = time.time()
        if mode == "zeroshot":
            summary[mode] = evaluate(HeadPolicy(c.base).to(args.device), c, test, args.budgets, tools=False, device=args.device)
        elif mode in ("report_only", "central"):
            policy, critic, rng = HeadPolicy(c.base).to(args.device), Critic().to(args.device), np.random.default_rng(args.seed)
            curve = []
            for r in range(args.rounds):
                train_cycles(policy, critic, c, train, args, args.central_cycles, tools=mode == "central", rng=rng)
                curve.append({"round": r + 1, "test": evaluate(policy, c, test, args.budgets, tools=mode == "central", device=args.device)})
                print(json.dumps({mode: r + 1, "test": curve[-1]["test"]}), flush=True)
            summary[mode] = curve
        elif mode == "fed":
            summary[mode] = run_fed(c, train, test, args, out)
        summary[f"{mode}_seconds"] = round(time.time() - start, 1)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"done, wrote {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
