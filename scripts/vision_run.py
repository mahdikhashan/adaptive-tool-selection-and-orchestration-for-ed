import argparse
import json
import time

import torch

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.common import results_dir, write_env, write_json
from fedorch.items import load_items

MODES = ["zeroshot", "central", "local", "fedavg"]


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--drones", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze", default="last_n")
    parser.add_argument("--n-last", type=int, default=4)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.mode not in MODES:
        parser.error(f"--mode must be one of {MODES}")
    return args


def name_of(args):
    base = args.mode if args.mode in ("zeroshot", "central") else f"{args.mode}_k{args.drones}"
    return f"{base}_s{args.seed}"


def subset(items, limit):
    return items[:limit] if limit else items


def train_rows(items, cache):
    return vision.examples(items, cache)


def run_central(agent, params, train, cache, args):
    losses = vision.train(agent, params, train_rows(train, cache), args.steps, args.batch_size, args.lr_head,
                          args.lr_backbone, args.seed, args.workers)
    return {"loss_first_100": sum(losses[:100]) / max(1, len(losses[:100])),
            "loss_last_100": sum(losses[-100:]) / max(1, len(losses[-100:])), "steps": len(losses)}


def run_fedavg(agent, params, train, cache, args):
    groups = vision.drones(train, args.drones)
    local_steps = max(1, args.steps // (args.rounds * args.drones))
    state, log = vision.snapshot(params), []
    for r in range(args.rounds):
        states, weights, losses = [], [], []
        for d, group in enumerate(groups):
            vision.assign(params, state)
            losses += vision.train(agent, params, train_rows(group, cache), local_steps, args.batch_size,
                                   args.lr_head, args.lr_backbone, args.seed * 100000 + r * 100 + d, args.workers)
            states.append(vision.snapshot(params))
            weights.append(len(group))
        state = vision.fedavg(states, weights)
        vision.assign(params, state)
        log.append({"round": r, "mean_loss": sum(losses) / len(losses)})
        print(json.dumps(log[-1]), flush=True)
    return {"drone_items": [len(g) for g in groups], "local_steps": local_steps, "rounds": log,
            "uplink_mb_per_drone_per_round_fp16": vision.megabytes(params)}


def run_local(agent, params, train, cache, args, rows):
    groups = vision.drones(train, args.drones)
    init, scores, info = vision.snapshot(params), [], []
    for d, group in enumerate(groups):
        vision.assign(params, init)
        losses = vision.train(agent, params, train_rows(group, cache), max(1, args.steps // args.drones),
                              args.batch_size, args.lr_head, args.lr_backbone, args.seed * 100000 + d, args.workers)
        probs = vision.predict(agent, rows, workers=args.workers)
        scores += [dict(row, drone=d) for row in score_rows(rows, probs)]
        info.append({"drone": d, "items": len(group), "loss_last_100": sum(losses[-100:]) / max(1, len(losses[-100:]))})
        print(json.dumps(info[-1]), flush=True)
    return {"drones": info}, scores


def score_rows(rows, probs):
    return [{"item_id": row["id"], "kind": row["kind"], "probs": p} for row, p in zip(rows, probs)]


def main():
    args = arguments()
    torch.manual_seed(args.seed)
    name = name_of(args)
    out = results_dir(f"vision/{name}")
    write_env(f"vision/{name}")
    items = load_items()
    cache = load_cache(vision.ROUTES)
    train = subset([i for i in items if i["split"] == "train"], args.limit)
    held = subset([i for i in items if i["split"] == "val"], args.limit) + \
        subset([i for i in items if i["split"] == "test"], args.limit)
    rows = vision.examples(held, cache)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = vision.load(device)
    params = vision.trainable(agent, args.freeze, args.n_last)
    print(f"{name}: train {len(train)} items, held-out {len(held)} items, "
          f"trainable {sum(p.numel() for p in params.values())} parameters", flush=True)
    start = time.time()
    info = {}
    if args.mode == "zeroshot":
        scores = score_rows(rows, vision.predict(agent, rows, workers=args.workers))
    elif args.mode == "local":
        info, scores = run_local(agent, params, train, cache, args, rows)
    else:
        info = (run_central if args.mode == "central" else run_fedavg)(agent, params, train, cache, args)
        scores = score_rows(rows, vision.predict(agent, rows, workers=args.workers))
    with (out / "scores.jsonl").open("w") as file:
        file.writelines(json.dumps(row) + "\n" for row in scores)
    write_json(out / "train.json", dict(vars(args), name=name, checkpoint=vision.CHECKPOINT,
                                        revision=vision.REVISION, seconds=round(time.time() - start, 1),
                                        trainable_parameters=sum(p.numel() for p in params.values()), **info))
    print(f"{name}: done in {time.time() - start:.0f} s, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
