import argparse
import json
import random
import time

import torch
from laya.vlm_train import collect_logits, fit_temperatures_from, metrics_from

from fedorch import federated as fl
from fedorch import vision
from fedorch.common import ROOT, WORK, results_dir, write_env, write_json
from fedorch.items import load_items

MODES = ["fedavg", "local", "central"]
POOL = "results/vision/good300_full"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--drones", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--passes", type=float, default=2.5)
    parser.add_argument("--towers", default="all")
    parser.add_argument("--start", default="base")
    parser.add_argument("--freeze", default="full")
    parser.add_argument("--n-last", type=int, default=4)
    parser.add_argument("--prox", type=float, default=0.0)
    parser.add_argument("--holdout", type=float, default=0.06)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--val-images", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latency-frames", type=int, default=100)
    args = parser.parse_args()
    if args.mode not in MODES or args.towers not in ("all", "server", "fleet"):
        parser.error(f"--mode must be one of {MODES}, --towers one of all, server, fleet")
    return args


def pool(rows):
    by_image = {r["image"]: r for r in rows}
    names = [line for f in ("train_images.txt", "calib_images.txt") for line in (ROOT / POOL / f).read_text().split("\n") if line]
    return [by_image[n] for n in names]


def val_score(agent, val_ex):
    m = metrics_from(collect_logits(agent.model, agent.processor, val_ex, batch_size=16), agent.temperature)["all"]
    return {"acc": round(m["acc"], 4), "nll": round(m["nll"], 4)}


def train_rounds(agent, params, group, args, round_index, anchor, drone):
    steps = fl.steps_for(group["train"], args.passes / args.rounds, args.batch_size)
    f = fl.lr_factor(round_index, args.rounds)
    return fl.local_train(agent, params, group["train"], steps, args.batch_size, args.lr_head * f, args.lr_backbone * f,
                          args.prox, anchor, fl.seed_for(args.seed, round_index, drone), args.workers), steps


def finish(agent, params, args, name, groups, held, info, start):
    out = results_dir(f"vision/{name}")
    write_env(f"vision/{name}")
    (out / "train_images.txt").write_text("\n".join(r["image"] for g in groups for r in g["train"]) + "\n")
    (out / "holdout_images.txt").write_text("\n".join(r["image"] for g in groups for r in g["holdout"]) + "\n")
    records = [rec for g in groups for rec in fl.holdout_logits(agent, g["holdout"])]
    agent.temperature = fit_temperatures_from(records) if records else agent.temperature
    model_dir = WORK / "models" / "vision" / name
    agent.save(str(model_dir))
    vision.predict_all(agent, held, out / "predictions.jsonl")
    test = [r for r in held if r["split"] == "test"] or held
    write_json(out / "run.json", dict(vars(args), run=name, model_dir=str(model_dir), checkpoint=vision.CHECKPOINT,
                                      revision=vision.REVISION, questions=vision.QUESTIONS, temperature=agent.temperature,
                                      drones_info=[{"drone": g["drone"], "train_images": len(g["train"]),
                                                    "holdout_images": len(g["holdout"]),
                                                    "defect_share": round(sum(r["defect"] for r in g["train"]) / len(g["train"]), 3)}
                                                   for g in groups],
                                      trainable_parameters=sum(p.numel() for p in params.values()),
                                      update_mb_fp16=fl.megabytes(params), seconds=round(time.time() - start, 1),
                                      fp32_training=True, device=str(agent.device), gpu=vision.gpu_name(), precision="fp32",
                                      peak_gpu_mb=vision.peak_mb(), images=len(held), **info,
                                      **vision.timing(agent, test, args.latency_frames)))
    print(f"{name}: done, wrote {out}", flush=True)


def main():
    args = arguments()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rows = vision.images(load_items())
    data = fl.split_towers(pool(rows), args.towers)
    if args.limit:
        data = data[: args.limit]
    k = 1 if args.mode == "central" else args.drones
    groups = fl.drones(data, k, args.holdout, rng)
    print(f"{args.name}: {args.mode}, {len(groups)} drones, towers {args.towers}, start {args.start}, "
          f"train images {[len(g['train']) for g in groups]}", flush=True)
    held = [r for r in rows if r["split"] in ("val", "test")]
    val_ex = fl.examples(rng.sample([r for r in rows if r["split"] == "val"], args.val_images))
    if args.limit:
        held, val_ex = held[: args.limit], val_ex[: args.limit]
    vision.fp32_on_old_gpus()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = vision.load(device, None if args.start == "base" else WORK / "models" / "vision" / args.start)
    params = fl.trainable(agent.model, args.freeze, args.n_last)
    start = time.time()
    if args.mode == "local":
        init, temperature = fl.snapshot(params), list(agent.temperature)
        for g in groups:
            fl.assign(params, init)
            agent.temperature = list(temperature)
            curve, steps = [], 0
            for r in range(args.rounds):
                losses, n = train_rounds(agent, params, g, args, r, init, g["drone"])
                steps += n
                curve.append({"round": r, "loss": round(sum(losses) / len(losses), 4)})
            finish(agent, params, args, f"{args.name}_d{g['drone']}", [g], held,
                   {"steps": steps, "rounds_log": curve, "val_after": val_score(agent, val_ex)}, start)
        return
    state = fl.snapshot(params)
    curve, steps = [{"round": -1, **val_score(agent, val_ex)}], 0
    print(json.dumps(curve[-1]), flush=True)
    total = sum(len(g["train"]) for g in groups)
    for r in range(args.rounds):
        summed, losses = None, []
        for g in groups:
            fl.assign(params, state)
            loss, n = train_rounds(agent, params, g, args, r, state, g["drone"])
            steps += n
            losses += loss
            w = len(g["train"]) / total
            local = fl.snapshot(params)
            summed = {k2: w * v for k2, v in local.items()} if summed is None else {k2: summed[k2] + w * v for k2, v in local.items()}
        state = summed
        fl.assign(params, state)
        curve.append({"round": r, "loss": round(sum(losses) / len(losses), 4), **val_score(agent, val_ex)})
        print(json.dumps(curve[-1]), flush=True)
    uplink = fl.megabytes(params) * len(groups) * args.rounds if args.mode == "fedavg" else 0.0
    finish(agent, params, args, args.name, groups, held,
           {"steps": steps, "rounds_log": curve, "uplink_mb_total_fp16": round(uplink, 1)}, start)


if __name__ == "__main__":
    main()
