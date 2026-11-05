import argparse
import json
import math
import random
import time

import torch
from laya.vlm_train import collect_logits, fit_temperatures_from, format_metrics, load_jsonl_examples, metrics_from, train

from fedorch import vision
from fedorch.common import WORK, results_dir, write_env, write_json
from fedorch.items import load_items


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="lastn_pc400")
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--calib-images", type=int, default=100)
    parser.add_argument("--val-images", type=int, default=600)
    parser.add_argument("--freeze", default="last_n")
    parser.add_argument("--n-last", type=int, default=4)
    parser.add_argument("--passes", type=float, default=2.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=2e-5)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--latency-frames", type=int, default=500)
    return parser.parse_args()


def sample(rows, per_class, rng):
    by_class = {}
    for row in rows:
        by_class.setdefault(row["condition"], []).append(row)
    out = []
    for condition in vision.CONDITION.values():
        group = by_class.get(condition, [])
        out += rng.sample(group, min(per_class, len(group)))
    rng.shuffle(out)
    return out


def write_split(folder, split, rows):
    with (folder / f"{split}.jsonl").open("w") as file:
        for row in rows:
            file.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in vision.labelled(row))


def scores(agent, examples):
    return metrics_from(collect_logits(agent.model, agent.processor, examples, batch_size=16), agent.temperature)


def main():
    args = arguments()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    rows = vision.images(load_items())
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    chosen = sample(split["train"], args.per_class, rng)
    calib, train_rows = chosen[: args.calib_images], chosen[args.calib_images:]
    val_rows = rng.sample(split["val"], min(args.val_images, len(split["val"])))
    if args.limit:
        train_rows, calib, val_rows = train_rows[: args.limit], calib[: args.limit], val_rows[: args.limit]
    out = results_dir(f"vision/{args.name}")
    write_env(f"vision/{args.name}")
    data = WORK / "data" / "laya_insplad"
    folder = data / args.name
    folder.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train_rows), ("calib", calib), ("val", val_rows)):
        write_split(folder, name, part)
        (out / f"{name}_images.txt").write_text("\n".join(r["image"] for r in part) + "\n")
    train_ex = load_jsonl_examples(str(data), args.name, "train")
    calib_ex = load_jsonl_examples(str(data), args.name, "calib")
    val_ex = load_jsonl_examples(str(data), args.name, "val")
    counts = {c: sum(r["condition"] == c for r in train_rows) for c in vision.CONDITION.values()}
    print(f"train {len(train_rows)} images / {len(train_ex)} questions, calib {len(calib_ex)}, val {len(val_ex)}", flush=True)
    print("train images per condition", json.dumps(counts), flush=True)
    fp32 = vision.fp32_on_old_gpus()
    agent = vision.load("cuda" if torch.cuda.is_available() else "cpu")
    before = scores(agent, val_ex)
    print("before:", format_metrics(before), flush=True)
    steps = max(1, math.ceil(args.passes * len(train_ex) / args.batch_size))
    stats, start = {}, time.time()
    losses = train(agent.model, agent.processor, train_ex, steps=steps, batch_size=args.batch_size, freeze=args.freeze,
                   n_last=args.n_last, lr_head=args.lr_head, lr_backbone=args.lr_backbone, warmup=args.warmup,
                   device=str(agent.device), seed=args.seed, log_every=50, stats=stats, num_workers=args.workers)
    train_seconds = time.time() - start
    agent.temperature = fit_temperatures_from(collect_logits(agent.model, agent.processor, calib_ex, batch_size=16))
    after = scores(agent, val_ex)
    print("after:", format_metrics(after), flush=True)
    model_dir = WORK / "models" / "vision" / args.name
    agent.save(str(model_dir))
    agent = vision.load("cuda" if torch.cuda.is_available() else "cpu", model_dir)
    held = split["val"] + split["test"]
    if args.limit:
        held = held[: args.limit]
    vision.predict_all(agent, held, out / "predictions.jsonl")
    test = [r for r in held if r["split"] == "test"] or held
    write_json(out / "run.json", dict(vars(args), checkpoint=vision.CHECKPOINT, revision=vision.REVISION, model_dir=str(model_dir),
                                      questions=vision.QUESTIONS, train_images=len(train_rows), train_questions=len(train_ex),
                                      train_images_per_condition=counts, calib_questions=len(calib_ex), val_questions=len(val_ex),
                                      steps=steps, train_seconds=round(train_seconds, 1), train_stats=stats,
                                      loss_every_50=[round(sum(losses[i:i + 50]) / len(losses[i:i + 50]), 4) for i in range(0, len(losses), 50)],
                                      temperature=agent.temperature, val_before=before, val_after=after,
                                      fp32_training=fp32, device=str(agent.device), gpu=vision.gpu_name(), precision="fp32",
                                      peak_gpu_mb=vision.peak_mb(), images=len(held),
                                      **vision.timing(agent, test, args.latency_frames)))
    print(f"done, wrote {out}", flush=True)


if __name__ == "__main__":
    main()
