import argparse
import json
import math
import os

os.environ.setdefault("USE_TF", "0")

import laya
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, finetune

from fedorch.cache import load_cache
from fedorch.common import results_dir, runs_dir, write_json
from fedorch.decision import ACTION_TEXT, ACTIONS, QUESTION, config_name, evidence_text, load_evidence, soft_target, subsample, training_items, write_p
from fedorch.items import load_items

QUESTIONS = {"action": {"type": "choice", "instructions": QUESTION, "criteria": dict(ACTION_TEXT)}}


def write_rows(path, items, cache, evidence, temperature):
    with path.open("w") as file:
        for item in items:
            rows = cache[item["item_id"]]
            file.write(json.dumps({"state": evidence_text(item, rows["classifier"], evidence[item["item_id"]]),
                                   "questions": QUESTIONS, "gold": {"action": {"probabilities": soft_target(item, rows, temperature)}}}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on the action choice with risk-based soft targets.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0, help="soft-target temperature")
    parser.add_argument("--oversample", type=int, default=1, help="copies of each informative training frame")
    parser.add_argument("--fraction", type=float, default=1.0, help="share of the train-tower frames (plan A)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-updates", type=int, default=300, help="minimum optimizer updates for small training sets")
    parser.add_argument("--step", default="step4b", help="results folder")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items per split (for a quick test)")
    parser.add_argument("--dry-run", action="store_true", help="check the training rows without training")
    args = parser.parse_args()

    items, cache, evidence = load_items(), load_cache(), load_evidence()
    split = {s: [i for i in items if i["split"] == s][:args.limit] for s in ["train", "val", "test"]}
    out = runs_dir(args.step)
    name = config_name("laya", args.temperature, args.oversample, args.fraction, args.seed)
    train = training_items(subsample(split["train"], args.fraction, args.seed), cache, args.oversample)
    write_rows(out / "laya_train.jsonl", train, cache, evidence, args.temperature)
    write_rows(out / "laya_val.jsonl", split["val"], cache, evidence, args.temperature)
    base = snapshot_download("convaiinnovations/laya")
    epochs = max(args.epochs, math.ceil(args.min_updates * 64 / len(train)))
    config = TrainConfig(epochs=epochs, seed=args.seed, eval_data=str(out / "laya_val.jsonl"))
    if args.dry_run:
        from laya.train import dry_run
        print(dry_run(str(out / "laya_train.jsonl"), base, config))
        return

    summary = finetune(str(out / "laya_train.jsonl"), base, str(out / name), config)
    agent = laya.load(str(out / name))
    p = {}
    for s in ["val", "test"]:
        for item in split[s]:
            answer = agent.predict(evidence_text(item, cache[item["item_id"]]["classifier"], evidence[item["item_id"]]), QUESTIONS)
            p[item["item_id"]] = {a: float(answer["answers"]["action"]["probabilities"][a]) for a in ACTIONS}
    write_p(name, p, args.step)
    write_json(results_dir(args.step) / f"{name}_train.json", {"epochs": epochs, "train_rows": len(train), "summary": summary})


if __name__ == "__main__":
    main()
