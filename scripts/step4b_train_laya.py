import argparse
import json
import os

os.environ.setdefault("USE_TF", "0")

import laya
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, finetune

from fedorch.cache import load_cache
from fedorch.common import results_dir, runs_dir, write_json
from fedorch.decision import ACTION_TEXT, ACTIONS, QUESTION, evidence_text, load_evidence, soft_target, write_p
from fedorch.items import load_items

QUESTIONS = {"action": {"type": "choice", "instructions": QUESTION, "criteria": dict(ACTION_TEXT)}}


def write_rows(path, items, cache, evidence):
    with path.open("w") as file:
        for item in items:
            rows = cache[item["item_id"]]
            file.write(json.dumps({"state": evidence_text(item, rows["classifier"], evidence[item["item_id"]]),
                                   "questions": QUESTIONS, "gold": {"action": {"probabilities": soft_target(item, rows)}}}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya on the action choice with risk-based soft targets.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items per split (for a quick test)")
    parser.add_argument("--dry-run", action="store_true", help="check the training rows without training")
    args = parser.parse_args()

    items, cache, evidence = load_items(), load_cache(), load_evidence()
    split = {s: [i for i in items if i["split"] == s][:args.limit] for s in ["train", "val", "test"]}
    out = runs_dir("step4b")
    write_rows(out / "laya_train.jsonl", split["train"], cache, evidence)
    write_rows(out / "laya_val.jsonl", split["val"], cache, evidence)
    base = snapshot_download("convaiinnovations/laya")
    config = TrainConfig(epochs=args.epochs, eval_data=str(out / "laya_val.jsonl"))
    if args.dry_run:
        from laya.train import dry_run
        print(dry_run(str(out / "laya_train.jsonl"), base, config))
        return

    summary = finetune(str(out / "laya_train.jsonl"), base, str(out / "laya_choice"), config)
    agent = laya.load(str(out / "laya_choice"))
    p = {}
    for s in ["val", "test"]:
        for item in split[s]:
            answer = agent.predict(evidence_text(item, cache[item["item_id"]]["classifier"], evidence[item["item_id"]]), QUESTIONS)
            p[item["item_id"]] = {a: float(answer["answers"]["action"]["probabilities"][a]) for a in ACTIONS}
    write_p("laya", p)
    write_json(results_dir("step4b") / "laya_train.json", {"epochs": args.epochs, "summary": summary})


if __name__ == "__main__":
    main()
