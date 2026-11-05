import argparse
import json
import os

os.environ.setdefault("USE_TF", "0")

import laya
from huggingface_hub import snapshot_download
from laya.train import TrainConfig, finetune

from fedorch.cache import load_cache
from fedorch.common import results_dir, runs_dir, write_json
from fedorch.features import TOOL_TEXT, TOOLS, evidence, write_q
from fedorch.items import load_items

QUESTIONS = {tool: {"type": "noul", "instructions": f"Will '{TOOL_TEXT[tool]}' give the correct answer for this frame?"} for tool in TOOLS}


def write_rows(path, items, cache):
    with path.open("w") as file:
        for item in items:
            rows = cache[item["item_id"]]
            file.write(json.dumps({"state": evidence(item, rows["classifier"]), "questions": QUESTIONS,
                                   "expected": {tool: bool(rows[tool]["correct"]) for tool in TOOLS}}) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Laya to predict whether each tool is correct (one yes/no question per tool).")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N items per split (for a quick test)")
    args = parser.parse_args()

    items, cache = load_items(), load_cache()
    split = {name: [i for i in items if i["split"] == name][:args.limit] for name in ["train", "val", "test"]}
    out = runs_dir("step4")
    write_rows(out / "laya_train.jsonl", split["train"], cache)
    write_rows(out / "laya_val.jsonl", split["val"], cache)

    summary = finetune(str(out / "laya_train.jsonl"), snapshot_download("convaiinnovations/laya"), str(out / "laya_finetuned"),
                       TrainConfig(epochs=args.epochs, eval_data=str(out / "laya_val.jsonl")))
    agent = laya.load(str(out / "laya_finetuned"))
    q = {}
    for name in ["val", "test"]:
        for item in split[name]:
            answers = agent.predict(evidence(item, cache[item["item_id"]]["classifier"]), QUESTIONS)["answers"]
            q[item["item_id"]] = {tool: float(answers[tool]["noul"]) for tool in TOOLS}
    write_q("laya", q)
    write_json(results_dir("step4") / "laya_train.json", {"epochs": args.epochs, "summary": summary})


if __name__ == "__main__":
    main()
