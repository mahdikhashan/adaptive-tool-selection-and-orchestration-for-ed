import argparse
import importlib.util
from pathlib import Path

import torch

from fedorch.cache import load_cache
from fedorch.decision import ACTIONS, load_evidence, numeric_v2
from fedorch.intents import bag_of_words, intent_sets, parsed, soft_target, training_pairs, write_p
from fedorch.items import load_items

spec = importlib.util.spec_from_file_location("step4b_mlp", Path(__file__).with_name("step4b_train_mlp.py"))
step4b_mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4b_mlp)

INTENT_FEATURES = {"blind": lambda intent: [], "bow": bag_of_words, "parsed": parsed}


def main():
    parser = argparse.ArgumentParser(description="Plan F: train the blind, bag-of-words and parsed-intent MLPs.")
    parser.add_argument("--kind", default="label", help="intent focus: label or component")
    args = parser.parse_args()
    items, cache, evidence = load_items(), load_cache(), load_evidence()
    assets = sorted({i["asset"] for i in items})
    split = {s: [i for i in items if i["split"] == s] for s in ["train", "val", "test"]}
    intents = intent_sets(args.kind)
    frame = {i["item_id"]: numeric_v2(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]], assets) for i in items}
    train = training_pairs(split["train"], cache, intents["train"])
    val_fit = training_pairs(split["val"], cache, intents["val"], per_item=1, oversample=1, seed=1)
    outputs = [(i, intent) for i in split["val"] for intent in intents["val"]] + [(i, intent) for i in split["test"] for intent in intents["test"]]

    for name, encode in INTENT_FEATURES.items():
        torch.manual_seed(0)

        def features(pairs):
            return torch.tensor([frame[i["item_id"]] + encode(intent) for i, intent in pairs])

        def targets(pairs):
            return torch.tensor([[soft_target(i, cache[i["item_id"]], intent)[a] for a in ACTIONS] for i, intent in pairs])

        x_train = features(train)
        mean, std = x_train.mean(0), x_train.std(0).clamp(min=1e-6)
        model = step4b_mlp.mlp(x_train.shape[1], len(ACTIONS))
        step4b_mlp.fit(model, step4b_mlp.soft_cross_entropy, (x_train - mean) / std, targets(train),
                       (features(val_fit) - mean) / std, targets(val_fit))
        with torch.no_grad():
            probs = torch.softmax(model((features(outputs) - mean) / std), -1).tolist()
        write_p(f"{args.kind}_mlp_{name}", [(i["item_id"], intent["id"], dict(zip(ACTIONS, p))) for (i, intent), p in zip(outputs, probs)])
        print(name, x_train.shape)


if __name__ == "__main__":
    main()
