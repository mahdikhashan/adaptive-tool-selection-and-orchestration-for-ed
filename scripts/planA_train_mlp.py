import importlib.util
from pathlib import Path

import torch

from fedorch.cache import load_cache
from fedorch.decision import ACTIONS, config_name, load_evidence, numeric_v2, soft_target, subsample, training_items, write_p
from fedorch.items import load_items

FRACTIONS = [0.01, 0.05, 0.2, 1.0]
SEEDS = [0, 1, 2]
TEMPERATURE, OVERSAMPLE = 1.0, 5

spec = importlib.util.spec_from_file_location("step4b_mlp", Path(__file__).with_name("step4b_train_mlp.py"))
step4b_mlp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step4b_mlp)


def main():
    items, cache, evidence = load_items(), load_cache(), load_evidence()
    assets = sorted({i["asset"] for i in items})
    split = {s: [i for i in items if i["split"] == s] for s in ["train", "val", "test"]}

    def features(rows):
        return torch.tensor([numeric_v2(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]], assets) for i in rows])

    def targets(rows):
        return torch.tensor([[soft_target(i, cache[i["item_id"]], TEMPERATURE)[a] for a in ACTIONS] for i in rows])

    raw = {s: features(split[s]) for s in ["val", "test"]}
    val_target = targets(split["val"])
    for fraction in FRACTIONS:
        for seed in SEEDS:
            torch.manual_seed(seed)
            train = training_items(subsample(split["train"], fraction, seed), cache, OVERSAMPLE)
            x_train = features(train)
            mean, std = x_train.mean(0), x_train.std(0).clamp(min=1e-6)
            x = {s: (v - mean) / std for s, v in raw.items()}
            model = step4b_mlp.mlp(x_train.shape[1], len(ACTIONS))
            step4b_mlp.fit(model, step4b_mlp.soft_cross_entropy, (x_train - mean) / std, targets(train), x["val"], val_target)
            name = config_name("mlp", TEMPERATURE, OVERSAMPLE, fraction, seed)
            with torch.no_grad():
                write_p(name, {i["item_id"]: dict(zip(ACTIONS, map(float, torch.softmax(model(x[s]), -1)[n])))
                               for s in ["val", "test"] for n, i in enumerate(split[s])}, "planA")
            print(name, len(train))


if __name__ == "__main__":
    main()
