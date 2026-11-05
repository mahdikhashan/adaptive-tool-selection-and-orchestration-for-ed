import torch

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.features import TOOLS, numeric, write_q
from fedorch.items import load_items


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    positives = sum(labels)
    negatives = len(labels) - positives
    rank_sum = sum(rank for rank, (_, label) in enumerate(pairs, 1) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def tensors(items, cache, assets):
    x = torch.tensor([numeric(i, cache[i["item_id"]]["classifier"], assets) for i in items])
    y = torch.tensor([[float(cache[i["item_id"]][t]["correct"]) for t in TOOLS] for i in items])
    return x, y


def fit(model, x, y, x_val, y_val, epochs=2000, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    best, best_state, waited = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss_fn(model(x), y).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val).item()
        if val_loss < best:
            best, best_state, waited = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if waited > patience:
                break
    model.load_state_dict(best_state)
    return best


def main():
    torch.manual_seed(0)
    items, cache = load_items(), load_cache()
    assets = sorted({i["asset"] for i in items})
    split = {name: [i for i in items if i["split"] == name] for name in ["train", "val", "test"]}
    x, y = tensors(split["train"], cache, assets)
    mean, std = x.mean(0), x.std(0).clamp(min=1e-6)
    data = {name: tensors(split[name], cache, assets) for name in ["val", "test"]}
    data = {name: ((xs - mean) / std, ys) for name, (xs, ys) in data.items()}
    x = (x - mean) / std

    models = {
        "logistic": torch.nn.Linear(x.shape[1], len(TOOLS)),
        "mlp": torch.nn.Sequential(torch.nn.Linear(x.shape[1], 64), torch.nn.ReLU(),
                                   torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, len(TOOLS))),
    }
    report = {}
    for name, model in models.items():
        val_loss = fit(model, x, y, *data["val"])
        with torch.no_grad():
            q = {s: torch.sigmoid(model(data[s][0])) for s in ["val", "test"]}
        write_q(name, {i["item_id"]: dict(zip(TOOLS, map(float, q[s][n]))) for s in ["val", "test"] for n, i in enumerate(split[s])})
        report[name] = {"val_loss": val_loss, "auroc": {s: {t: round(auroc(q[s][:, k].tolist(), data[s][1][:, k].tolist()), 4)
                                                          for k, t in enumerate(TOOLS)} for s in ["val", "test"]}}
        print(name, report[name]["auroc"]["test"])
    write_json(results_dir("step4") / "numeric_train.json", report)


if __name__ == "__main__":
    main()
