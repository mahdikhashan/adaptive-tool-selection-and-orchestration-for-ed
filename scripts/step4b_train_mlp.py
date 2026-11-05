import torch

from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import ACTIONS, load_evidence, numeric_v2, soft_target, write_p
from fedorch.features import TOOLS, write_q
from fedorch.items import load_items


def mlp(inputs, outputs):
    return torch.nn.Sequential(torch.nn.Linear(inputs, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(),
                               torch.nn.Linear(64, outputs))


def fit(model, loss_fn, x, y, x_val, y_val, epochs=2000, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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


def soft_cross_entropy(logits, target):
    return -(target * torch.log_softmax(logits, -1)).sum(-1).mean()


def main():
    torch.manual_seed(0)
    items, cache, evidence = load_items(), load_cache(), load_evidence()
    assets = sorted({i["asset"] for i in items})
    split = {s: [i for i in items if i["split"] == s] for s in ["train", "val", "test"]}
    x = {s: torch.tensor([numeric_v2(i, cache[i["item_id"]]["classifier"], evidence[i["item_id"]], assets) for i in split[s]])
         for s in split}
    mean, std = x["train"].mean(0), x["train"].std(0).clamp(min=1e-6)
    x = {s: (v - mean) / std for s, v in x.items()}
    choice = {s: torch.tensor([[soft_target(i, cache[i["item_id"]])[a] for a in ACTIONS] for i in split[s]]) for s in split}
    correct = {s: torch.tensor([[float(cache[i["item_id"]][t]["correct"]) for t in TOOLS] for i in split[s]]) for s in split}

    choice_model = mlp(x["train"].shape[1], len(ACTIONS))
    choice_loss = fit(choice_model, soft_cross_entropy, x["train"], choice["train"], x["val"], choice["val"])
    q_model = mlp(x["train"].shape[1], len(TOOLS))
    q_loss = fit(q_model, torch.nn.BCEWithLogitsLoss(), x["train"], correct["train"], x["val"], correct["val"])

    with torch.no_grad():
        write_p("mlp", {i["item_id"]: dict(zip(ACTIONS, map(float, torch.softmax(choice_model(x[s]), -1)[n])))
                        for s in ["val", "test"] for n, i in enumerate(split[s])})
        write_q("mlp_v2", {i["item_id"]: dict(zip(TOOLS, map(float, torch.sigmoid(q_model(x[s]))[n])))
                           for s in ["val", "test"] for n, i in enumerate(split[s])}, step="step4b")
    write_json(results_dir("step4b") / "mlp_train.json", {"features": x["train"].shape[1], "choice_val_loss": choice_loss, "q_val_loss": q_loss})


if __name__ == "__main__":
    main()
