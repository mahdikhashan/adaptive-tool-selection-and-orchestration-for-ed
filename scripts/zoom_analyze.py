"""Does ResNet test-time zoom help? Image-level defect detection from results/zoom/resnet_views.jsonl.
Threshold per variant chosen on val (cost: miss 10, false alarm 1); test scored once."""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from fedorch.common import results_dir, write_json

TARGET = "polymer-insulator-lower-shackle"
VARIANTS = {
    "full image": lambda p: p[0],
    "max over 5 crops": lambda p: p[1:].max(),
    "max over full + 5 crops": lambda p: p.max(),
    "mean over full + 5 crops": lambda p: p.mean(),
    "full, raised by best crop if crop > 0.5": lambda p: max(p[0], p[1:].max()) if p[1:].max() > .5 else p[0],
}


def risk(y, d):
    return 1000 * (10 * np.sum(y & ~d) + np.sum(~y & d)) / len(y)


def main():
    rows = [json.loads(l) for l in (results_dir("zoom") / "resnet_views.jsonl").open()]
    report = {}
    for name, f in VARIANTS.items():
        score = {s: np.array([f(1 - np.array(r["probs"])[:, 0]) for r in rows if r["split"] == s]) for s in ("val", "test")}
        y = {s: np.array([r["defect"] for r in rows if r["split"] == s]) for s in ("val", "test")}
        target = {s: np.array([r["asset"] == TARGET for r in rows if r["split"] == s]) for s in ("val", "test")}
        grid = np.unique(np.quantile(score["val"], np.linspace(0, 1, 401)))
        th = min(grid, key=lambda t: risk(y["val"], score["val"] >= t))
        out = {"threshold_from_val": round(float(th), 4)}
        for s in ("val", "test"):
            d = score[s] >= th
            t = target[s]
            out[s] = {"risk_per_1000": round(risk(y[s], d), 1), "auroc": round(float(roc_auc_score(y[s], score[s])), 4),
                      "recall": round(float(d[y[s]].mean()), 4), "false_alarm": round(float(d[~y[s]].mean()), 4),
                      "lower_shackle_recall": round(float(d[t & y[s]].mean()), 4),
                      "lower_shackle_false_alarm": round(float(d[t & ~y[s]].mean()), 4),
                      "lower_shackle_auroc": round(float(roc_auc_score(y[s][t], score[s][t])), 4)}
        report[name] = out
        v, te = out["val"], out["test"]
        print(f"{name:42s} th {th:.3f} | val risk {v['risk_per_1000']:6.1f} AUROC {v['auroc']:.4f} LS rec {v['lower_shackle_recall']:.3f} "
              f"| test risk {te['risk_per_1000']:6.1f} AUROC {te['auroc']:.4f} rec {te['recall']:.3f} FA {te['false_alarm']:.3f} "
              f"LS rec {te['lower_shackle_recall']:.3f} LS FA {te['lower_shackle_false_alarm']:.3f}")
    write_json(results_dir("zoom") / "analysis.json", report)


if __name__ == "__main__":
    main()
