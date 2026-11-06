import json
import statistics
import sys

from fedorch.common import results_dir, write_json

KEYS = [("condition", "accuracy"), ("condition", "balanced_accuracy"), ("condition", "macro_f1"),
        ("component", "accuracy"), ("defect_noul", "auroc"), ("defect_noul", "average_precision")]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "fl_local_k6"
    runs = sorted(p.parent for p in results_dir("vision").glob(f"{name}_d*/summary.json"))
    per = {r.name: json.loads((r / "summary.json").read_text())["test"] for r in runs}
    out = {"runs": list(per), "test_per_drone": {r: {f"{p}.{k}": t[p][k] for p, k in KEYS} for r, t in per.items()}}
    out["test_mean_min_max"] = {f"{p}.{k}": [round(statistics.mean(t[p][k] for t in per.values()), 4),
                                             min(t[p][k] for t in per.values()), max(t[p][k] for t in per.values())] for p, k in KEYS}
    write_json(results_dir(f"vision/{name}") / "summary.json", out)
    print(json.dumps(out["test_mean_min_max"], indent=1))


if __name__ == "__main__":
    main()
