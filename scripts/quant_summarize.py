import json

from fedorch.common import results_dir, write_json


def laya(path):
    s = json.loads((path / "summary.json").read_text())
    run = json.loads((path / "run.json").read_text())
    t = s["test"]
    return {"family": "laya-vision", "model": run["model"], "format": run["format"], "stored_mb": run["stored_mb"],
            "weight_mb": run["weight_mb"], "peak_gpu_mb": run["peak_gpu_mb"], "item_accuracy": s["test_items"]["accuracy"],
            "item_risk_per_1000": s["test_items"]["risk_per_1000"], "condition_macro_f1": t["condition"]["macro_f1"],
            "component_accuracy": t["component"]["accuracy"], "defect_auroc": t["defect_noul"]["auroc"],
            "median_ms": run["all_questions_ms"]["median"], "p95_ms": run["all_questions_ms"]["p95"], "per": "image, 4 queries"}


def smolvlm(path):
    s = json.loads((path / "summary.json").read_text())
    return {"family": "SmolVLM-256M", "model": s["model"], "format": s["format"], "stored_mb": s["stored_mb"],
            "weight_mb": s["weight_mb"], "peak_gpu_mb": s["peak_gpu_mb"], "item_accuracy": s["test_items"]["accuracy"],
            "item_risk_per_1000": s["test_items"]["risk_per_1000"], "non_finite": s["non_finite"],
            "median_ms": s["latency_ms"]["median"], "p95_ms": s["latency_ms"]["p95"], "per": "item, 1 query"}


def main():
    rows = [laya(p) for p in sorted(results_dir("vision").glob("q_laya_*")) if (p / "summary.json").exists()]
    rows += [smolvlm(p) for p in sorted(results_dir("quant").glob("q_smolvlm_*")) if (p / "summary.json").exists()]
    write_json(results_dir("quant") / "summary.json", rows)
    keys = ["family", "model", "format", "stored_mb", "peak_gpu_mb", "item_accuracy", "item_risk_per_1000",
            "condition_macro_f1", "defect_auroc", "median_ms", "p95_ms"]
    print(" | ".join(keys))
    for row in rows:
        print(" | ".join(str(row.get(k, "-")) for k in keys))


if __name__ == "__main__":
    main()
