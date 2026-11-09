import json

from fedorch.common import results_dir, write_json


def read(path):
    return json.loads(path.read_text())


def laya(model):
    run = read(results_dir("vision") / f"q4_laya_{model}" / "run.json")
    s = read(results_dir("vision") / f"q4_laya_{model}" / "summary.json")
    t, ci = s["test"], s["test_bootstrap_95ci"]
    return {"family": "laya-vision", "model": model, "stored_mb": run["stored_mb"], "weight_mb": run["weight_mb"],
            "gpu_mb_after_load": run["gpu_mb_after_load"], "peak_gpu_mb": run["peak_gpu_mb"],
            "item_accuracy": s["test_items"]["accuracy"], "item_risk_per_1000": s["test_items"]["risk_per_1000"],
            "condition_macro_f1": t["condition"]["macro_f1"], "condition_macro_f1_95ci": ci["condition_macro_f1"],
            "component_accuracy": t["component"]["accuracy"], "defect_auroc": t["defect_noul"]["auroc"],
            "defect_auroc_95ci": ci["defect_auroc"], "latency_median_ms": run["all_questions_ms"]["median"],
            "latency_p95_ms": run["all_questions_ms"]["p95"], "latency_unit": "image, 4 questions", "non_finite": run["non_finite"],
            "gpu": run["gpu"]}


def smolvlm(model):
    s = read(results_dir("quant") / f"q4_smolvlm_{model}" / "summary.json")
    return {"family": "SmolVLM-256M", "model": model, "stored_mb": s["stored_mb"], "weight_mb": s["weight_mb"],
            "gpu_mb_after_load": s["gpu_mb_after_load"], "peak_gpu_mb": s["peak_gpu_mb"],
            "item_accuracy": s["test_items"]["accuracy"], "item_accuracy_95ci": s["test_items"]["accuracy_95ci"],
            "item_risk_per_1000": s["test_items"]["risk_per_1000"], "latency_median_ms": s["latency_ms"]["median"],
            "latency_p95_ms": s["latency_ms"]["p95"], "latency_unit": "item, 1 question", "non_finite": s["non_finite"], "gpu": s["gpu"]}


def main():
    rows = []
    for fn, models in ((laya, ("base", "good300_full")), (smolvlm, ("base", "lora"))):
        for model in models:
            try:
                rows.append(fn(model))
            except FileNotFoundError as error:
                print(f"skip {fn.__name__} {model}: {error.filename}")
    write_json(results_dir("quant") / "q4_table.json", rows)
    keys = ["family", "model", "stored_mb", "weight_mb", "gpu_mb_after_load", "peak_gpu_mb", "item_accuracy", "item_risk_per_1000",
            "condition_macro_f1", "component_accuracy", "defect_auroc", "latency_median_ms", "latency_p95_ms", "non_finite"]
    table = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    table += ["| " + " | ".join(str(row.get(k, "-")) for k in keys) + " |" for row in rows]
    (results_dir("quant") / "q4_table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
