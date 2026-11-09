import json
import statistics

import numpy as np

from fedorch import vision
from fedorch.common import ROOT, results_dir, write_json
from fedorch.items import load_items

LAYA_REFERENCE = {"base": "zeroshot", "good300_full": "good300_full"}
SMOLVLM_REFERENCE = {"base": ("tool_smolvlm_base.jsonl", "shared"), "lora": ("tool_smolvlm_local.jsonl", "full")}


def lines(path):
    return [json.loads(line) for line in path.open()]


def auroc(scores, labels):
    ranks = np.argsort(np.argsort(scores)) + 1
    pos = np.array(labels, dtype=bool)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def top(probs):
    return max(probs, key=probs.get)


def laya_scores(rows, truth):
    return {"condition_accuracy": round(float(np.mean([top(a["condition"]) == truth[i]["condition"] for i, a in rows])), 3),
            "component_accuracy": round(float(np.mean([top(a["component"]) == truth[i]["component"] for i, a in rows])), 3),
            "defect_auroc": round(auroc([a["defect"] for _, a in rows], [truth[i]["defect"] for i, _ in rows]), 3)}


def laya(path, truth):
    run = json.loads((path / "run.json").read_text())
    ref_dir = results_dir("vision") / LAYA_REFERENCE[run["model"]]
    quant = {p["image"]: p for p in lines(path / "predictions.jsonl")}
    ref = {p["image"]: p for p in lines(ref_dir / "predictions.jsonl") if p["image"] in quant}
    images = sorted(i for i in ref if i in truth)
    q = [(i, quant[i]["answers"]) for i in images]
    f = [(i, ref[i]["answers"]) for i in images]
    ref_run = json.loads((ref_dir / "run.json").read_text())
    return {"family": "laya-vision", "model": run["model"], "format": run["format"], "n": len(images),
            "stored_mb": run["stored_mb"], "weight_mb": run["weight_mb"], "peak_gpu_mb": run["peak_gpu_mb"],
            "quantized": laya_scores(q, truth), "fp32": laya_scores(f, truth),
            "agreement_condition": round(float(np.mean([top(a["condition"]) == top(b["condition"]) for (_, a), (_, b) in zip(q, f)])), 3),
            "mean_abs_diff_p_defect": round(float(np.mean([abs(a["defect"] - b["defect"]) for (_, a), (_, b) in zip(q, f)])), 4),
            "median_ms": round(statistics.median(quant[i]["ms"] for i in images), 1), "gpu": run["gpu"],
            "fp32_median_ms": ref_run["all_questions_ms"]["median"], "fp32_gpu": ref_run["gpu"]}


def smolvlm(path):
    s = json.loads((path / "summary.json").read_text())
    name, model = SMOLVLM_REFERENCE[s["model"]]
    quant = {p["item_id"]: p for p in lines(path / "predictions.jsonl")}
    ref = {p["item_id"]: p for p in lines(ROOT / "results" / "step2" / name) if p["item_id"] in quant and p["tool_model"] == model}
    items = sorted(ref)
    env = json.loads((ROOT / "results" / "step2" / "env.json").read_text())
    return {"family": "SmolVLM-256M", "model": s["model"], "format": s["format"], "n": len(items),
            "stored_mb": s["stored_mb"], "weight_mb": s["weight_mb"], "peak_gpu_mb": s["peak_gpu_mb"],
            "quantized": {"item_accuracy": round(float(np.mean([quant[i]["prediction"] == quant[i]["label"] for i in items])), 3)},
            "fp32": {"item_accuracy": round(float(np.mean([ref[i]["correct"] for i in items])), 3)},
            "agreement": round(float(np.mean([quant[i]["prediction"] == ref[i]["prediction"] for i in items])), 3),
            "non_finite": s["non_finite"], "median_ms": round(statistics.median(quant[i]["ms"] for i in items), 1), "gpu": s["gpu"],
            "fp32_median_ms": round(statistics.median(ref[i]["compute_ms"] for i in items), 1), "fp32_gpu": env["gpu"]}


def main():
    truth = {x["image"]: x for x in vision.images(load_items(("test",)))}
    rows = [laya(p, truth) for p in sorted(results_dir("vision").glob("q_laya_*")) if (p / "run.json").exists()]
    rows += [smolvlm(p) for p in sorted(results_dir("quant").glob("q_smolvlm_*")) if (p / "summary.json").exists()]
    write_json(results_dir("quant") / "compare.json", rows)
    for row in rows:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
