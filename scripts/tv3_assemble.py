import json
import math
import os
import statistics
from collections import defaultdict

import numpy as np

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.items import CLASSES, load_items
from fedorch.select_env import Teacher, report_loss, truth_of
from fedorch.tools_v3 import TOOLS, class_probs, group_of, load_tool, recut_file, timing_path

STEP2 = {"smolvlm": "smolvlm_local", "smolvlm_base": "smolvlm_base", "smolvlm_split": "smolvlm_split", "qwen": "cloud"}
LAYA = {"laya": "good300_full", "laya_q4": "q4_laya_good300_full"}
CLASS_OF = {v: k for k, v in vision.CONDITION.items()}
SCORE_TOOLS = ("quality", "exposure", "size")


def from_step2(row):
    good, cannot = row["probs"].get("good", 0.), row["probs"].get("cannot-judge", 0.)
    return max(0., 1 - good - cannot) / max(1e-6, 1 - cannot)


def from_laya(answers):
    return {"condition": class_probs({CLASS_OF[k]: v for k, v in answers["condition"].items()}),
            "component": dict(answers["component"]), "defect": float(answers["defect"])}


def clean(out):
    if "condition" in out and not all(math.isfinite(v) for v in out["condition"].values()):
        return None
    if "defect" in out and not math.isfinite(out["defect"]):
        return None
    return out


def main():
    items = [i for i in load_items() if i["split"] in ("val", "test")]
    by_image = defaultdict(list)
    for i in items:
        by_image[i["image"]].append(i)
    rows = [r for r in vision.images(items)]
    step2 = load_cache(("classifier", *STEP2.values()))
    laya = {name: {json.loads(l)["image"]: json.loads(l) for l in open(f"results/vision/{run}/predictions.jsonl")} for name, run in LAYA.items()}
    zoom = {json.loads(l)["image"]: json.loads(l) for l in open("results/zoom/resnet_views.jsonl")}
    timing = json.loads(timing_path().read_text()) if timing_path().exists() else {}
    special = set(STEP2) | set(LAYA) | {"resnet", "resnet_crops_mean", "resnet_crops_max"}
    caches = {t: load_tool(t) for t in TOOLS if t not in special}
    caches = {t: c for t, c in caches.items() if c and t in timing}
    scores = {t: [r["score"] for r in caches[t].values()] for t in SCORE_TOOLS if t in caches}
    norm = {t: (statistics.mean(v), statistics.pstdev(v) or 1.) for t, v in scores.items()}
    jpeg = statistics.median(r["cloud"]["payload_bytes"] for r in step2.values() if "cloud" in r)
    resnet_ms = statistics.median(r["classifier"]["compute_ms"] for r in step2.values() if "classifier" in r)
    image_bytes = {}
    for i in items:
        row = step2.get(i["item_id"], {}).get("cloud")
        if row and row.get("payload_bytes"):
            image_bytes[i["image"]] = row["payload_bytes"]
    print(f"images {len(rows)}, cached new tools: {sorted(caches)}", flush=True)
    table = []
    for r in rows:
        out, cost = {}, {}
        size = image_bytes.get(r["image"], jpeg)
        recut = recut_file(r)
        recut_size = os.path.getsize(recut) if recut.exists() else size
        views = zoom.get(r["image"])
        if views:
            probs = [class_probs(dict(zip(CLASSES, v))) for v in views["probs"]]
            out["resnet"] = {"condition": probs[0]}
            out["resnet_crops_mean"] = {"condition": {c: float(np.mean([p[c] for p in probs])) for c in CLASSES}}
            out["resnet_crops_max"] = {"defect": float(max(1 - p["good"] for p in probs))}
            cost["resnet"] = {"gpu_ms": resnet_ms, "bytes": size}
            cost["resnet_crops_mean"] = cost["resnet_crops_max"] = {"gpu_ms": 6 * resnet_ms, "bytes": size}
        its = sorted(by_image[r["image"]], key=lambda i: i["task"])
        for tool, key in STEP2.items():
            row = next((step2[i["item_id"]][key] for i in its if key in step2.get(i["item_id"], {})), None)
            if row is None:
                continue
            out[tool] = {"defect": from_step2(row)}
            cost[tool] = {"gpu_ms": row["compute_ms"], "bytes": size}
        for tool in LAYA:
            row = laya[tool].get(r["image"])
            if row:
                out[tool] = from_laya(row["answers"])
                cost[tool] = {"gpu_ms": row["ms"], "bytes": size}
        for tool, cache in caches.items():
            row = cache.get(r["image"])
            if not row:
                continue
            if "probs" in row:
                o = {"condition": class_probs(row["probs"])}
            elif "answers" in row:
                o = from_laya(row["answers"])
            elif "defect" in row:
                kind, value = next(iter(row["defect"].items()))
                o = {"defect": float(value)} if kind == "any" else {"kind": {kind: float(value)}}
            else:
                mean, std = norm[tool]
                o = {"score": round((row["score"] - mean) / std, 3)}
            o = clean(o)
            if o is None:
                continue
            out[tool] = o
            cost[tool] = {"gpu_ms": timing[tool]["gpu_ms"], "bytes": recut_size if "recut" in tool else size}
        if not out:
            continue
        table.append({"image": r["image"], "split": r["split"], "tower": r["tower"], "asset": r["asset"], "group": group_of(r["asset"]),
                      "component": r["component"], "condition": r["condition"], "condition_label": CLASS_OF[r["condition"]],
                      "defect": bool(r["defect"]), "out": out, "cost": cost})
    root = results_dir("tools_v3")
    with (root / "images.jsonl").open("w") as f:
        f.writelines(json.dumps(t) + "\n" for t in table)
    present = [t for t in TOOLS if any(t in row["out"] for row in table)]
    tools = {t: {"location": TOOLS[t][0], "description": TOOLS[t][1], "coverage": round(sum(t in row["out"] for row in table) / len(table), 4),
                 "median_gpu_ms": round(statistics.median(row["cost"][t]["gpu_ms"] for row in table if t in row["out"]), 2)}
             for t in present}
    write_json(root / "tools.json", tools)
    print(f"images {len(table)} (val {sum(t['split'] == 'val' for t in table)}, test {sum(t['split'] == 'test' for t in table)}), tools {len(tools)}", flush=True)
    for t, v in tools.items():
        print(f"  {t:22s} {v['location']:6s} coverage {v['coverage']:.2f} GPU {v['median_gpu_ms']:.1f} ms", flush=True)
    value_report(table, present)


def metrics(rows, reports):
    truths = [truth_of(r) for r in rows]
    said = np.array([rep["defect"] == "yes" for rep in reports])
    true = np.array([t["defect"] for t in truths])
    conditions = sorted({t["condition"] for t in truths})
    f1 = []
    for c in conditions:
        tp = sum(rep["condition"] == c and t["condition"] == c for rep, t in zip(reports, truths))
        fp = sum(rep["condition"] == c and t["condition"] != c for rep, t in zip(reports, truths))
        fn = sum(rep["condition"] != c and t["condition"] == c for rep, t in zip(reports, truths))
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"images": len(rows), "loss_per_1000": round(1000 * float(np.mean([report_loss(rep, t) for rep, t in zip(reports, truths)])), 1),
            "component_accuracy": round(float(np.mean([rep["component"] == t["component"] for rep, t in zip(reports, truths)])), 4),
            "condition_accuracy": round(float(np.mean([rep["condition"] == t["condition"] for rep, t in zip(reports, truths)])), 4),
            "condition_macro_f1": round(float(np.mean(f1)), 4),
            "defect_recall": round(float(said[true].mean()), 4) if true.any() else None,
            "defect_false_alarm": round(float(said[~true].mean()), 4),
            "defect_risk_per_1000": round(1000 * float(np.mean(10 * (true & ~said) + (~true & said))), 1),
            "all_three_correct": round(float(np.mean([rep["component"] == t["component"] and rep["condition"] == t["condition"]
                                                      and (rep["defect"] == "yes") == t["defect"] for rep, t in zip(reports, truths)])), 4)}


def value_report(table, tools):
    val = [r for r in table if r["split"] == "val"]
    test = [r for r in table if r["split"] == "test"]
    teacher = Teacher(tools).fit(val)
    report = {"tools": tools, "teacher": {}, "single_tool": {}}
    sets = {"no tool": set(), "resnet only": {"resnet"}, "laya only": {"laya"}, "resnet + laya": {"resnet", "laya"}, "all tools": set(tools)}
    for name, called in sets.items():
        report["teacher"][name] = metrics(test, [teacher.report(r, called) for r in test])
        print(f"teacher, {name}:", json.dumps(report["teacher"][name]), flush=True)
    for t in tools:
        rows = [r for r in test if t in r["out"]]
        if len(rows) >= 50:
            report["single_tool"][t] = metrics(rows, [teacher.report(r, {t}) for r in rows])
    write_json(results_dir("tools_v3") / "value_report.json", report)


if __name__ == "__main__":
    main()
