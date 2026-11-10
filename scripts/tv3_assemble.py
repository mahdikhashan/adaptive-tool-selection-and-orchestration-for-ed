import json
import math
import statistics

import numpy as np

from fedorch import vision
from fedorch.cache import load_cache
from fedorch.common import results_dir, write_json
from fedorch.decision import answer_risk
from fedorch.items import CLASSES, load_items
from fedorch.select_env import BAYES, Fusion, corrected, em_prior, out_of_fold
from fedorch.tools_v3 import EDGE_MBPS, EDGE_RTT, JETSON, TOOLS, class_probs, group_of, load_tool, timing_path

STEP2 = {"resnet": "classifier", "smolvlm": "smolvlm_local", "smolvlm_base": "smolvlm_base", "smolvlm_split": "smolvlm_split", "qwen": "cloud"}
LAYA = {"laya": "good300_full", "laya_q4": "q4_laya_good300_full", "laya_edge": "good300_full"}
RECUT_S = 0.02


def defect_option(item):
    return next(o for o in item["options"].values() if o not in ("good", "cannot-judge"))


def from_classes(item, probs):
    probs = class_probs(probs)
    if item["task"] == "T2":
        return 1 - probs["good"]
    d, g = probs.get(defect_option(item), 0.), probs["good"]
    return d / max(1e-9, d + g)


def from_step2(row):
    good, cannot = row["probs"].get("good", 0.), row["probs"].get("cannot-judge", 0.)
    return max(0., 1 - good - cannot) / max(1e-6, 1 - cannot)


def from_laya(item, answers):
    if item["task"] == "T2":
        return answers["defect"]
    d = answers["condition"][vision.CONDITION[defect_option(item)]]
    g = answers["condition"][vision.CONDITION["good"]]
    return d / max(1e-9, d + g)


def from_defect(item, kind, p):
    if kind == "any":
        return p
    if item["task"] == "T3" and defect_option(item) == kind:
        return p
    return None


def question(item):
    if item["task"] == "T2":
        return "Does this component have a defect?"
    return f"Is there {vision.CONDITION[defect_option(item)]} on this component?"


def main():
    items = [i for i in load_items() if i["split"] in ("val", "test")]
    step2 = load_cache(tuple(STEP2.values()))
    laya = {name: {json.loads(l)["image"]: json.loads(l) for l in open(f"results/vision/{run}/predictions.jsonl")} for name, run in LAYA.items()}
    zoom = {json.loads(l)["image"]: json.loads(l) for l in open("results/zoom/resnet_views.jsonl")}
    timing = json.loads(timing_path().read_text()) if timing_path().exists() else {}
    special = set(STEP2) | set(LAYA) | {"qwen_edge", "resnet_crops_mean", "resnet_crops_max"}
    caches = {t: load_tool(t) for t in TOOLS if t not in special}
    caches = {t: c for t, c in caches.items() if c and t in timing}
    scores = {t: [r["score"] for r in c.values()] for t, c in caches.items() if "score" in next(iter(c.values()))}
    norm = {t: (statistics.mean(v), statistics.pstdev(v) or 1.) for t, v in scores.items()}
    jpeg = statistics.median(r["cloud"]["payload_bytes"] for r in step2.values() if "cloud" in r) if step2 else 35000
    resnet_s = JETSON * statistics.median(r["classifier"]["compute_ms"] for r in step2.values() if "classifier" in r) / 1000
    print(f"cached new tools: {sorted(caches)}", flush=True)
    rows = []
    for i in items:
        p, cost = {}, {}
        s2 = step2.get(i["item_id"], {})
        for tool, key in STEP2.items():
            if key in s2:
                p[tool] = from_step2(s2[key])
                ms = s2[key]["compute_ms"]
                if TOOLS[tool][0] == "cloud":
                    cost[tool] = {"compute_s": ms / 1000, "bytes": s2[key]["payload_bytes"] or jpeg}
                else:
                    cost[tool] = {"s": JETSON * ms / 1000}
        if "cloud" in s2:
            p["qwen_edge"] = from_step2(s2["cloud"])
            cost["qwen_edge"] = {"s": s2["cloud"]["compute_ms"] / 1000 + EDGE_RTT + jpeg * 8 / (EDGE_MBPS * 1e6), "bytes": jpeg}
        for tool in LAYA:
            row = laya[tool].get(i["image"])
            if row:
                p[tool] = from_laya(i, row["answers"])
                cost[tool] = ({"s": row["ms"] / 1000 + EDGE_RTT + jpeg * 8 / (EDGE_MBPS * 1e6), "bytes": jpeg} if tool == "laya_edge"
                              else {"s": JETSON * row["ms"] / 1000})
        views = zoom.get(i["image"])
        if views:
            view_scores = [from_classes(i, dict(zip(CLASSES, v))) for v in views["probs"]]
            p["resnet_crops_mean"], p["resnet_crops_max"] = float(np.mean(view_scores)), float(np.max(view_scores))
            cost["resnet_crops_mean"] = cost["resnet_crops_max"] = {"s": 6 * resnet_s}
        for tool, cache in caches.items():
            row = cache.get(i["image"])
            if not row:
                continue
            if "probs" in row:
                v = from_classes(i, row["probs"])
            elif "answers" in row:
                v = from_laya(i, row["answers"])
            elif "defect" in row:
                kind, value = next(iter(row["defect"].items()))
                v = from_defect(i, kind, value)
            else:
                mean, std = norm[tool]
                v = 1 / (1 + math.exp(-(row["score"] - mean) / std))
            if v is None:
                continue
            p[tool] = v
            ms = timing[tool]["gpu_ms"]
            location = TOOLS[tool][0]
            if location == "cloud":
                cost[tool] = {"compute_s": ms / 1000, "bytes": timing[tool]["payload_bytes"] or jpeg}
            elif tool in ("quality", "exposure", "size"):
                cost[tool] = {"s": ms / 1000}
            else:
                cost[tool] = {"s": JETSON * ms / 1000 + (RECUT_S if "recut" in tool else 0.)}
        p = {t: float(min(max(v, 0.), 1.)) for t, v in p.items() if v is not None and math.isfinite(v)}
        rows.append({"item_id": i["item_id"], "split": i["split"], "tower": i["tower"], "asset": i["asset"], "group": group_of(i["asset"]),
                     "task": i["task"], "label": i["label"], "defect": i["label"] != "good", "defect_option": defect_option(i),
                     "component": vision.COMPONENT[i["asset"]], "question": question(i), "image": i["image"],
                     "p": p, "cost": {t: cost[t] for t in p}})
    out = results_dir("tools_v3")
    with (out / "items.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    present = sorted({t for r in rows for t in r["p"]}, key=list(TOOLS).index)
    tools = {t: {"location": TOOLS[t][0], "description": TOOLS[t][1], "coverage": sum(t in r["p"] for r in rows) / len(rows),
                 "median_cost_s": statistics.median(r["cost"][t].get("s", r["cost"][t].get("compute_s", 0.)) for r in rows if t in r["p"])}
             for t in present}
    write_json(out / "tools.json", tools)
    print(f"items {len(rows)}, tools {len(tools)}", flush=True)
    for t, v in tools.items():
        print(f"  {t:20s} {v['location']:6s} coverage {v['coverage']:.2f} cost {v['median_cost_s']:.3f} s", flush=True)
    value_report(rows, present)


def value_report(rows, tools):
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    assets = sorted({r["asset"] for r in rows})
    prior = float(np.mean([r["defect"] for r in val]))

    def risk(items, probs):
        return 1000 * float(np.mean([answer_risk(i["defect_option"] if q >= BAYES else "good", i["label"]) for i, q in zip(items, probs)]))

    report = {"tools": tools, "single": {}}
    for t in tools:
        fv = [r for r in val if t in r["p"]]
        ft = [r for r in test if t in r["p"]]
        if len(fv) < 50 or len(ft) < 50 or len({r["defect"] for r in fv}) < 2:
            continue
        model = Fusion([t], assets).fit(fv, subsets=1)
        p = model.prob(ft, [{t}] * len(ft))
        q = corrected(p, prior, em_prior(p, prior))
        report["single"][t] = {"items": len(ft), "risk_per_1000": round(risk(ft, q), 1)}
    full = Fusion(tools, assets).fit(val)
    for name, called in (("resnet only", {"resnet"}), ("all tools", set(tools))):
        p = full.prob(test, [called] * len(test))
        q = corrected(p, prior, em_prior(p, prior))
        report[f"fusion, {name}"] = round(risk(test, q), 1)
    p_all = full.prob(test, [set(tools)] * len(test))
    report["test_prior_em"] = round(em_prior(p_all, prior), 4)
    report["test_prior_true"] = round(float(np.mean([r["defect"] for r in test])), 4)
    cheapest = []
    for r in test:
        options = sorted((r["cost"][t].get("s", r["cost"][t].get("compute_s", 0.)), t) for t in r["p"])
        ok = [t for _, t in options if (r["p"][t] >= 0.5) == r["defect"]]
        cheapest.append(ok[0] if ok else None)
    report["oracle_one_tool_correct_share"] = round(float(np.mean([c is not None for c in cheapest])), 4)
    report["oracle_cheapest_tool_counts"] = {t: cheapest.count(t) for t in tools if cheapest.count(t)}
    write_json(results_dir("tools_v3") / "value_report.json", report)
    print("value report:", json.dumps({k: v for k, v in report.items() if k not in ("tools",)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
