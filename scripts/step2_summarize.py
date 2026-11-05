import collections
import json
import statistics

from fedorch.common import results_dir, write_json
from fedorch.items import load_items

COST_ORDER = ["classifier", "smolvlm_local", "smolvlm_split", "cloud"]


def read_rows(tool):
    path = results_dir("step2") / f"tool_{tool}.jsonl"
    return {row["item_id"]: row for row in map(json.loads, path.open())} if path.exists() else {}


def rate(values):
    values = list(values)
    return round(sum(values) / len(values), 4) if values else None


def tool_summary(rows, items):
    pairs = [(items[i], r) for i, r in rows.items()]
    by_label = collections.defaultdict(list)
    for item, r in pairs:
        by_label[item["label"]].append(r["correct"])
    return {
        "items": len(pairs),
        "accuracy": rate(r["correct"] for _, r in pairs),
        "accuracy_by_task": {t: rate(r["correct"] for i, r in pairs if i["task"] == t) for t in ["T2", "T3"]},
        "accuracy_by_split": {s: rate(r["correct"] for i, r in pairs if i["split"] == s) for s in ["train", "val", "test"]},
        "recall_by_label": {label: rate(v) for label, v in sorted(by_label.items())},
        "abstention_rate": rate(r["abstained"] for _, r in pairs),
        "nonfinite": sum(not r["finite"] for _, r in pairs),
        "median_compute_ms": round(statistics.median(r["compute_ms"] for _, r in pairs), 2),
        "mean_payload_bytes": round(statistics.mean(r["payload_bytes"] for _, r in pairs)),
    }


def cheapest_correct(rows_by_tool, items):
    shared = set(items).intersection(*(rows.keys() for rows in rows_by_tool.values()))
    counts = collections.Counter()
    for item_id in shared:
        counts[next((t for t in COST_ORDER if rows_by_tool[t][item_id]["correct"]), "none")] += 1
    return {"items": len(shared), "share": {t: round(n / len(shared), 4) for t, n in counts.items()} if shared else {}}


def main():
    items = {item["item_id"]: item for item in load_items()}
    rows_by_tool = {tool: read_rows(tool) for tool in ["classifier", "smolvlm_base", "smolvlm_local", "smolvlm_split", "cloud"]}
    summary = {tool: tool_summary(rows, items) for tool, rows in rows_by_tool.items() if rows}
    if all(rows_by_tool[t] for t in COST_ORDER):
        summary["oracle_cheapest_correct"] = cheapest_correct({t: rows_by_tool[t] for t in COST_ORDER}, items)
    write_json(results_dir("step2") / "summary.json", summary)
    print(json.dumps({t: {k: s[k] for k in ("items", "accuracy", "median_compute_ms")} for t, s in summary.items() if "accuracy" in s}, indent=2))


if __name__ == "__main__":
    main()
