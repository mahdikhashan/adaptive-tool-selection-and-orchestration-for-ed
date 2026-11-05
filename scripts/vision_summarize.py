import collections
import json

import numpy as np

from fedorch.common import results_dir, write_json
from fedorch.items import load_items
from fedorch.vision import answer_risk, labels_of


def metrics(rows):
    correct = [p == i["label"] for i, p in rows]
    defects = [(i, p) for i, p in rows if i["label"] != "good"]
    by_label = collections.defaultdict(list)
    for i, p in rows:
        by_label[i["label"]].append(p == i["label"])
    return {"items": len(rows),
            "accuracy": round(float(np.mean(correct)), 4),
            "risk_per_1000": round(1000 * sum(answer_risk(p, i["label"]) for i, p in rows) / len(rows), 2),
            "defect_recall": round(float(np.mean([p not in ("good", "cannot-judge") for _, p in defects])), 4),
            "abstain_rate": round(float(np.mean([p == "cannot-judge" for _, p in rows])), 4),
            "recall_by_label": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_label.items())}}


KEY_FRAMES_PER_SEGMENT = 3.45
JETSON_FACTOR = 9.0
EVLM = {"hardware": "NVIDIA Jetson, 8-core CPU, 16 GB, 1024 CUDA cores",
        "evlm_segment_s (Table V, 4 s segment, histogram filter + generation)": 6.448,
        "evlm_defect_detection_s (Fig. 8a, other intents 1.4 to 2.3)": [1.4, 2.3],
        "histogram_filter_ms_per_frame (Table IV)": 1.9}


def latency(run):
    t = run["latency"]["total_ms"]
    segment = KEY_FRAMES_PER_SEGMENT * t["median"] / 1000
    return {"device": run["device"], "gpu": run["gpu"], "precision": run["precision"],
            "batch_1": run["latency"], "batch_throughput_ms_per_item": run["batch_ms_per_item"],
            "segment_s_measured": round(segment, 3),
            "segment_s_jetson_estimate_x9": round(JETSON_FACTOR * segment, 3),
            "evlm_reference": EVLM}


def main():
    items = {i["item_id"]: i for i in load_items()}
    path = results_dir("vision/zeroshot") / "scores.jsonl"
    rows = []
    for line in path.open():
        row = json.loads(line)
        item = items[row["item_id"]]
        rows.append((item, labels_of(item)[int(np.argmax(row["probs"]))]))
    summary = {"all": metrics(rows)}
    for split in ("train", "val", "test"):
        summary[split] = metrics([r for r in rows if r[0]["split"] == split])
    for task in ("T2", "T3"):
        summary[f"test {task}"] = metrics([r for r in rows if r[0]["split"] == "test" and r[0]["task"] == task])
    summary["latency"] = latency(json.loads((results_dir("vision/zeroshot") / "run.json").read_text()))
    cpu = results_dir("vision") / "latency_cpu.json"
    if cpu.exists():
        summary["latency_cpu"] = json.loads(cpu.read_text())
    write_json(results_dir("vision") / "summary.json", summary)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
