import json
import statistics
import sys

from fedorch.common import results_dir, write_json

KEYS = ["risk_per_1000", "defect_recall", "false_alarm_rate", "escalation_rate", "uplink_mb_per_1000",
        "truth_labels", "cloud_labels", "audit_uploads", "cloud_uploads", "upload_mb"]


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "weak"
    runs = {}
    for path in sorted(results_dir("fc2").glob(f"metrics_*_{tag}_s*.json")):
        for name, run in json.loads(path.read_text())["runs"].items():
            runs.setdefault(name, []).append(run)
    table = {}
    for name, rows in sorted(runs.items()):
        table[name] = {k: {"mean": round(statistics.mean(r[k] for r in rows), 3),
                           "std": round(statistics.pstdev(r[k] for r in rows), 3)} for k in KEYS if k in rows[0]}
        table[name]["seeds"] = len(rows)
        risk = table[name]["risk_per_1000"]
        print(f"{name:28s} risk {risk['mean']:7.1f} ± {risk['std']:5.1f}  truth {table[name]['truth_labels']['mean']:7.0f}  "
              f"upload MB {table[name]['upload_mb']['mean']:6.1f}  seeds {len(rows)}")
    write_json(results_dir("fc2") / f"summary_{tag}.json", table)


if __name__ == "__main__":
    main()
