import collections
import json
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results" / "route"
RULES = ("ResNet only", "greedy cascade with defer", "paced cascade, tuned on val")


def row(name, values):
    print(f"{name:44s} " + " | ".join(values))


def main():
    folder = RESULTS / (sys.argv[1] if len(sys.argv) > 1 else "mission_warm")
    groups = collections.defaultdict(list)
    for path in sorted(folder.glob("*.json")):
        variant, _, seed = path.stem.rpartition("_s")
        groups[variant].append(json.loads(path.read_text()))
    for variant, reports in sorted(groups.items()):
        budgets = [str(b) for b in reports[0]["budgets_per_item"]]
        print(f"\n[{variant}] cap {reports[0]['cap']} | {len(reports)} seeds | risk per 1,000 test items at " + ", ".join(f"{b} s" for b in budgets))
        for name in RULES:
            row(name, [f"{reports[0]['methods'][name][b]['risk_per_1000']:6.1f}" for b in budgets])
        for kind in ("central PPO", "FedAvg PPO"):
            runs = [m for r in reports for k, m in r["methods"].items() if k.startswith(kind)]
            if not runs:
                continue
            cells = []
            for b in budgets:
                v = [m[b]["risk_per_1000"] for m in runs]
                d = statistics.mean(m[b]["deferred"] for m in runs)
                cells.append(f"{statistics.mean(v):6.1f} ± {statistics.pstdev(v):4.1f} (defer {d:.3f})")
            row(f"{kind} (n={len(runs)})", cells)
            curves = [c for r in reports for k, c in r.get("curves", {}).items() if k.startswith(kind)]
            if curves and curves[0][0]["round"] == 0:
                row(f"  {kind} after imitation only", [f"{statistics.mean(c[0]['test'][b]['risk_per_1000'] for c in curves):6.1f}" for b in budgets])


if __name__ == "__main__":
    main()
