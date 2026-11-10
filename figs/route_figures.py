import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save, use_style

RESULTS = HERE.parent / "results" / "route"
LEARNED = (("central PPO", PALETTE["blue"]), ("FedAvg PPO", PALETTE["orange"]))


def seeds(s, kind, b, key):
    return np.array([s["methods"][f"{kind} seed {k}"][-1]["test"][b][key] for k in range(3)])


def route():
    s = {L: json.loads((RESULTS / f"summary_lambda{L}.json").read_text()) for L in ("0.01", "0.03")}
    m = s["0.01"]["methods"]
    budgets = list(m["ResNet only"])
    x = np.array([float(b) for b in budgets])
    fig, (a, b) = figure_grid(1, 2, width=WIDE, ratio=0.36)
    fixed = [("ResNet only", BASELINE, ":"), ("all tools that fit", PALETTE["purple"], "-"),
             ("uncertainty cascade", PALETTE["green"], "-"), ("oracle subset (labels)", BASELINE, "--")]
    for name, color, ls in fixed:
        a.plot(x, [m[name][k]["risk_per_1000"] for k in budgets], color=color, ls=ls, marker="o", ms=3, label=name)
    for kind, color in LEARNED:
        r = [seeds(s["0.01"], kind, k, "risk_per_1000") for k in budgets]
        a.errorbar(x, [v.mean() for v in r], yerr=[v.std() for v in r], color=color, marker="o", ms=3, capsize=2, label=kind)
    a.set_xscale("log")
    a.set_xticks(x, budgets)
    a.minorticks_off()
    a.set_xlabel("Time budget per item (s)")
    a.set_ylabel("Defect risk per 1,000 test items")
    a.legend(frameon=False, fontsize=6, loc="lower left")
    for name, color in (("all tools that fit", PALETTE["purple"]), ("uncertainty cascade", PALETTE["green"])):
        b.plot([m[name][k]["mean_time_s"] for k in budgets], [m[name][k]["risk_per_1000"] for k in budgets],
               color=color, marker="s", ms=3, label=name)
    for L, fill, ls in (("0.01", "full", "-"), ("0.03", "none", "--")):
        for kind, color in LEARNED:
            t = [seeds(s[L], kind, k, "mean_time_s").mean() for k in budgets]
            r = [seeds(s[L], kind, k, "risk_per_1000").mean() for k in budgets]
            b.plot(t, r, color=color, marker="o", ms=4, fillstyle=fill, ls=ls, label=f"{kind}, λ = {L}")
    b.plot(m["ResNet only"]["0.25"]["mean_time_s"], m["ResNet only"]["0.25"]["risk_per_1000"], marker="o", color=BASELINE, ms=4, ls="", label="ResNet only")
    b.set_xlabel("Mean time spent per item (s)")
    b.set_ylabel("Defect risk per 1,000 test items")
    b.legend(frameon=False, fontsize=5.5, loc="upper right")
    panel_labels([a, b])
    save(fig, str(HERE / "fig_route"))


def drones():
    fig, axes = figure_grid(1, 2, width=WIDE, ratio=0.36, sharey=True)
    fixed = [("fixed cascade", PALETTE["green"], "s"), ("cascade tuned per drone", PALETTE["green"], "D"), ("all tools that fit", PALETTE["purple"], "s")]
    learned = [("central PPO", PALETTE["blue"]), ("local PPO", PALETTE["red"]), ("FedAvg PPO", PALETTE["orange"]), ("FedAvg + local fine-tune", PALETTE["cyan"])]
    for ax, split in zip(axes, ("components", "towers")):
        m = json.loads((RESULTS / f"drones_{split}_lambda0.01.json").read_text())["methods"]
        budgets = ["0.75", "1.5", "3.0"]
        for name, color, marker in fixed:
            ax.plot([m[name][k]["fleet"]["mean_time_s"] for k in budgets], [m[name][k]["fleet"]["risk_per_1000"] for k in budgets],
                    color=color, marker=marker, ms=3, ls="--" if "tuned" in name else "-", label=name.replace("fixed", "static"))
        for name, color in learned:
            r = np.array([[m[f"{name} seed {s}"][k]["fleet"]["risk_per_1000"] for s in range(3)] for k in budgets])
            t = np.array([[m[f"{name} seed {s}"][k]["fleet"]["mean_time_s"] for s in range(3)] for k in budgets])
            ax.errorbar(t.mean(1), r.mean(1), yerr=r.std(1), color=color, marker="o", ms=3, capsize=2, label=name)
        ax.axhline(m["ResNet only"]["3.0"]["fleet"]["risk_per_1000"], color=BASELINE, ls=":", lw=0.8)
        ax.set_xlabel("Mean time spent per item (s)")
    axes[0].set_ylabel("Defect risk per 1,000 test items")
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    panel_labels(axes)
    save(fig, str(HERE / "fig_drones"))


if __name__ == "__main__":
    use_style()
    route()
    drones()
