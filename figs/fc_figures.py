import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from orx_figstyle import BASELINE, PALETTE, WIDE, figure_grid, panel_labels, save, use_style

RESULTS = HERE.parent / "results"
AUDITS = [0.0, 0.01, 0.02, 0.05]


def load(path):
    return json.loads((RESULTS / path).read_text())


def main():
    use_style()
    fc1, fc2, step3 = load("fc1/metrics_weak.json"), load("fc2/summary_weak.json"), load("step3/metrics.json")
    cascade, oracle = fc1["static_cascade"]["risk_per_1000"], step3["policies"]["oracle"]["weak"]["risk_per_1000"]
    fig, (a, b) = figure_grid(1, 2, width=WIDE, ratio=0.34, sharey=True)

    groups = [("Central", "central"), ("FedAvg", "fedavg"), ("FedProx", "fedprox")] + \
             [(f"R{r + 1}", f"local_r{r}") for r in range(5)]
    for x, (label, method) in enumerate(groups):
        risks = [run["risk_per_1000"] for run in fc1["runs"].values() if run["method"] == method]
        color = PALETTE["blue"] if method in ("fedavg", "fedprox") else BASELINE if method == "central" else PALETTE["orange"]
        a.bar(x, statistics.mean(risks), color=color, width=0.7)
        a.errorbar(x, statistics.mean(risks), yerr=statistics.pstdev(risks), color="black", capsize=2, lw=0.8)
        a.scatter([x] * len(risks), risks, s=6, color="black", zorder=3)
    a.set_xticks(range(len(groups)), [g[0] for g in groups])
    a.set_ylabel("Mission risk per 1,000 frames")
    a.set_xlabel("Controller training (R1–R5: one region alone)")

    lines = [("Full labels", "full", BASELINE, "-"), ("Cloud as truth", "cloud_oracle", PALETTE["red"], "-"),
             ("True labels", "truth_naive", PALETTE["purple"], "-"), ("True labels, IPW", "truth_ipw", PALETTE["blue"], "-"),
             ("Multi-fidelity", "multifid", PALETTE["green"], "-")]
    for label, learner, color, dash in lines:
        keys = [f"fed_{learner}_a{x}" for x in AUDITS]
        mean = np.array([fc2[k]["risk_per_1000"]["mean"] for k in keys])
        std = np.array([fc2[k]["risk_per_1000"]["std"] for k in keys])
        b.plot(np.array(AUDITS) * 100, mean, marker="o", ms=3, color=color, ls=dash, label=label + ", federated")
        b.fill_between(np.array(AUDITS) * 100, mean - std, mean + std, color=color, alpha=0.15, lw=0)
    for label, learner, color in [("True labels, IPW", "truth_ipw", PALETTE["blue"])]:
        keys = [f"local_{learner}_a{x}" for x in AUDITS[1:]]
        b.plot(np.array(AUDITS[1:]) * 100, [fc2[k]["risk_per_1000"]["mean"] for k in keys], marker="s", ms=3,
               color=color, ls="--", label=label + ", regions alone")
    b.set_xlabel("Audit rate λ (% of non-cloud frames)")
    b.set_xticks([0, 1, 2, 5])

    for ax in (a, b):
        ax.axhline(cascade, color=BASELINE, ls=":", lw=1)
        ax.axhline(oracle, color=BASELINE, ls="-.", lw=1)
    a.text(len(groups) - 0.6, cascade + 2, "static cascade", ha="right", va="bottom", fontsize=6, color=BASELINE)
    a.text(len(groups) - 0.6, oracle + 2, "oracle", ha="right", va="bottom", fontsize=6, color=BASELINE)
    a.set_ylim(50, 200)
    b.legend(loc="lower left", bbox_to_anchor=(0.0, 0.11), ncol=2, frameon=False, fontsize=6)
    panel_labels([a, b])
    save(fig, str(HERE / "fig_fc"))


if __name__ == "__main__":
    main()
