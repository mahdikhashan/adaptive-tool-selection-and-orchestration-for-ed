import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from orx_figstyle import PALETTE, WIDE, figure_grid, panel_labels, save, use_style

RESULTS = HERE.parent / "results" / "route" / "mission_warm"
LINES = [("central PPO", "anchor", PALETTE["blue"], "-"), ("FedAvg PPO", "anchor", PALETTE["orange"], "-"),
         ("central PPO", "warm", PALETTE["blue"], "--"), ("FedAvg PPO", "warm", PALETTE["orange"], "--")]


def curves(variant, cap, kind):
    out = []
    for path in sorted(RESULTS.glob(f"{variant}_cap{cap}_s*.json")):
        r = json.loads(path.read_text())
        out += [c for k, c in r["curves"].items() if k.startswith(kind)]
    return out


def main():
    use_style()
    budgets = ["0.1", "0.15", "0.2", "0.3"]
    caps = [("05", "5%"), ("025", "2.5%")]
    fig, axes = figure_grid(2, 4, width=WIDE, ratio=0.7, sharex=True)
    for row, (cap, cap_name) in enumerate(caps):
        rules = json.loads(next(RESULTS.glob(f"anchor_cap{cap}_s0.json")).read_text())["methods"]
        for col, b in enumerate(budgets):
            ax = axes[row, col]
            for kind, variant, color, ls in LINES:
                cs = curves(variant, cap, kind)
                rounds = [p["round"] for p in cs[0]]
                v = np.array([[p["test"][b]["risk_per_1000"] for p in c] for c in cs])
                label = f"{kind.replace(' PPO', '')}, {'anchored' if variant == 'anchor' else 'free'} PPO"
                ax.plot(rounds, v.mean(0), color=color, ls=ls, lw=1.1, label=label)
                ax.fill_between(rounds, v.mean(0) - v.std(0), v.mean(0) + v.std(0), color=color, alpha=.12, lw=0)
            ax.axhline(rules["paced cascade, tuned on val"][b]["risk_per_1000"], color=PALETTE["green"], ls="-.", lw=.9, label="paced rule (tuned on val)")
            ax.axhline(rules["greedy cascade with defer"][b]["risk_per_1000"], color=PALETTE["green"], ls=":", lw=.9, label="greedy rule")
            ax.set_xticks([0, 20, 40, 60])
            if row == 0:
                ax.text(.5, 1.02, f"{b} s per item", transform=ax.transAxes, ha="center", va="bottom", fontsize=7)
            if row == 1:
                ax.set_xlabel("PPO round")
            ax.set_ylabel(f"Risk per 1,000 (cap {cap_name})" if col == 0 else "Risk per 1,000")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=6)
    panel_labels(axes.ravel())
    save(fig, str(HERE / "fig_mission"))


if __name__ == "__main__":
    main()
