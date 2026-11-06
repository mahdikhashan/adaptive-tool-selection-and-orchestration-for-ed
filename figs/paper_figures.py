import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from orx_figstyle import BASELINE, MUTED, PALETTE, TEXT, family, figure, figure_grid, panel_labels, save, use_style

RESULTS = HERE.parent / "results" / "vision"
METRICS = [("Condition\nmacro F1", "condition", "macro_f1", "condition_macro_f1"),
           ("Component\naccuracy", "component", "accuracy", "component_accuracy"),
           ("Defect\nAUROC", "defect_noul", "auroc", "defect_auroc")]


def summary(run):
    return json.loads((RESULTS / run / "summary.json").read_text())


def run_info(run):
    return json.loads((RESULTS / run / "run.json").read_text())


def metric(run, part, key, ci_key):
    s = summary(run)
    lo, hi = s["test_bootstrap_95ci"][ci_key]
    return s["test"][part][key], lo, hi


def grouped(ax, runs, colors):
    width = 0.8 / len(runs)
    for i, ((name, run), color) in enumerate(zip(runs, colors)):
        vals = [metric(run, *m[1:]) for m in METRICS]
        x = np.arange(len(METRICS)) + (i - (len(runs) - 1) / 2) * width
        v = np.array([a for a, _, _ in vals])
        err = np.array([[a - lo for a, lo, _ in vals], [hi - a for a, _, hi in vals]])
        ax.bar(x, v, width * 0.92, color=color, label=name, yerr=err, error_kw=dict(lw=0.6, capsize=1.2, ecolor="#404040"))
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.03, f"{vi:.2f}", ha="center", va="bottom", fontsize=5, rotation=90)
    ax.set_xticks(np.arange(len(METRICS)), [m[0] for m in METRICS])
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Score on test towers")


COL2 = 252 / 72.27
WIDE2 = 522 / 72.27


def legend_above(ax, ncols):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02, 1, 0.2), mode="expand", ncols=ncols, fontsize=6, borderaxespad=0)


def fig_finetune_2col():
    runs = [("Zero-shot", "zeroshot"), ("R1: 1,793 imgs, last 4", "lastn_pc400"),
            ("R2: 4,807 imgs, last 4", "good300_lastn"), ("R2: 4,807 imgs, full", "good300_full")]
    fig, ax = figure(width=COL2, ratio=0.78)
    grouped(ax, runs, [MUTED] + family("red", 3))
    ax.set_ylim(0, 1.15)
    legend_above(ax, 2)
    save(fig, str(HERE / "fig2_finetune"))


def fig_starts_2col():
    runs = [("Server towers only", "fl_server_central"), ("Fleet FedAvg, base start", "fl_fleet_base_k5"),
            ("Fleet FedAvg, fine-tuned start", "fl_fleet_warm_k5"), ("Central, all towers", "good300_full")]
    fig, ax = figure(width=COL2, ratio=0.78)
    grouped(ax, runs, [MUTED, PALETTE["blue"], PALETTE["red"], BASELINE])
    ax.set_ylim(0, 1.15)
    legend_above(ax, 2)
    save(fig, str(HERE / "fig2_starts"))


def fig_finetune():
    runs = [("Zero-shot", "zeroshot"), ("Round 1: 1,793 images, last 4 layers", "lastn_pc400"),
            ("Round 2: 4,807 images, last 4 layers", "good300_lastn"), ("Round 2: 4,807 images, full", "good300_full")]
    fig, ax = figure(width=TEXT, ratio=0.42)
    grouped(ax, runs, [MUTED] + family("red", 3))
    ax.legend(loc="upper left", ncols=2, frameon=True, framealpha=1, edgecolor="none", fontsize=6)
    save(fig, str(HERE / "fig_finetune"))


def fig_federated(width=TEXT, stem="fig_federated"):
    fig, axes = figure_grid(1, 3, width=width, ratio=0.36 if width == TEXT else 0.3)
    curves = [("1 drone (control)", "fl_central_k1"), ("3 drones", "fl_fedavg_k3"), ("6 drones", "fl_fedavg_k6"), ("18 drones", "fl_fedavg_k18")]
    colors = [BASELINE] + family("blue", 3)
    for (name, run), color in zip(curves, colors):
        log = run_info(run)["rounds_log"]
        axes[0].plot([x["round"] + 1 for x in log], [x["acc"] for x in log], marker="o", color=color, label=name)
    axes[0].set_xlabel("FedAvg rounds completed")
    axes[0].set_ylabel("Val accuracy")
    axes[0].set_ylim(0.55, 1.0)
    axes[0].legend(loc="lower right", fontsize=6)
    drones = [(1, "fl_central_k1"), (3, "fl_fedavg_k3"), (6, "fl_fedavg_k6"), (18, "fl_fedavg_k18")]
    for ax, (label, part, key, ci_key) in zip(axes[1:], [METRICS[0], METRICS[2]]):
        vals = [metric(run, part, key, ci_key) for _, run in drones]
        k = [d for d, _ in drones]
        v = np.array([a for a, _, _ in vals])
        ax.errorbar(k, v, yerr=[v - [lo for _, lo, _ in vals], [hi for _, _, hi in vals] - v], marker="o",
                    color=PALETTE["blue"], capsize=1.5, lw=1.0, label="FedAvg")
        local = [summary(f"fl_local_k6_d{i}")["test"][part][key] for i in range(6)]
        ax.errorbar([6.6], [np.mean(local)], yerr=[[np.mean(local) - min(local)], [max(local) - np.mean(local)]], marker="s",
                    color=PALETTE["orange"], capsize=1.5, lw=1.0, label="local only (6)")
        ref = summary("good300_full")["test"][part][key]
        ax.axhline(ref, color=BASELINE, lw=0.8, ls="--", label="central")
        ax.set_xscale("log")
        ax.set_xticks(k, [str(d) for d in k])
        ax.minorticks_off()
        ax.set_xlabel("Drones")
        ax.set_ylabel(label.replace("\n", " "))
    axes[1].set_ylim(0.2, 0.8)
    axes[2].set_ylim(0.8, 1.0)
    axes[1].legend(loc="lower left", fontsize=6)
    panel_labels(axes)
    save(fig, str(HERE / stem))


def fig_starts():
    runs = [("Server towers only (start model)", "fl_server_central"),
            ("Fleet FedAvg, base start", "fl_fleet_base_k5"),
            ("Fleet FedAvg, fine-tuned start", "fl_fleet_warm_k5"),
            ("Central, all towers", "good300_full")]
    fig, ax = figure(width=TEXT, ratio=0.42)
    grouped(ax, runs, [MUTED, PALETTE["blue"], PALETTE["red"], BASELINE])
    ax.legend(loc="upper left", ncols=2, frameon=True, framealpha=1, edgecolor="none", fontsize=6)
    save(fig, str(HERE / "fig_starts"))


if __name__ == "__main__":
    use_style()
    fig_finetune()
    fig_federated()
    fig_starts()
    fig_finetune_2col()
    fig_federated(WIDE2, "fig2_federated")
    fig_starts_2col()
