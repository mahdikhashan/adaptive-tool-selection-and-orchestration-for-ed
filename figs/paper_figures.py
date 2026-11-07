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
    fig, axes = figure_grid(1, 3, width=width, ratio=0.36 if width == TEXT else 0.27)
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


def uplink_gb(run):
    r = run_info(run)
    return r.get("uplink_mb_total", r["uplink_mb_total_fp16"]) / 1000


def fig_round_b_2col():
    fig, axes = figure_grid(1, 2, width=WIDE2, ratio=0.27)
    curves = [("head only", "fl_b_head"), ("LoRA r8 + head", "fl_b_lora8"), ("last 4 + head", "fl_b_last4"),
              ("all language layers", "fl_b_layers"), ("full model", "fl_fedavg_k6"), ("last 4 + head, 30 rounds", "fl_b_last4_r30")]
    colors = family("blue", 3) + [PALETTE["orange"], PALETTE["red"], BASELINE]
    for (name, run), color in zip(curves, colors):
        log = run_info(run)["rounds_log"]
        axes[0].plot([x["round"] + 1 for x in log], [x["acc"] for x in log], color=color, label=name,
                     ls="--" if run == "fl_b_last4_r30" else "-", marker="o" if len(log) < 15 else None, markersize=2)
    axes[0].set_xlabel("FedAvg rounds completed")
    axes[0].set_ylabel("Val accuracy")
    axes[0].set_ylim(0.55, 0.9)
    axes[0].legend(loc="upper left", fontsize=5.5, ncols=2)
    points = [("head", "fl_b_head"), ("LoRA", "fl_b_lora8"), ("last 4, int8", "fl_b_last4_int8"), ("last 4", "fl_b_last4"),
              ("last 4, FedProx", "fl_b_last4_prox"), ("last 4, 30 rounds", "fl_b_last4_r30"), ("all layers", "fl_b_layers"),
              ("full", "fl_fedavg_k6"), ("full, FedProx", "fl_b_full_prox")]
    for name, run in points:
        v, lo, hi = metric(run, "condition", "macro_f1", "condition_macro_f1")
        x = uplink_gb(run)
        axes[1].errorbar([x], [v], yerr=[[v - lo], [hi - v]], marker="o", color=PALETTE["blue"], capsize=1.2, lw=0.8, markersize=3)
        dx, dy = {"head": (-4, 2), "LoRA": (3, 6), "last 4": (3, 8), "last 4, FedProx": (3, -10), "full": (3, 7),
                  "full, FedProx": (3, -10), "last 4, 30 rounds": (3, 6), "all layers": (3, -10), "last 4, int8": (3, 9)}[name]
        axes[1].annotate(name, (x, v), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, ha="right" if name == "head" else "left")
    axes[1].axhline(summary("good300_full")["test"]["condition"]["macro_f1"], color=BASELINE, ls="--", lw=0.8)
    axes[1].text(1.05, 0.71, "central", fontsize=5.5, color=BASELINE)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.7, 20)
    axes[1].set_xlabel("Total uplink of the run (GB, fp16 or int8)")
    axes[1].set_ylabel("Condition macro F1")
    axes[1].set_ylim(0.2, 0.76)
    panel_labels(axes)
    save(fig, str(HERE / "fig2_roundb"))


def fig_round_c_2col():
    fig, axes = figure_grid(1, 2, width=COL2, ratio=0.58)
    budgets = [50, 100, 200]
    for ax, (label, part, key, ci) in zip(axes, [("Condition macro F1", "condition", "macro_f1", "condition_macro_f1"),
                                                ("Defect AUROC", "defect_noul", "auroc", "defect_auroc")]):
        for name, prefix, bs, color, marker in [("FedAvg, fine-tuned start", "fl_c_warm", budgets, PALETTE["red"], "o"),
                                                ("FedAvg, base start", "fl_c_base", budgets, PALETTE["blue"], "s"),
                                                ("pooled centrally", "fl_c_warm_central", [50, 200], BASELINE, "^")]:
            style = "none" if prefix == "fl_c_warm_central" else "-"
            vals = [metric(f"{prefix}_n{n}", part, key, ci) for n in bs]
            v = np.array([a for a, _, _ in vals])
            ax.errorbar(bs, v, yerr=[v - [lo for _, lo, _ in vals], [hi for _, _, hi in vals] - v], marker=marker, color=color,
                        capsize=1.2, lw=0.9, markersize=2.5, label=name, ls=style)
        local = [[summary(f"fl_c_warm_local_n{n}_d{i}")["test"][part][key] for i in range(5)] for n in (50, 200)]
        m = np.array([np.mean(x) for x in local])
        ax.errorbar([50, 200], m, yerr=[m - [min(x) for x in local], [max(x) for x in local] - m], marker="D", color=PALETTE["orange"],
                    capsize=1.2, lw=0.9, markersize=2.5, ls="none", label="each drone alone")
        ax.axhline(summary("fl_server_central")["test"][part][key], color=MUTED, ls="--", lw=0.8, label="server model only")
        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel("Labels per drone")
        ax.set_ylabel(label)
    axes[0].set_ylim(0.1, 0.7)
    axes[1].set_ylim(0.7, 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, fontsize=5, bbox_to_anchor=(0.5, -0.01))
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    panel_labels(axes)
    save(fig, str(HERE / "fig2_roundc"))


if __name__ == "__main__":
    use_style()
    fig_finetune()
    fig_federated()
    fig_starts()
    fig_finetune_2col()
    fig_federated(WIDE2, "fig2_federated")
    fig_starts_2col()
    fig_round_b_2col()
    fig_round_c_2col()
