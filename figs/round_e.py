import numpy as np
from paper_figures import BASELINE, MUTED, PALETTE, figure_grid, metric, panel_labels, run_info, save, summary, use_style, HERE, COL2, WIDE2

RUNS = [("full, int8", "fl_e1_full", PALETTE["red"]), ("20 layers, int8", "fl_e1_layers", PALETTE["orange"]),
        ("last 4, int8", "fl_e1_last4", PALETTE["blue"]), ("LoRA r8, int8", "fl_e1_lora8", BASELINE)]


def main():
    use_style()
    fig, axes = figure_grid(1, 3, width=WIDE2, ratio=0.3)
    for name, run, color in RUNS:
        log = run_info(run)["rounds_log"]
        axes[0].plot([x["round"] + 1 for x in log], [x["acc"] for x in log], marker="o", markersize=2, color=color, label=name)
    axes[0].set_xlabel("FedAvg rounds completed")
    axes[0].set_ylabel("Val accuracy (600 images)")
    axes[0].legend(fontsize=5.5)
    server = summary("fl_server_central")["test"]
    fp16 = summary("fl_fleet_warm_k5")["test"]
    for ax, part, key, ci, label in [(axes[1], "condition", "macro_f1", "condition_macro_f1", "Condition macro F1"),
                                     (axes[2], "defect_noul", "auroc", "defect_auroc", "Defect AUROC")]:
        for name, run, color in RUNS:
            v, lo, hi = metric(run, part, key, ci)
            x = run_info(run)["uplink_mb_total"]
            ax.errorbar([x], [v], yerr=[[v - lo], [hi - v]], marker="o", color=color, capsize=1.2, markersize=3)
            if part == "condition":
                ax.plot([x], [summary(run)["prior_corrected"]["test"]["condition"]["macro_f1"]], marker="x", color=color, markersize=4)
        v, lo, hi = metric("fl_fleet_warm_k5", part, key, ci)
        ax.errorbar([11475], [v], yerr=[[v - lo], [hi - v]], marker="s", color=MUTED, capsize=1.2, markersize=3)
        ax.annotate("full, fp16", (11475, v), xytext=(0, -14), textcoords="offset points", fontsize=5.5, ha="center")
        ax.axhline(server[part][key], color=MUTED, ls="--", lw=0.8)
        ax.text(420, server[part][key], "server model", fontsize=5.5, color=MUTED, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(400, 16000)
        ax.set_xlabel("Total uplink of the run (MB)")
        ax.set_ylabel(label)
    axes[1].plot([], [], "x", color=MUTED, label="class-share corrected")
    axes[1].legend(fontsize=5.5, loc="lower right")
    panel_labels(axes)
    save(fig, str(HERE / "fig_round_e1"))


if __name__ == "__main__":
    main()
