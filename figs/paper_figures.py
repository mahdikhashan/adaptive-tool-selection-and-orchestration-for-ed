import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
        x = 1000 * uplink_gb(run)
        axes[1].errorbar([x], [v], yerr=[[v - lo], [hi - v]], marker="o", color=PALETTE["blue"], capsize=1.2, lw=0.8, markersize=3)
        dx, dy = {"head": (-4, 2), "LoRA": (3, 6), "last 4": (3, 8), "last 4, FedProx": (3, -10), "full": (3, 7),
                  "full, FedProx": (3, -10), "last 4, 30 rounds": (3, 6), "all layers": (3, -10), "last 4, int8": (3, 9)}[name]
        axes[1].annotate(name, (x, v), xytext=(dx, dy), textcoords="offset points", fontsize=5.5, ha="right" if name == "head" else "left")
    axes[1].axhline(summary("good300_full")["test"]["condition"]["macro_f1"], color=BASELINE, ls="--", lw=0.8)
    axes[1].text(1050, 0.71, "central", fontsize=5.5, color=BASELINE)
    axes[1].set_xscale("log")
    axes[1].set_xlim(700, 20000)
    axes[1].set_xlabel("Total uplink of the run (MB, fp16 or int8)")
    axes[1].set_ylabel("Condition macro F1")
    axes[1].set_ylim(0.2, 0.76)
    panel_labels(axes)
    save(fig, str(HERE / "fig2_roundb"))


def fig_round_c_2col():
    fig, axes = figure_grid(1, 2, width=COL2, ratio=0.58)
    budgets = [25, 50, 100, 200]
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


def fleet_defects(run):
    return sum(round(g["defect_share"] * g["train_images"]) for g in run_info(run)["drones_info"])


def fig_overview_2col():
    groups = [("Round A", PALETTE["blue"], [("control, 1 drone", "fl_central_k1"), ("3 drones", "fl_fedavg_k3"), ("6 drones", "fl_fedavg_k6"),
                                            ("18 drones", "fl_fedavg_k18"), ("fleet, base start", "fl_fleet_base_k5"), ("fleet, fine-tuned start", "fl_fleet_warm_k5")]),
              ("Round B", PALETTE["orange"], [("head", "fl_b_head"), ("LoRA", "fl_b_lora8"), ("last 4", "fl_b_last4"), ("last 4, int8", "fl_b_last4_int8"),
                                              ("last 4, FedProx", "fl_b_last4_prox"), ("last 4, 30 rounds", "fl_b_last4_r30"), ("all layers", "fl_b_layers"),
                                              ("full, FedProx", "fl_b_full_prox")]),
              ("Round C", PALETTE["red"], [(f"fleet, {n} labels/drone", f"fl_c_warm_n{n}") for n in (25, 50, 100, 200)]),
              ("Round D", BASELINE, [("random split", "fl_d_iid"), ("component split", "fl_d_component"), ("component + FedProx", "fl_d_component_prox"),
                                     ("Dirichlet split", "fl_d_dirichlet")])]
    central = summary("good300_full")["test"]["condition"]["macro_f1"]
    rows = []
    for group, color, runs in groups:
        for name, run in runs:
            v, lo, hi = metric(run, "condition", "macro_f1", "condition_macro_f1")
            rows.append((v - central, lo - central, hi - central, name, color, group))
    rows.sort(key=lambda r: r[0])
    fig, ax = figure(width=COL2, ratio=1.2)
    for i, (v, lo, hi, name, color, group) in enumerate(rows):
        ax.errorbar([v], [i], xerr=[[v - lo], [hi - v]], marker="o", color=color, capsize=1.2, lw=0.8, markersize=3)
    ax.set_yticks(range(len(rows)), [r[3] for r in rows], fontsize=5.5)
    ax.axvline(0, color=BASELINE, lw=0.8, ls="--")
    ax.set_xlabel("Condition macro F1 minus central training")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for group, color, _ in groups:
        ax.plot([], [], "o", color=color, label=group, markersize=3)
    ax.legend(loc="lower right", fontsize=5.5)
    save(fig, str(HERE / "fig2_overview"))


def fig_round_d_2col():
    names = ["shackle", "vari-grip clamp", "glass insulator, spacer", "lightning rod, plate", "yoke susp., yoke", "polymer insul., dampers"]
    rows = [(f"alone: {names[i]}", f"fl_d_component_local_d{i}") for i in range(6)]
    rows += [("FedAvg last 4, component", "fl_d_component"), ("same + FedProx", "fl_d_component_prox"), ("FedAvg last 4, random", "fl_d_iid"),
             ("FedAvg last 4, Dirichlet", "fl_d_dirichlet"), ("FedAvg full, tower (round A)", "fl_fedavg_k6"), ("central full model", "good300_full")]
    conditions = ["good, no defect", "rust", "corrosion", "missing cap", "bird nest"]
    columns = ["good", "rust", "corrosion", "missing\ncap", "bird\nnest", "comp.\nacc."]
    values = []
    for _, run in rows:
        t = summary(run)["test"]
        values.append([t["condition"]["per_class"][c]["recall"] for c in conditions] + [t["component"]["accuracy"]])
    values = np.array(values)
    fig, ax = figure(width=COL2, ratio=0.8)
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if values[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(columns)), [c.replace("\\n", "\n") for c in columns], fontsize=5.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=5.5)
    ax.axhline(5.5, color="white", lw=1.2)
    ax.axhline(9.5, color="white", lw=1.2)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).set_label("Test recall / accuracy", fontsize=6)
    save(fig, str(HERE / "fig2_roundd"))


def fig_round_c_defects_2col():
    fig, axes = figure_grid(1, 2, width=COL2, ratio=0.58)
    for ax, (label, part, key, ci) in zip(axes, [("Condition macro F1", "condition", "macro_f1", "condition_macro_f1"),
                                                ("Defect AUROC", "defect_noul", "auroc", "defect_auroc")]):
        for name, prefix, bs, color, marker, style in [("FedAvg, fine-tuned start", "fl_c_warm", [25, 50, 100, 200], PALETTE["red"], "o", "-"),
                                                       ("FedAvg, base start", "fl_c_base", [25, 50, 100, 200], PALETTE["blue"], "s", "-"),
                                                       ("pooled on one server", "fl_c_warm_central", [50, 200], BASELINE, "^", "none")]:
            x = [fleet_defects(f"{prefix}_n{n}") if prefix != "fl_c_warm_central" else fleet_defects(f"fl_c_warm_n{n}") for n in bs]
            vals = [metric(f"{prefix}_n{n}", part, key, ci) for n in bs]
            v = np.array([a for a, _, _ in vals])
            ax.errorbar(x, v, yerr=[v - [lo for _, lo, _ in vals], [hi for _, _, hi in vals] - v], marker=marker, color=color,
                        capsize=1.2, lw=0.9, markersize=2.5, label=name, ls=style)
        local = [[summary(f"fl_c_warm_local_n{n}_d{i}")["test"][part][key] for i in range(5)] for n in (50, 200)]
        m = np.array([np.mean(a) for a in local])
        ax.errorbar([fleet_defects(f"fl_c_warm_n{n}") for n in (50, 200)], m, yerr=[m - [min(a) for a in local], [max(a) for a in local] - m],
                    marker="D", color=PALETTE["orange"], capsize=1.2, lw=0.9, markersize=2.5, ls="none", label="each drone alone")
        ax.axhline(summary("fl_server_central")["test"][part][key], color=MUTED, ls="--", lw=0.8, label="server model only")
        ax.set_xscale("log")
        ax.set_xticks([18, 37, 69, 149], ["18", "37", "69", "149"])
        ax.minorticks_off()
        ax.set_xlabel("Defects labelled in fleet")
        ax.set_ylabel(label)
    axes[0].set_ylim(0.1, 0.7)
    axes[1].set_ylim(0.7, 1.0)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncols=3, fontsize=5, bbox_to_anchor=(0.5, -0.01))
    fig.get_layout_engine().set(rect=(0, 0.13, 1, 0.87))
    panel_labels(axes)
    save(fig, str(HERE / "fig2_roundc"))


def fig_samples_2col():
    import json
    samples = json.loads((HERE / "samples.json").read_text())
    short = {"good, no defect": "good", "other defect (torn part or peeling paint)": "other defect",
             "lightning rod suspension": "lightning rod susp.", "glass insulator": "glass insulator", "vari-grip clamp": "vari-grip clamp"}
    fig, axes = plt.subplots(1, len(samples), figsize=(WIDE2, WIDE2 * 0.27), layout="constrained")
    for i, (ax, s) in enumerate(zip(axes, samples)):
        path = HERE / "samples" / f"sample{i}.jpg"
        if path.exists():
            ax.imshow(plt.imread(path))
        else:
            ax.imshow(np.full((10, 10, 3), 0.85))
            ax.text(5, 5, "image\n(exported on server)", ha="center", va="center", fontsize=5)
        ax.set_xticks([])
        ax.set_yticks([])
        a = s["answers"]
        truth = f"Truth: {short.get(s['condition'], s['condition'])}\n{short.get(s['component'], s['component'])}"
        answer = (f"component: {short.get(a['component'][0], a['component'][0])}\n  p = {a['component'][1]:.2f}\n"
                  f"defect: P(yes) = {a['defect']:.2f}\n"
                  f"condition: {short.get(a['condition'][0], a['condition'][0])}\n  p = {a['condition'][1]:.2f}")
        ax.set_title(truth, fontsize=5.5, loc="left", color="black" if s["correct"] else PALETTE["red"])
        ax.set_xlabel(answer, fontsize=5, loc="left", multialignment="left", color="black" if s["correct"] else PALETTE["red"])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(MUTED if s["correct"] else PALETTE["red"])
    fig.get_layout_engine().set(w_pad=0.08)
    save(fig, str(HERE / "fig2_samples"))


BASE_RUNS = ["fl_fedavg_k3", "fl_fedavg_k6", "fl_fedavg_k18", "fl_fleet_base_k5", "fl_b_head", "fl_b_lora8", "fl_b_last4", "fl_b_last4_int8",
             "fl_b_last4_prox", "fl_b_last4_r30", "fl_b_layers", "fl_b_full_prox", "fl_c_base_n25", "fl_c_base_n50", "fl_c_base_n100",
             "fl_c_base_n200", "fl_d_iid", "fl_d_component", "fl_d_component_prox", "fl_d_dirichlet"]
TUNED_FP16 = ["fl_fleet_warm_k5", "fl_c_warm_n25", "fl_c_warm_n50", "fl_c_warm_n100", "fl_c_warm_n200"]
TUNED_INT8 = ["fl_e1_full", "fl_e1_layers", "fl_e1_last4", "fl_e1_lora8", "fl_e1_head", "fl_e2_plain_n50", "fl_e2_plain_n200",
              "fl_e2_balance_n50", "fl_e2_balance_n200", "fl_e2_prox_n50", "fl_e2_prox_n200", "fl_e2_slr_n50", "fl_e2_slr_n200",
              "fl_e3_iid", "fl_e3_component", "fl_e3_component_prox", "fl_e3_dirichlet"]


def fig_uplink_all_2col():
    fig, axes = figure_grid(1, 2, width=WIDE2, ratio=0.27)
    groups = [("published checkpoint (fp16 or int8)", BASE_RUNS, PALETTE["blue"], "s"),
              ("fine-tuned start, fp16", TUNED_FP16, PALETTE["orange"], "^"),
              ("fine-tuned start, int8", TUNED_INT8, PALETTE["red"], "o")]
    for ax, part, key, ci, label in [(axes[0], "condition", "macro_f1", "condition_macro_f1", "Condition macro F1"),
                                     (axes[1], "defect_noul", "auroc", "defect_auroc", "Defect AUROC")]:
        for name, runs, color, marker in groups:
            for i, run in enumerate(runs):
                v, lo, hi = metric(run, part, key, ci)
                ax.errorbar([1000 * uplink_gb(run)], [v], yerr=[[v - lo], [hi - v]], marker=marker, color=color, capsize=1, lw=0.6,
                            markersize=2.8, alpha=0.9, label=name if i == 0 else None)
        for ref, text in [("good300_full", "central, all towers"), ("fl_server_central", "server model")]:
            y = summary(ref)["test"][part][key]
            ax.axhline(y, color=MUTED, ls="--", lw=0.7)
            ax.text(330, y, text, fontsize=5.5, color=MUTED, va="bottom")
        ax.set_xscale("log")
        ax.set_xlim(300, 60000)
        ax.set_xlabel("Total uplink of the run (MB)")
        ax.set_ylabel(label)
    axes[0].set_ylim(0.1, 0.75)
    axes[1].set_ylim(0.7, 1.0)
    axes[0].legend(fontsize=5.5, loc="lower right")
    panel_labels(axes)
    save(fig, str(HERE / "fig2_uplink_all"))


def fig_round_e3_2col():
    names = ["shackle", "vari-grip clamp", "glass insulator, spacer", "lightning rod, plate", "yoke susp., yoke", "polymer insul., dampers"]
    rows = [("server model (start)", "fl_server_central")]
    rows += [(f"alone: {names[i]}", f"fl_e3_component_local_d{i}") for i in range(6)]
    rows += [("FedAvg, component", "fl_e3_component"), ("same + FedProx", "fl_e3_component_prox"), ("FedAvg, random", "fl_e3_iid"),
             ("FedAvg, tower", "fl_e1_last4"), ("FedAvg, Dirichlet", "fl_e3_dirichlet"), ("FedAvg full, int8, tower", "fl_e1_full"),
             ("central, all towers", "good300_full")]
    conditions = ["good, no defect", "rust", "corrosion", "missing cap", "bird nest"]
    columns = ["good", "rust", "corrosion", "missing\ncap", "bird\nnest", "comp.\nacc."]
    values = np.array([[summary(run)["test"]["condition"]["per_class"][c]["recall"] for c in conditions] + [summary(run)["test"]["component"]["accuracy"]]
                       for _, run in rows])
    fig, ax = figure(width=COL2, ratio=0.86)
    image = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=5, color="white" if values[i, j] < 0.6 else "black")
    ax.set_xticks(range(len(columns)), [c.replace("\\n", "\n") for c in columns], fontsize=5.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=5.5)
    for y in (0.5, 6.5, 11.5):
        ax.axhline(y, color="white", lw=1.2)
    ax.grid(False)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02).set_label("Test recall / accuracy", fontsize=6)
    save(fig, str(HERE / "fig2_rounde3"))


if __name__ == "__main__":
    use_style()
    fig_finetune()
    fig_federated()
    fig_starts()
    fig_finetune_2col()
    fig_federated(WIDE2, "fig2_federated")
    fig_starts_2col()
    fig_round_b_2col()
    fig_round_c_defects_2col()
    fig_overview_2col()
    fig_round_d_2col()
    fig_samples_2col()
    fig_uplink_all_2col()
    fig_round_e3_2col()
