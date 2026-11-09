import json
from pathlib import Path

import numpy as np

from orx_figstyle import BASELINE, MUTED, PALETTE, figure_grid, panel_labels, save, use_style

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
WIDE2 = 522 / 72.27
LABELS = ["laya-vision\nbase", "laya-vision\nfine-tuned", "SmolVLM\nbase", "SmolVLM\nLoRA"]
COLORS = [PALETTE["blue"], PALETTE["blue"], PALETTE["orange"], PALETTE["orange"]]


def table():
    return json.loads((RESULTS / "quant" / "q4_table.json").read_text())


def bars(ax, values, colors=COLORS, low=None, high=None, fmt="{:.0f}"):
    x = np.arange(len(values))
    tuned = [i % 2 == 1 for i in range(len(values))]
    ax.bar(x, values, color=colors, width=0.65)
    tops = values if high is None else high
    for i, (bar_x, v, top) in enumerate(zip(x, values, tops)):
        ax.patches[i].set_alpha(1.0 if tuned[i] else 0.55)
        ax.text(bar_x, top, fmt.format(v), ha="center", va="bottom", fontsize=5.5)
    if low is not None:
        ax.errorbar(x, values, yerr=[np.array(values) - low, np.array(high) - values], fmt="none", ecolor=BASELINE, capsize=1.5, lw=0.7)
    ax.set_xticks(x, LABELS, fontsize=5.5)


def fig_overview():
    rows = table()
    fig, axes = figure_grid(1, 3, width=WIDE2, ratio=0.3)
    x = np.arange(len(rows))
    w = 0.38
    for off, key, name, alpha in [(-w / 2, "stored_mb", "size on disk", 1.0), (w / 2, "peak_gpu_mb", "peak GPU memory", 0.5)]:
        values = [r[key] for r in rows]
        axes[0].bar(x + off, values, w, color=COLORS, alpha=alpha, label=name)
        for xi, v in zip(x + off, values):
            axes[0].text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=5)
    axes[0].set_xticks(x, LABELS, fontsize=5.5)
    axes[0].set_ylabel("MB")
    axes[0].legend(fontsize=5.5, loc="upper left")
    axes[0].set_ylim(0, 450)
    acc = [r["item_accuracy"] for r in rows]
    low = [r.get("item_accuracy_95ci", [a, a])[0] for r, a in zip(rows, acc)]
    high = [r.get("item_accuracy_95ci", [a, a])[1] for r, a in zip(rows, acc)]
    bars(axes[1], acc, low=low, high=high, fmt="{:.3f}")
    axes[1].set_ylim(0.75, 1.0)
    axes[1].set_ylabel("Test item accuracy")
    med = [r["latency_median_ms"] for r in rows]
    bars(axes[2], med, low=med, high=[r["latency_p95_ms"] for r in rows], fmt="{:.0f}")
    axes[2].set_ylim(0, 140)
    axes[2].set_ylabel("Latency (ms, median; whisker p95)")
    panel_labels(axes)
    save(fig, str(HERE / "fig_q4_overview"))


def fig_details():
    fig, axes = figure_grid(1, 2, width=WIDE2, ratio=0.3)
    metrics = [("condition", "macro_f1", "condition_macro_f1", "Condition\nmacro F1"), ("component", "accuracy", None, "Component\naccuracy"),
               ("defect_noul", "auroc", "defect_auroc", "Defect\nAUROC")]
    x = np.arange(len(metrics))
    for off, model, name, alpha in [(-0.19, "base", "base", 0.55), (0.19, "good300_full", "fine-tuned", 1.0)]:
        s = json.loads((RESULTS / "vision" / f"q4_laya_{model}" / "summary.json").read_text())
        v = [s["test"][p][k] for p, k, _, _ in metrics]
        ci = [s["test_bootstrap_95ci"][c] if c else [a, a] for (_, _, c, _), a in zip(metrics, v)]
        axes[0].bar(x + off, v, 0.38, color=PALETTE["blue"], alpha=alpha, label=name)
        axes[0].errorbar(x + off, v, yerr=[[a - c[0] for a, c in zip(v, ci)], [c[1] - a for a, c in zip(v, ci)]], fmt="none",
                         ecolor=BASELINE, capsize=1.5, lw=0.7)
        for xi, a in zip(x + off, v):
            axes[0].text(xi, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=5)
    axes[0].set_xticks(x, [m[3] for m in metrics], fontsize=5.5)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("laya-vision, 4-bit, test towers")
    axes[0].legend(fontsize=5.5, loc="upper left", ncols=2)
    groups = ["truth", "SmolVLM base", "SmolVLM LoRA"]
    shares = []
    for m in ("base", "lora"):
        rows = [json.loads(line) for line in (RESULTS / "quant" / f"q4_smolvlm_{m}" / "predictions.jsonl").open()]
        if not shares:
            shares.append(np.mean([r["label"] == "good" for r in rows]))
        shares.append(np.mean([r["prediction"] == "good" for r in rows]))
    axes[1].bar(range(3), shares, 0.6, color=[MUTED, PALETTE["orange"], PALETTE["orange"]], alpha=1.0)
    axes[1].patches[1].set_alpha(0.55)
    for i, v in enumerate(shares):
        axes[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=5.5)
    axes[1].set_xticks(range(3), groups, fontsize=5.5)
    axes[1].set_ylim(0.7, 1.03)
    axes[1].set_ylabel('Share of test items answered "good"')
    panel_labels(axes)
    save(fig, str(HERE / "fig_q4_details"))


if __name__ == "__main__":
    use_style()
    fig_overview()
    fig_details()
