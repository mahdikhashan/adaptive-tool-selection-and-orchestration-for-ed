import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import BASELINE, PALETTE, figure, save, use_style

SEL = Path(__file__).resolve().parent.parent / "results/tools_v3/select"
SIZES = [("0.6b", "eval_newtools_500.json"), ("1.7b", "eval_newtools_500_1.7b.json"), ("4b", "eval_newtools_500_4b.json")]
SERIES = [("SFT", "sft", MUTED := "#B8C7DA"), ("SFT + GRPO", "grpo", PALETTE["blue"]),
          ("SFT, thinking", "sft_think", "#E8B48A"), ("SFT + GRPO, thinking", "grpo_think", PALETTE["orange"])]


def acc(file, name):
    path = SEL / file
    if not path.exists():
        return None
    res = json.loads(path.read_text())["methods"].get(name, {}).get("4.0")
    return res["all_three_correct"] if isinstance(res, dict) and "all_three_correct" in res else None


retrained = json.loads((SEL / "eval_nolaya.json").read_text())["methods"]["qwen3-0.6b_nolaya sft"]["4.0"]["all_three_correct"]

use_style()
fig, ax = figure(width=252 / 72.27, ratio=0.62)
x = np.arange(len(SIZES))
w = 0.2
for k, (label, method, color) in enumerate(SERIES):
    pos = x + (k - 1.5) * w
    vals = [acc(f, f"qwen3-{s} {method}") for s, f in SIZES]
    ax.bar([p for p, v in zip(pos, vals) if v is not None], [v for v in vals if v is not None], w, color=color, label=label)
    for p, v, (s, _) in zip(pos, vals, SIZES):
        if v is None and not (method in ("sft", "grpo") and s != "0.6b"):
            ax.text(p, 0.02, "running", ha="center", va="bottom", fontsize=5, rotation=90, color=BASELINE)
ax.axhline(retrained, color=BASELINE, linestyle="--", linewidth=0.8)
ax.text(-0.45, retrained + 0.015, "retrained without laya (0.959)", ha="left", va="bottom", fontsize=5.5, color=BASELINE, backgroundcolor="white")
ax.set_xticks(x, ["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B"])
ax.set_ylabel("All three fields correct")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=6)
save(fig, str(Path(__file__).parent / "fig_tv3_transfer"))
