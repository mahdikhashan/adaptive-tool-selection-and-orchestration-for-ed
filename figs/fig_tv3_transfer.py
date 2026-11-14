import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import PALETTE, figure, save, use_style

SEL = Path(__file__).resolve().parent.parent / "results/tools_v3/select"
GROUPS = [("Qwen3-0.6B", "eval_newtools_500.json", "qwen3-0.6b sft", "qwen3-0.6b grpo"),
          ("Qwen3-1.7B, thinking", "eval_newtools_500_1.7b.json", "qwen3-1.7b sft_think", "qwen3-1.7b grpo_think")]


def acc(file, name):
    return json.loads((SEL / file).read_text())["methods"][name]["4.0"]["all_three_correct"]


use_style()
fig, ax = figure(width=252 / 72.27, ratio=0.6)
x = np.arange(len(GROUPS))
w = 0.32
sft = [acc(f, a) for _, f, a, _ in GROUPS]
grpo = [acc(f, b) for _, f, _, b in GROUPS]
ax.bar(x - w / 2, sft, w, color="#B8C7DA", label="SFT")
ax.bar(x + w / 2, grpo, w, color=PALETTE["blue"], label="SFT + GRPO")
for pos, vals in ((x - w / 2, sft), (x + w / 2, grpo)):
    for p, v in zip(pos, vals):
        ax.text(p, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
ax.set_xticks(x, [g[0] for g in GROUPS])
ax.set_ylabel("All three fields correct")
ax.set_ylim(0, 1.08)
ax.legend(loc="upper left", frameon=False, fontsize=6.5)
save(fig, str(Path(__file__).parent / "fig_tv3_transfer"))
