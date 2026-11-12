import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import BASELINE, PALETTE, figure, save, use_style

ROOT = Path(__file__).resolve().parent.parent
ev = json.loads((ROOT / "results/tools_v3/select/eval.json").read_text())["methods"]
ev.update({k: v for k, v in json.loads((ROOT / "results/tools_v3/select/eval_nolaya.json").read_text())["methods"].items() if "_nolaya" in k})
single = json.loads((ROOT / "results/tools_v3/value_report.json").read_text())["single_tool"]
fields = [("component_accuracy", "Component"), ("defect_accuracy", "Defect"), ("condition_accuracy", "Condition")]
bars = [("Agent, no training", ev["qwen3-0.6b base"]["4.0"], BASELINE),
        ("Agent after SFT, with laya-vision", ev["qwen3-0.6b sft"]["4.0"], PALETTE["blue"]),
        ("Agent after SFT, without laya-vision", ev["qwen3-0.6b_nolaya sft"]["4.0"], PALETTE["purple"]),
        ("laya-vision alone", single["laya"], PALETTE["orange"]),
        ("ConvNeXt re-cut alone", single["convnext_recut"], PALETTE["green"])]

use_style()
fig, ax = figure(width=252 / 72.27, ratio=0.62)
x = np.arange(len(fields))
w = 0.16
for k, (name, res, color) in enumerate(bars):
    vals = [res.get(f, np.nan) for f, _ in fields]
    pos = x + (k - (len(bars) - 1) / 2) * w
    ax.bar(pos, vals, w, color=color, label=name)
    for p, v in zip(pos, vals):
        if np.isnan(v):
            ax.text(p, 0.02, "n/a", ha="center", va="bottom", fontsize=5, rotation=90, color=BASELINE)
ax.set_xticks(x, [lab for _, lab in fields])
ax.set_ylabel("Accuracy on test images")
ax.set_ylim(0, 1.0)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False, fontsize=6)
save(fig, str(Path(__file__).parent / "fig_tv3_accuracy"))
