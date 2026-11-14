import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from orx_figstyle import PALETTE, save, use_style

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(sys.argv[1])
PICK = ["Fotos 27-11-2020_DJI_0484_cadeia_isolador_vidro_2321.jpg", "Fotos 21-10-2020_DJI_0536_cadeia_isolador_vidro_1613_1.jpg",
        "09-06-2021_DJI_0482_593_2.jpg"]
NAMES = {"good": "good", "missing-cap": "missing cap", "nest": "bird nest", "other-defect": "other defect"}
OK, BAD = "#222222", PALETTE["red"]

files = {f: os.path.join(root, f) for root, _, names in os.walk(DATA) for f in names}
rows = {os.path.basename(r["image"]): r for r in map(json.loads, open(ROOT / "results/tools_v3/images.jsonl")) if r["split"] == "test"}


def square(path):
    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((300, 300))


def best(probs):
    key = max(probs, key=probs.get)
    return key, probs[key]


use_style()
width = 252 / 72.27
fig, axes = plt.subplots(len(PICK), 2, figsize=(width, 3.45), gridspec_kw={"width_ratios": [1, 2.05], "hspace": 0.1, "wspace": 0.04})
fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01)
axes[0, 0].set_title("(a) Image only", fontsize=7, pad=4)
axes[0, 1].set_title("(b) With true labels and tool answers", fontsize=7, pad=4)
for line, name in enumerate(PICK):
    row = rows[name]
    truth = row["condition"].split(",")[0].split(" (")[0]
    left, right = axes[line]
    left.imshow(square(files[name]), interpolation="none")
    left.set_xticks([])
    left.set_yticks([])
    for spine in left.spines.values():
        spine.set_linewidth(0.6)
    right.axis("off")
    comp, comp_p = best(row["out"]["laya"]["component"])
    cond, cond_p = best(row["out"]["laya"]["condition"])
    res, res_p = best(row["out"]["resnet"]["condition"])
    cond, res = NAMES.get(cond, cond), NAMES.get(res, res)
    lines = [("True labels:", OK, "bold"), (f"  {row['component']}, {truth}", OK, "normal"),
             ("laya-vision:", OK, "bold"),
             (f"  component: {comp} ({100 * comp_p:.0f}%)", OK if comp == row["component"] else BAD, "normal"),
             (f"  condition: {cond} ({100 * cond_p:.0f}%)", OK if cond == truth else BAD, "normal"),
             ("ResNet-50:", OK, "bold"),
             (f"  condition: {res} ({100 * res_p:.0f}%)", OK if res == truth else BAD, "normal")]
    for k, (text, color, weight) in enumerate(lines):
        right.text(0.03, 0.95 - 0.135 * k, text, transform=right.transAxes, fontsize=6.3, color=color, fontweight=weight, va="top")
fig.savefig(Path(__file__).parent / "fig_tv3_samples.png", dpi=150)
save(fig, str(Path(__file__).parent / "fig_tv3_samples"), formats=("pdf",))
