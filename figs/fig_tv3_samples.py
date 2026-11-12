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
PICK = ["Fotos 27-11-2020_DJI_0484_cadeia_isolador_vidro_2321.jpg", "01-06-2021_DJI_0263_77_2.jpg",
        "Fotos 21-10-2020_DJI_0536_cadeia_isolador_vidro_1613_1.jpg", "09-06-2021_DJI_0482_593_2.jpg"]
NAMES = {"good": "good", "missing-cap": "missing cap", "nest": "bird nest", "other-defect": "other defect"}
OK, BAD = "#222222", PALETTE["red"]

files = {f: os.path.join(root, f) for root, _, names in os.walk(DATA) for f in names}
rows = {os.path.basename(r["image"]): r for r in map(json.loads, open(ROOT / "results/tools_v3/images.jsonl")) if r["split"] == "test"}


def square(path):
    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left, top = (image.width - side) // 2, (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize((320, 320))


def best(probs):
    key = max(probs, key=probs.get)
    return key, probs[key]


use_style()
fig, axes = plt.subplots(2, 4, figsize=(522 / 72.27, 3.95), gridspec_kw={"hspace": 0.06, "wspace": 0.05})
fig.subplots_adjust(left=0.035, right=0.995, top=0.995, bottom=0.21)
for col, name in enumerate(PICK):
    row, image = rows[name], square(files[name])
    truth = row["condition"].split(",")[0].split(" (")[0]
    for line in (0, 1):
        ax = axes[line, col]
        ax.imshow(image, interpolation="none")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    comp, comp_p = best(row["out"]["laya"]["component"])
    cond, cond_p = best(row["out"]["laya"]["condition"])
    res, res_p = best(row["out"]["resnet"]["condition"])
    cond, res = NAMES.get(cond, cond), NAMES.get(res, res)
    lines = [(f"True: {row['component']}, {truth}", OK, "bold"),
             (f"laya-vision: {comp} {100 * comp_p:.0f}%,", OK if comp == row["component"] else BAD, "normal"),
             (f"    {cond} {100 * cond_p:.0f}%", OK if cond == truth else BAD, "normal"),
             (f"ResNet-50: {res} {100 * res_p:.0f}%", OK if res == truth else BAD, "normal")]
    for k, (text, color, weight) in enumerate(lines):
        axes[1, col].text(0.0, -0.06 - 0.115 * k, text, transform=axes[1, col].transAxes, fontsize=5.6, color=color, fontweight=weight, va="top")
for line, label in enumerate(["(a) image only", "(b) with tool answers"]):
    axes[line, 0].set_ylabel(label, fontsize=7)
fig.savefig(Path(__file__).parent / "fig_tv3_samples.png", dpi=130)
save(fig, str(Path(__file__).parent / "fig_tv3_samples"), formats=("pdf",))
