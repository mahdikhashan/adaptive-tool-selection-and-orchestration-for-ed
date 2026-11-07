import json
from pathlib import Path

from PIL import Image

from fedorch.items import INSPLAD

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figs" / "samples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(json.loads((ROOT / "figs" / "samples.json").read_text())):
        image = Image.open(INSPLAD / sample["image"]).convert("RGB")
        image.thumbnail((320, 320))
        path = OUT / f"sample{i}.jpg"
        image.save(path, quality=85)
        print(path, image.size)


if __name__ == "__main__":
    main()
