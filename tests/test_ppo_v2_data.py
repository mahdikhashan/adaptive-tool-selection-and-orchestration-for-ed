import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fedorch.ppo_v2_data import prepare, select_training


def items():
    rows = []
    for split, towers in (("train", 6), ("val", 2), ("test", 2)):
        for tower in range(towers):
            for source in range(20):
                for question in range(2):
                    rows.append({"item_id": f"{split}-{tower}-{source}-{question}", "split": split,
                                 "tower": f"{split}-{tower}", "image": f"{split}/{tower}/rust/{source}.jpg",
                                 "task": "T2" if question == 0 else "T3", "label": "rust"})
    return rows


class DataTests(unittest.TestCase):
    def test_source_groups_and_fixed_audits(self):
        rows = items()
        selected = select_training([r for r in rows if r["split"] == "train"], .1, 0)
        self.assertEqual(len(selected), 24)
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            prepare(rows, a, tmp, .1, check_images=False)
            prepare(rows, b, tmp, .1, check_images=False)
            self.assertEqual((a / "audits.json").read_bytes(), (b / "audits.json").read_bytes())
            public = [json.loads(r) for r in (a / "items.jsonl").read_text().splitlines()]
            self.assertTrue(all("label" not in r for r in public))
            train = [r for r in public if r["split"] == "train"]
            self.assertEqual(len(train), 2 * len({r["source_id"] for r in train}))
            with self.assertRaises(ValueError):
                prepare(rows, a, tmp, .1, check_images=False)

    def test_split_leakage_rejected(self):
        rows = items()
        rows[-1]["image"] = rows[0]["image"]
        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(ValueError, "crosses"):
            prepare(rows, Path(tmp) / "a", tmp, check_images=False)


if __name__ == "__main__":
    unittest.main()
