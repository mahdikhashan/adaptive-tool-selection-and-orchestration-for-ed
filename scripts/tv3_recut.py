import argparse
import json
import multiprocessing
import re
from collections import defaultdict

import cv2
import numpy as np

from fedorch.items import INSPLAD
from fedorch.common import results_dir
from fedorch.tools_v3 import Progress, held_images, image_file, recut_file

SCALES = (1.25, 1.0, 0.8, 0.65, 0.5)


def frame_index():
    index = defaultdict(list)
    for path in (INSPLAD / "InsPLAD-det").rglob("*"):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            number = re.search(r"DJI_(\d+)", path.name)
            if number:
                index[number.group(1)].append(path)
    return index


def candidates(index, name):
    number = re.search(r"DJI_(\d+)", name)
    if not number:
        return []
    frames = index.get(number.group(1), [])
    date = re.search(r"(\d{2}-\d{2}-\d{4})", name)
    if date:
        dated = [f for f in frames if date.group(1) in str(f)]
        frames = dated or frames
    return frames


def match(args):
    row, frames, shrink, margin, threshold = args
    crop = cv2.imread(str(image_file(row)))
    if crop is None:
        return {"image": row["image"], "ok": False, "reason": "crop unreadable"}
    best = None
    for frame_path in frames:
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        small = cv2.resize(frame, None, fx=shrink, fy=shrink, interpolation=cv2.INTER_AREA)
        for scale in SCALES:
            w, h = int(crop.shape[1] * shrink * scale), int(crop.shape[0] * shrink * scale)
            if w < 8 or h < 8 or w >= small.shape[1] or h >= small.shape[0]:
                continue
            result = cv2.matchTemplate(small, cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA), cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(result)
            if best is None or score > best[0]:
                best = (score, frame_path, frame, loc[0] / shrink, loc[1] / shrink, w / shrink, h / shrink)
    if best is None:
        return {"image": row["image"], "ok": False, "reason": "no frame"}
    score, frame_path, frame, x, y, w, h = best
    out = {"image": row["image"], "frame": str(frame_path.relative_to(INSPLAD)), "score": round(float(score), 4),
           "box": [round(v) for v in (x, y, w, h)], "ok": bool(score >= threshold)}
    if out["ok"]:
        cx, cy, half_w, half_h = x + w / 2, y + h / 2, w * margin / 2, h * margin / 2
        x0, y0 = int(max(0, cx - half_w)), int(max(0, cy - half_h))
        x1, y1 = int(min(frame.shape[1], cx + half_w)), int(min(frame.shape[0], cy + half_h))
        recut = frame[y0:y1, x0:x1]
        side = max(recut.shape[:2])
        if side > 1024:
            recut = cv2.resize(recut, None, fx=1024 / side, fy=1024 / side, interpolation=cv2.INTER_AREA)
        target = recut_file(row)
        target.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(target), recut)
        out["recut_size"] = list(recut.shape[:2])
    return out


def main():
    parser = argparse.ArgumentParser(description="Match each val/test crop to its full InsPLAD-det photo and cut it again at full resolution with context.")
    parser.add_argument("--shrink", type=float, default=0.25)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    out = results_dir("tools_v3") / "recut_matches.jsonl"
    done = {json.loads(l)["image"] for l in out.open()} if out.exists() else set()
    rows = [r for r in held_images() if r["image"] not in done]
    rows = rows[: args.limit] if args.limit else rows
    index = frame_index()
    print(f"full photos indexed: {sum(len(v) for v in index.values())}, crops to match: {len(rows)}", flush=True)
    jobs = [(r, candidates(index, r["image"].split("/")[-1]), args.shrink, args.margin, args.threshold) for r in rows]
    progress, ok = Progress("recut", len(jobs), 200), 0
    with multiprocessing.Pool(args.workers) as pool, out.open("a") as file:
        for result in pool.imap_unordered(match, jobs, chunksize=4):
            file.write(json.dumps(result) + "\n")
            ok += result["ok"]
            progress.tick()
    total = len(done) + len(rows)
    matched = sum(json.loads(l)["ok"] for l in out.open())
    print(f"recut done: {matched}/{total} crops matched ({100 * matched / max(1, total):.1f}%)", flush=True)


if __name__ == "__main__":
    main()
