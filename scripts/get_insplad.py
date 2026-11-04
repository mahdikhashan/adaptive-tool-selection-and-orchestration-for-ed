#!/usr/bin/env python
"""Download InsPLAD from Mendeley Data and extract only the crops ElecVQA-Bench uses.

ElecVQA-Bench's tower splits reference only supervised_fault_classification/ and
unsupervised_anomaly_detection/ (35,415 unique crops), never the full-size InsPLAD-det photos.
Standard library only. Resumable download, SHA-256 checked.

    python scripts/get_insplad.py --dest "$INSPLAD_DIR" [--keep-zip] [--all-folders]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

URL = ("https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/"
       "96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded")
SIZE = 6_400_493_732
SHA256 = "effd039e4b192088b17e9f6237bcd1b83ae1082a7eb5db89a4944dad0f489139"
NEEDED = ("supervised_fault_classification/", "unsupervised_anomaly_detection/")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def download(zpath: Path) -> None:
    if zpath.exists() and zpath.stat().st_size == SIZE:
        print(f"zip already complete: {zpath}")
        return
    if shutil.which("curl") is None:
        sys.exit("curl not found")
    print(f"downloading {SIZE / 1e9:.1f} GB to {zpath} (resumable) ...")
    subprocess.check_call(["curl", "-L", "--fail", "-C", "-", "-o", str(zpath), URL])
    if zpath.stat().st_size != SIZE:
        sys.exit(f"size mismatch: {zpath.stat().st_size} != {SIZE}; rerun to resume")


def extract(zpath: Path, dest: Path, all_folders: bool) -> int:
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = info.filename.replace("\\", "/")
            if info.is_dir():
                continue
            # Strip any wrapper folders so paths start at the InsPLAD subset folder.
            hit = next((name.index(k) for k in NEEDED if k in name), None)
            if hit is None:
                if not all_folders:
                    continue
                rel = name
            else:
                rel = name[hit:]
            out = dest / rel
            if out.exists() and out.stat().st_size == info.file_size:
                n += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=Path(os.environ.get("INSPLAD_DIR", "data/raw")))
    ap.add_argument("--zip", type=Path, default=None, help="where to keep the zip (default: next to dest)")
    ap.add_argument("--keep-zip", action="store_true")
    ap.add_argument("--all-folders", action="store_true", help="also extract InsPLAD-det (full-size photos)")
    ap.add_argument("--skip-sha", action="store_true")
    a = ap.parse_args()
    a.dest.mkdir(parents=True, exist_ok=True)
    zpath = a.zip or a.dest.parent / "InsPLAD_Dataset.zip"
    print(f"free space at {a.dest}: {shutil.disk_usage(a.dest).free / 1e9:.1f} GB")
    download(zpath)
    if not a.skip_sha:
        print("checking SHA-256 ...")
        got = sha256(zpath)
        if got != SHA256:
            sys.exit(f"SHA-256 mismatch: {got}")
    n = extract(zpath, a.dest, a.all_folders)
    print(f"extracted {n} files into {a.dest}")
    for k in NEEDED:
        print(f"  {k:36s} {'present' if (a.dest / k).exists() else 'MISSING'}")
    if not a.keep_zip:
        zpath.unlink()
        print(f"deleted {zpath}")


if __name__ == "__main__":
    main()
