import hashlib
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

URL = "https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded"
SIZE = 6_400_493_732
SHA256 = "effd039e4b192088b17e9f6237bcd1b83ae1082a7eb5db89a4944dad0f489139"
FOLDERS = ("InsPLAD-det/", "supervised_fault_classification/", "unsupervised_anomaly_detection/")

DEST = Path(os.environ.get("INSPLAD_DIR", "data/insplad"))
ZIP = DEST.parent / "InsPLAD_Dataset.zip"


def is_extracted():
    return all((DEST / folder).is_dir() for folder in FOLDERS)


def download():
    if ZIP.exists() and ZIP.stat().st_size == SIZE:
        return
    subprocess.check_call(["curl", "-L", "--fail", "-C", "-", "-o", str(ZIP), URL])


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(name):
    for folder in FOLDERS:
        if folder in name:
            return name[name.index(folder):]
    return None


def extract():
    with zipfile.ZipFile(ZIP) as archive:
        for member in archive.infolist():
            path = relative_path(member.filename.replace("\\", "/"))
            if member.is_dir() or path is None:
                continue
            target = DEST / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if is_extracted():
        print(f"InsPLAD is already in {DEST}")
        return
    download()
    if sha256(ZIP) != SHA256:
        raise SystemExit(f"{ZIP} has a wrong SHA-256. Delete it and run again.")
    extract()
    ZIP.unlink()
    print(f"InsPLAD is in {DEST}")


if __name__ == "__main__":
    main()
