import hashlib
import os
import shutil
import subprocess
import unicodedata
import zipfile
from pathlib import Path

URL = "https://data.mendeley.com/public-files/datasets/5n3fjgvfyz/files/96707044-99bb-40b2-bf23-6fa1b41ab9b0/file_downloaded"
SIZE = 6_400_493_732
SHA256 = "effd039e4b192088b17e9f6237bcd1b83ae1082a7eb5db89a4944dad0f489139"
FOLDERS = ("InsPLAD-det", "supervised_fault_classification", "unsupervised_anomaly_detection")

DEST = Path(os.environ.get("INSPLAD_DIR", "data/insplad"))
ZIP = DEST.parent / "InsPLAD_Dataset.zip"


def file_count(folder):
    path = DEST / folder
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.is_dir() else 0


def is_extracted():
    return all(file_count(folder) > 0 for folder in FOLDERS)


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


def member_name(member):
    name = member.filename
    if not member.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("utf-8")
        except UnicodeError:
            pass
    return unicodedata.normalize("NFC", name.replace("\\", "/"))


def target_path(folder, name):
    marker = folder + "/"
    return name[name.index(marker):] if marker in name else f"{folder}/{name}"


def is_metadata(name):
    return name.startswith("__MACOSX/") or "/._" in name or name.startswith("._")


def extract_inner(outer, folder):
    inner_zip = DEST.parent / f"{folder}.zip"
    with outer.open(f"{folder}.zip") as source, inner_zip.open("wb") as sink:
        shutil.copyfileobj(source, sink, 1 << 20)
    with zipfile.ZipFile(inner_zip) as archive:
        for member in archive.infolist():
            name = member_name(member)
            if member.is_dir() or is_metadata(name):
                continue
            target = DEST / target_path(folder, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink)
    inner_zip.unlink()
    print(f"{folder}: {file_count(folder)} files")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if is_extracted():
        print(f"InsPLAD is already in {DEST}: " + ", ".join(f"{f} {file_count(f)}" for f in FOLDERS))
        return
    download()
    if sha256(ZIP) != SHA256:
        raise SystemExit(f"{ZIP} has a wrong SHA-256. Delete it and run again.")
    with zipfile.ZipFile(ZIP) as outer:
        for folder in FOLDERS:
            extract_inner(outer, folder)
    if not is_extracted():
        raise SystemExit(f"Extraction is not complete. The zip stays in {ZIP}.")
    ZIP.unlink()
    print(f"InsPLAD is in {DEST}")


if __name__ == "__main__":
    main()
