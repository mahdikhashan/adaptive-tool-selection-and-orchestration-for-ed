#!/usr/bin/env python3
"""Explicit checkpoint downloads only: no training, inference or quantization."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--only", choices=["all", "yolo", "laya", "smol", "qwen"], default="all")
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {}
    if args.only in ("all", "yolo"):
        from ultralytics import YOLO
        original_directory = Path.cwd()
        try:
            os.chdir(output)
            YOLO("yolo26n-cls.pt", task="classify")
        finally:
            os.chdir(original_directory)
        record["yolo_pretrained_imagenet_classifier"] = str(output / "yolo26n-cls.pt")
        record["yolo_finetuning_required"] = False
    if args.only in ("all", "laya", "smol", "qwen"):
        from huggingface_hub import HfApi, snapshot_download
        for name, repository, revision in (
            ("laya", "thaitea/laya-vision", "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"),
            ("smol", "HuggingFaceTB/SmolVLM-256M-Instruct", "main"),
            ("qwen", "Qwen/Qwen3-VL-8B-Instruct", "main"),
        ):
            if args.only not in ("all", name):
                continue
            resolved = HfApi().model_info(repository, revision=revision).sha
            destination = output / {"laya": "laya_base", "smol": "smol_base", "qwen": "qwen8b"}[name]
            path = snapshot_download(repository, revision=resolved, local_dir=destination)
            record[name] = {"path": path, "repository": repository, "revision": resolved}
    (output / f"downloads-{args.only}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print("Supply your EXISTING fine-tuned Smol adapter separately; this command does not train it.")


if __name__ == "__main__":
    main()
