import argparse
import collections
import io
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from fnmatch import fnmatch
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedorch.labels import item_image, item_label

DATA = Path(os.environ.get("FEDORCH_DATA", ROOT / "data"))
INSPLAD = Path(os.environ.get("INSPLAD_DIR", DATA / "insplad"))
ELECVQA = DATA / "elecvqa-bench"
LTE = DATA / "lte4g"
OUT = ROOT / "results" / "step1"

ELECVQA_REPO = "https://github.com/GhostDino/elecvqa-bench.git"
LTE_ZIP = "https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip"

MODELS = {
    "sdmlai/nano-jev": "controller small",
    "convaiinnovations/laya": "controller medium",
    "jaredpalmer/kev-4b": "controller large (adapter)",
    "Qwen/Qwen3.5-4B-Base": "controller large (base)",
    "perplexity-ai/pplx-decider-flash-v1.2": "controller multimodal",
    "HuggingFaceTB/SmolVLM-256M-Instruct": "tool onboard VLM",
    "Qwen/Qwen3-VL-8B-Instruct": "tool cloud VLM",
}
FALLBACKS = ["microsoft/MiniLM-L12-H384-uncased", "answerdotai/ModernBERT-large", "fastino/GLiNER2.5-Decide"]
SKIP_FILES = ["*.onnx", "onnx/*", "*.onnx_data", "*.h5", "*.msgpack", "*.gguf", "*.tflite", "*.ot", "coreml/*"]
OTHER_GB = {"insplad zip + files": 13.0, "python env": 10.0, "outputs": 20.0}

STATE = ("Intent: detect rust or corrosion on clamps. Corrosion is critical. Abstain is allowed. "
         "Evidence: classifier says rust p=0.58, good p=0.31. Crop 212x180 px. Blur 0.21.")
OPTIONS = {
    "accept_classifier": "accept the onboard classifier label",
    "interpret_local": "run SmolVLM-256M on the drone",
    "interpret_cloud": "send the image to the cloud VLM",
    "abstain": "report cannot judge",
}
QUESTION = "Which action must the drone do for this frame?"


def env():
    import torch
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": platform.node(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "hf_home": os.environ.get("HF_HOME", "default"),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "gpus": [],
    }
    for package in ["transformers", "peft", "flwr", "huggingface-hub", "laya", "nano-jev"]:
        info[package] = version(package)
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        info["gpus"].append({
            "index": index,
            "name": props.name,
            "memory_gb": round(props.total_memory / 1e9, 1),
            "free_gb": round(torch.cuda.mem_get_info(index)[0] / 1e9, 1),
            "capability": f"{props.major}.{props.minor}",
            "bf16": props.major >= 8,
            "kernel_ok": kernel_ok(index),
        })
    return info


def kernel_ok(index):
    import torch
    try:
        x = torch.ones(256, 256, device=f"cuda:{index}")
        return (x @ x).sum().item() == 256 ** 3
    except RuntimeError as error:
        return repr(error)


def repo_gb(api, model):
    files = api.list_repo_tree(model, recursive=True)
    size = sum(f.size for f in files if getattr(f, "size", None) and not any(fnmatch(f.path, p) for p in SKIP_FILES))
    return round(size / 1e9, 2)


def disk():
    from huggingface_hub import HfApi
    api = HfApi()
    models_gb = {model: repo_gb(api, model) for model in MODELS}
    need = round(sum(models_gb.values()) + sum(OTHER_GB.values()), 1)
    free = round(shutil.disk_usage(DATA if DATA.exists() else ROOT).free / 1e9, 1)
    return {"models_gb": models_gb, "other_gb": OTHER_GB, "need_gb": need, "free_gb": free, "enough": free > need}


def models():
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    found = {}
    for model, role in MODELS.items():
        info = api.model_info(model)
        path = snapshot_download(model, revision=info.sha, ignore_patterns=SKIP_FILES)
        card = info.card_data.to_dict() if info.card_data else {}
        found[model] = {"role": role, "sha": info.sha, "license": card.get("license"), "path": path}
    for model in FALLBACKS:
        found[model] = {"role": "fallback", "sha": api.model_info(model).sha}
    return found


def median_ms(fn, runs=30):
    fn()
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return round(statistics.median(times), 2)


def smoke():
    import laya
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("sdmlai/nano-jev", revision="v1.0")
    model = AutoModelForSequenceClassification.from_pretrained("sdmlai/nano-jev", revision="v1.0").to(device).eval()
    temperature = list(json.load(open(hf_hub_download("sdmlai/nano-jev", "calibration.json", revision="v1.0"))).values())[0]
    batch = tokenizer([f"question: {QUESTION} option: {name}: {text}" for name, text in OPTIONS.items()],
                      [STATE] * len(OPTIONS), truncation="only_second", max_length=512,
                      padding=True, return_tensors="pt").to(device)

    def nano_jev():
        with torch.no_grad():
            return model(**batch).logits.squeeze(-1)

    probabilities = torch.softmax(nano_jev() / temperature, dim=0).tolist()

    agent = laya.load("convaiinnovations/laya")
    questions = {"action": {"type": "choice", "instructions": QUESTION, "criteria": OPTIONS}}

    return {
        "device": device,
        "nano_jev": {"probabilities": dict(zip(OPTIONS, probabilities)), "median_ms": median_ms(nano_jev)},
        "laya": {"answer": agent.predict(STATE, questions)["answers"]["action"],
                 "median_ms": median_ms(lambda: agent.predict(STATE, questions), runs=20)},
        "note": "zero-shot, not fine-tuned",
    }


def split_stats(path):
    towers = collections.defaultdict(collections.Counter)
    for line in path.open():
        item = json.loads(line)
        towers[item["_tower"]][item_label(item)] += 1
    labels = sum(towers.values(), collections.Counter())
    sizes = sorted(sum(c.values()) for c in towers.values())
    return {"items": sum(sizes), "towers": len(towers), "tower_size_min_median_max": [sizes[0], sizes[len(sizes) // 2], sizes[-1]],
            "labels": dict(labels.most_common()), "per_tower": {t: dict(c) for t, c in sorted(towers.items())}}


def image_coverage(path):
    paths = {item_image(json.loads(line)) for line in path.open()}
    missing = [p for p in paths if not (INSPLAD / p).exists()]
    return {"images": len(paths), "missing": len(missing), "missing_examples": missing[:5]}


def data():
    if not ELECVQA.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", ELECVQA_REPO, str(ELECVQA)])
    splits = ELECVQA / "data/benchmark/split_comparison/tower14"
    tasks = ELECVQA / "data/benchmark/tasks"
    return {
        "elecvqa_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ELECVQA, capture_output=True, text=True).stdout.strip(),
        "insplad_dir": str(INSPLAD),
        "tower14": {split: split_stats(splits / f"{split}.jsonl") for split in ["train", "val", "test"]},
        "images": {task.stem: image_coverage(task) for task in sorted(tasks.glob("*.jsonl"))},
    }


def throughput_mbps(log):
    rows = [line.split() for line in log.read_text().splitlines()]
    return [int(r[4]) * 8 / int(r[5]) / 1000 for r in rows if len(r) == 6 and int(r[5]) > 0]


def trace():
    if not any(LTE.glob("**/*.log")):
        LTE.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(LTE_ZIP, timeout=120) as response:
            zipfile.ZipFile(io.BytesIO(response.read())).extractall(LTE)
    traces = {}
    for log in sorted(LTE.glob("**/*.log")):
        mbps = sorted(throughput_mbps(log))
        traces[log.stem] = {"seconds": len(mbps), "mean_mbps": round(statistics.mean(mbps), 2),
                            "p10_mbps": round(mbps[len(mbps) // 10], 2),
                            "below_2mbps": round(sum(m < 2 for m in mbps) / len(mbps), 3)}
    return {"traces": traces}


CHECKS = {"env": env, "disk": disk, "models": models, "smoke": smoke, "data": data, "trace": trace}


def run(name):
    start = time.time()
    try:
        result = {"ok": True, **CHECKS[name]()}
    except Exception as error:
        result = {"ok": False, "error": repr(error), "traceback": traceback.format_exc()}
    result["seconds"] = round(time.time() - start, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(f"{name}: {'ok' if result['ok'] else 'FAILED ' + result['error']} ({result['seconds']} s)")


def main():
    parser = argparse.ArgumentParser(description="Step 1: check the environment, models, data and network traces.")
    parser.add_argument("checks", nargs="*", help=f"checks to run: {', '.join(CHECKS)} (default: all)")
    checks = parser.parse_args().checks or list(CHECKS)
    unknown = [name for name in checks if name not in CHECKS]
    if unknown:
        parser.error(f"unknown checks: {', '.join(unknown)}")
    for name in checks:
        run(name)


if __name__ == "__main__":
    main()
