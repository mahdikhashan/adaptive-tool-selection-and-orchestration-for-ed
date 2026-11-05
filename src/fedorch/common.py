import json
import os
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("FEDORCH_WORK", ROOT))


def runs_dir(step):
    path = WORK / "runs" / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(step):
    path = ROOT / "results" / step
    path.mkdir(parents=True, exist_ok=True)
    return path


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gpu_dtype():
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def write_env(step):
    env = {
        "host": platform.node(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_dtype": str(gpu_dtype()),
    }
    env.update({package: version(package) for package in ["transformers", "peft"]})
    write_json(results_dir(step) / "env.json", env)
