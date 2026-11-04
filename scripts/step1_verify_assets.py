#!/usr/bin/env python
"""Step 1: verify every asset the experiments depend on, on the A10 server.

Run from the repo root:
    uv run --locked python scripts/step1_verify_assets.py --all
or a subset:
    uv run --locked python scripts/step1_verify_assets.py --env --models --data

Writes small, git-committable JSON + a Markdown summary to results/step1/.
Each check runs independently; a failing check is recorded, not fatal.
"""

from __future__ import annotations

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fedorch.labels import item_image, item_label  # noqa: E402

OUT = ROOT / "results" / "step1"
# On the JKU servers scripts/server_env.sh sets FEDORCH_DATA / INSPLAD_DIR to studentwork paths.
EXT = Path(os.environ["FEDORCH_DATA"]) / "external" if os.environ.get("FEDORCH_DATA") else ROOT / "data" / "external"
INSPLAD_DEFAULT = Path(os.environ.get("INSPLAD_DIR", ROOT / "data" / "raw"))

# Decision-model candidates (see docs/federated_learning_design.md §2.2).
MODELS = {
    "small": ["sdmlai/nano-jev"],
    "medium": ["convaiinnovations/laya", "fastino/GLiNER2.5-Decide"],
    "large": ["jaredpalmer/kev-4b", "Qwen/Qwen3.5-4B-Base", "perplexity-ai/pplx-decider-flash-v1.2"],
    "optional_multimodal": ["LiquidAI/d1-3B"],
    "perception_tools": ["HuggingFaceTB/SmolVLM-256M-Instruct", "Qwen/Qwen3-VL-8B-Instruct"],
    "fallback_backbones": ["microsoft/MiniLM-L12-H384-uncased", "answerdotai/ModernBERT-large"],
}
# Small/medium tiers are fetched with --download; large ones only with --download-large.
DOWNLOAD_DEFAULT = ["sdmlai/nano-jev", "convaiinnovations/laya", "HuggingFaceTB/SmolVLM-256M-Instruct"]
# Never downloaded by --download-large (dropped: non-standard licence; fallbacks only on demand).
DOWNLOAD_SKIP_LARGE = {"LiquidAI/d1-3B"}
# Skip export formats we never load (SmolVLM ships ~3 GB of ONNX, ModernBERT ~4.7 GB).
IGNORE = ["*.onnx", "onnx/*", "*.onnx_data", "*.h5", "*.msgpack", "*.gguf", "*.tflite", "*.ot", "*.mlmodel", "coreml/*"]
# Disk needed besides the models (GB): InsPLAD zip 6.4 + unzipped ~6.5, venv ~8, outputs ~20.
OTHER_GB = {"InsPLAD (zip, delete after unzip)": 6.4, "InsPLAD unzipped": 6.5,
            "Python .venv (torch + CUDA libs)": 8.0, "experiment outputs (adapters, caches, logs)": 20.0}

ELECVQA_REPO = "https://github.com/GhostDino/elecvqa-bench.git"
LTE_ZIP = "https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip"

# One inspection-style decision used for the smoke tests.
SMOKE_STATE = (
    "Intent: detect rust or corrosion on clamps; corrosion is critical; abstain allowed. "
    "Evidence: classifier says rust p=0.58 (second: good p=0.31); crop 212x180 px; blur 0.21; exposure ok."
)
SMOKE_OPTIONS = {
    "accept_classifier": "accept the onboard classifier's label as final",
    "interpret_local": "run the onboard SmolVLM-256M for a 7-class answer",
    "interpret_cloud": "send the image to a larger cloud VLM",
    "abstain": "report cannot judge and keep the frame as evidence",
}


def save(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def guarded(fn):
    def run(*a, **k):
        t0 = time.time()
        try:
            res = fn(*a, **k)
            res = res if isinstance(res, dict) else {"result": res}
            res["ok"] = True
        except Exception as e:  # record and continue
            res = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
        res["seconds"] = round(time.time() - t0, 1)
        save(fn.__name__.removeprefix("check_"), res)
        print(f"[{'OK ' if res['ok'] else 'ERR'}] {fn.__name__} ({res['seconds']}s)")
        return res
    return run


@guarded
def check_env():
    import torch
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "disk_free_gb": round(shutil.disk_usage(ROOT).free / 1e9, 1),
    }
    env["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset: all GPUs visible>")
    env["hf_home"] = os.environ.get("HF_HOME", "<default ~/.cache/huggingface>")
    if torch.cuda.is_available():
        env["torch_arch_list"] = torch.cuda.get_arch_list()
        env["cuda"] = torch.version.cuda
        gpus = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            cc = f"{p.major}.{p.minor}"
            rec = {"index": i, "name": p.name, "mem_gb": round(p.total_memory / 1e9, 1), "capability": cc,
                   "free_gb": round(torch.cuda.mem_get_info(i)[0] / 1e9, 1)}
            try:  # does this torch build actually run kernels on this GPU?
                x = torch.ones(256, 256, device=f"cuda:{i}")
                rec["kernel_ok"] = bool((x @ x).sum().item() == 256 ** 3)
            except Exception as e:
                rec["kernel_ok"] = False
                rec["kernel_error"] = f"{type(e).__name__}: {e}"[:300]
            rec["bf16"] = p.major >= 8  # Ampere+; V100 (7.0) and P40 (6.1) must use fp16/fp32
            if p.major < 7:
                rec["warning"] = "Pascal-class GPU: no bf16, slow fp16, may be unsupported by recent torch builds"
            gpus.append(rec)
        env["gpus"] = gpus
    for pkg in ["transformers", "peft", "flwr", "huggingface_hub", "laya", "nanojev"]:
        try:
            env[pkg] = __import__(pkg).__version__
        except Exception as e:
            env[pkg] = f"missing ({type(e).__name__})"
    try:
        env["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        pass
    return env


@guarded
def check_models(download: bool, download_large: bool):
    from huggingface_hub import HfApi, snapshot_download
    api, out = HfApi(), {}
    for tier, ids in MODELS.items():
        for mid in ids:
            rec = {"tier": tier}
            try:
                info = api.model_info(mid)
                card = info.card_data.to_dict() if info.card_data else {}
                rec.update(exists=True, sha=info.sha, license=card.get("license"),
                           license_name=card.get("license_name"), library=info.library_name,
                           pipeline=info.pipeline_tag, gated=info.gated,
                           params=(info.safetensors.total if info.safetensors else None),
                           files=len(info.siblings or []))
                want = (download and mid in DOWNLOAD_DEFAULT) or (
                    download_large and tier != "fallback_backbones" and mid not in DOWNLOAD_SKIP_LARGE)
                if want:
                    path = snapshot_download(mid, revision=info.sha, ignore_patterns=IGNORE)
                    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
                    rec.update(downloaded=True, local_path=path, size_gb=round(size / 1e9, 2))
            except Exception as e:
                rec.update(exists=False, error=f"{type(e).__name__}: {e}")
            out[mid] = rec
    return {"models": out}


def _repo_gb(api, mid: str) -> float:
    from fnmatch import fnmatch
    files = api.list_repo_tree(mid, recursive=True)
    size = sum(getattr(f, "size", 0) or 0 for f in files
               if getattr(f, "size", None) is not None and not any(fnmatch(f.path, pat) for pat in IGNORE))
    return round(size / 1e9, 2)


@guarded
def check_disk():
    """Disk needed (models after IGNORE filtering + data + venv + outputs) vs free space."""
    from huggingface_hub import HfApi, constants
    api = HfApi()
    plan = {"--download (small/medium + SmolVLM)": DOWNLOAD_DEFAULT,
            "--download-large (adds)": [m for t, ids in MODELS.items() if t != "fallback_backbones"
                                        for m in ids if m not in DOWNLOAD_DEFAULT and m not in DOWNLOAD_SKIP_LARGE],
            "fallback backbones (only if needed)": MODELS["fallback_backbones"]}
    models, totals = {}, {}
    for group, ids in plan.items():
        totals[group] = 0.0
        for mid in ids:
            models[mid] = _repo_gb(api, mid)
            totals[group] += models[mid]
        totals[group] = round(totals[group], 2)
    need = round(sum(totals.values()) + sum(OTHER_GB.values()), 1)
    hf_home = Path(constants.HF_HUB_CACHE)
    probe = hf_home if hf_home.exists() else hf_home.parent if hf_home.parent.exists() else ROOT
    return {"model_gb": models, "group_gb": totals, "other_gb": OTHER_GB, "total_needed_gb": need,
            "hf_cache": str(hf_home), "free_gb_at_hf_cache": round(shutil.disk_usage(probe).free / 1e9, 1),
            "free_gb_at_repo": round(shutil.disk_usage(ROOT).free / 1e9, 1)}


def _timeit(fn, n=30):
    fn()  # warm-up
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": round(statistics.median(ts), 2), "p95_ms": round(sorted(ts)[int(0.95 * len(ts)) - 1], 2)}


@guarded
def check_smoke():
    """Load the small and medium decision models and score one inspection decision."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    res, dev = {}, ("cuda" if torch.cuda.is_available() else "cpu")

    # Small: nano-jev, plain transformers, per-option scoring (format from its model card).
    repo, rev = "sdmlai/nano-jev", "v1.0"
    tok = AutoTokenizer.from_pretrained(repo, revision=rev)
    model = AutoModelForSequenceClassification.from_pretrained(repo, revision=rev).to(dev).eval()
    temps = json.load(open(hf_hub_download(repo, "calibration.json", revision=rev)))
    question = "Which action should the drone take for this frame?"
    names = list(SMOKE_OPTIONS)
    enc = tok([f"question: {question} option: {n}: {d}" for n, d in SMOKE_OPTIONS.items()],
              [SMOKE_STATE] * len(names), truncation="only_second", max_length=512,
              padding=True, return_tensors="pt").to(dev)

    def score():
        with torch.no_grad():
            return model(**enc).logits.squeeze(-1)

    probs = torch.softmax(score() / list(temps.values())[0], dim=0).tolist()
    res["nano_jev"] = {"device": dev, "probs": dict(zip(names, [round(p, 3) for p in probs])),
                       "latency_4_options": _timeit(score), "temperatures": temps,
                       "params": sum(p.numel() for p in model.parameters())}

    # Medium: Laya via its SDK (format from its model card).
    os.environ.setdefault("USE_TF", "0")
    import laya
    agent = laya.load("convaiinnovations/laya")
    qs = {"action": {"type": "choice", "instructions": question, "criteria": SMOKE_OPTIONS}}
    out = agent.predict(SMOKE_STATE, qs)
    res["laya"] = {"answer": out["answers"]["action"], "latency_1_question": _timeit(lambda: agent.predict(SMOKE_STATE, qs), n=20)}
    res["note"] = "Zero-shot only: these probabilities are not meaningful until fine-tuned on our decision traces."
    return res


@guarded
def check_data(insplad_dir: Path):
    EXT.mkdir(parents=True, exist_ok=True)
    repo = EXT / "elecvqa-bench"
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", ELECVQA_REPO, str(repo)])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    split_dir = repo / "data" / "benchmark" / "split_comparison" / "tower14"
    manifest = json.loads((split_dir / "manifest.json").read_text())
    res = {"elecvqa_commit": commit, "manifest": manifest, "splits": {}}
    missing_examples, have_insplad = [], insplad_dir.exists()
    for split in ["train", "val", "test"]:
        per_tower = collections.defaultdict(collections.Counter)
        n_img = n_found = 0
        for line in (split_dir / f"{split}.jsonl").open():
            item = json.loads(line)
            per_tower[item["_tower"]][item_label(item)] += 1
            if have_insplad:
                n_img += 1
                if (insplad_dir / item_image(item)).exists():
                    n_found += 1
                elif len(missing_examples) < 5:
                    missing_examples.append(item_image(item))
        totals = collections.Counter()
        for c in per_tower.values():
            totals += c
        sizes = sorted(sum(c.values()) for c in per_tower.values())
        res["splits"][split] = {
            "items": sum(sizes), "towers": len(per_tower),
            "tower_size_min_median_max": [sizes[0], sizes[len(sizes) // 2], sizes[-1]],
            "labels": dict(totals.most_common()),
            "per_tower": {t: dict(c) for t, c in sorted(per_tower.items())},
            "images_found": f"{n_found}/{n_img}" if have_insplad else "InsPLAD not found",
        }
    res["insplad_dir"] = str(insplad_dir)
    res["insplad_present"] = have_insplad
    res["missing_image_examples"] = missing_examples
    return res


@guarded
def check_trace():
    dest = EXT / "lte4g"
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.glob("**/*.log")):
        with urllib.request.urlopen(LTE_ZIP, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(dest)
    traces = {}
    for f in sorted(dest.glob("**/*.log")):
        mbps = []
        for line in f.read_text().split("\n"):
            cols = line.split()
            if len(cols) == 6 and int(cols[5]) > 0:
                mbps.append(int(cols[4]) * 8 / (int(cols[5]) / 1000) / 1e6)  # bytes / ms -> Mbit/s
        if mbps:
            s = sorted(mbps)
            traces[f.stem] = {"seconds": len(mbps), "mean_mbps": round(statistics.mean(mbps), 2),
                              "p10_mbps": round(s[len(s) // 10], 2), "min_mbps": round(s[0], 2),
                              "max_mbps": round(s[-1], 2),
                              "frac_below_2mbps": round(sum(m < 2 for m in mbps) / len(mbps), 3)}
    return {"n_traces": len(traces), "traces": traces,
            "format": "ms_epoch, ms_since_start, lat, lon, bytes_since_last, ms_since_last"}


def write_summary() -> None:
    lines = ["# Step 1 report", ""]
    for name in ["env", "disk", "models", "smoke", "data", "trace"]:
        p = OUT / f"{name}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        lines.append(f"## {name}: {'OK' if d.get('ok') else 'ERROR'} ({d.get('seconds')} s)")
        if not d.get("ok"):
            lines.append(f"```\n{d.get('error')}\n```")
        elif name == "env":
            lines.append(", ".join(f"{k}={v}" for k, v in d.items() if k not in ("ok", "seconds", "gpus", "torch_arch_list")))
            for g in d.get("gpus", []):
                lines.append(f"- GPU {g['index']}: {g['name']} {g['mem_gb']} GB (free {g['free_gb']}), cc {g['capability']}, "
                             f"kernel_ok={g['kernel_ok']}, bf16={g['bf16']} {g.get('warning', '')}")
        elif name == "disk":
            lines.append(f"needed ~{d['total_needed_gb']} GB; free {d['free_gb_at_hf_cache']} GB at HF cache ({d['hf_cache']}), "
                         f"{d['free_gb_at_repo']} GB at repo; groups {d['group_gb']}")
        elif name == "models":
            lines.append("| model | tier | exists | license | params | downloaded |\n|---|---|---|---|---|---|")
            for mid, r in d["models"].items():
                lines.append(f"| {mid} | {r['tier']} | {r.get('exists')} | {r.get('license')} | {r.get('params')} | {r.get('size_gb', '')} |")
        elif name == "smoke":
            lines.append(f"nano-jev latency: {d['nano_jev']['latency_4_options']} on {d['nano_jev']['device']}; "
                         f"laya latency: {d['laya']['latency_1_question']}")
        elif name == "data":
            for s, v in d["splits"].items():
                lines.append(f"- {s}: {v['items']} items, {v['towers']} towers, sizes {v['tower_size_min_median_max']}, images {v['images_found']}")
        elif name == "trace":
            lines.append(f"{d['n_traces']} LTE traces parsed")
        lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'summary.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="env + models(+download) + smoke + data + trace")
    ap.add_argument("--env", action="store_true")
    ap.add_argument("--disk", action="store_true", help="disk needed vs free (no downloads)")
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--download", action="store_true", help="download small/medium models + SmolVLM")
    ap.add_argument("--download-large", action="store_true", help="also download 3-9B models (~40 GB)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--insplad-dir", type=Path, default=INSPLAD_DEFAULT,
                    help="folder containing InsPLAD-det/, supervised_fault_classification/, unsupervised_anomaly_detection/")
    a = ap.parse_args()
    if a.all:
        a.env = a.disk = a.models = a.download = a.smoke = a.data = a.trace = True
    if not any([a.env, a.disk, a.models, a.smoke, a.data, a.trace]):
        ap.error("choose at least one check, or --all")
    if a.env:
        check_env()
    if a.disk:
        check_disk()
    if a.models:
        check_models(a.download, a.download_large)
    if a.smoke:
        check_smoke()
    if a.data:
        check_data(a.insplad_dir)
    if a.trace:
        check_trace()
    write_summary()


if __name__ == "__main__":
    main()
