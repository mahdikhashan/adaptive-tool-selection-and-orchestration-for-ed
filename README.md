# Federated tool orchestration for edge-AI drone inspection

Target: FGCS special issue *Agentic Federated Learning for the Edge Intelligence* (deadline 2027-01-31).
Design: `docs/federated_learning_design.md`. Novelty check: `docs/novelty_check.md`.
Asset verification: `docs/asset_verification.md`.

## Workflow

Code is written here and pushed to GitHub. You pull it on the A10 server, run the scripts, and push back the
**small** result files. Large files never go into git (see "Transferring results").

```
 laptop (Claude)              GitHub                 A10 server (you)
 write scripts  ── push ──▶  branch  ── pull ──▶   uv sync; run scripts
 analyse results ◀── pull ──  branch  ◀── push ──   commit results/<step>/
```

## One-time setup on the A10 server

```bash
git clone https://github.com/mahdikhashan/adaptive-tool-selection-and-orchestration-for-ed.git fedorch
cd fedorch
git checkout step1-asset-verification      # or main once merged

# uv (skip if installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync --locked                            # creates .venv from uv.lock (Python 3.11-3.13)
uv run --locked python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `False`, check the NVIDIA driver (`nvidia-smi`): the locked torch wheel
needs a driver that supports its CUDA runtime.

Optional: `export HF_HOME=/path/with/space/hf_cache` before downloading models (the large tier is about 40 GB).

## Disk space

| What | GB |
|---|---|
| Small + medium decision models + SmolVLM (`--download`) | 3.0 |
| Large tier + cloud VLM (`--download-large`: GLiNER2.5-Decide 1.95, Kev-4B adapter 0.16, Qwen3.5-4B-Base 9.3, pplx-decider-flash 9.1, Qwen3-VL-8B 17.6) | 38.1 |
| Fallback backbones (only if a checkpoint fails) | 3.3 |
| InsPLAD: 6.4 zip + ~6.5 unzipped (delete the zip afterwards) | 12.9 |
| Python `.venv` (torch + CUDA libraries), Kev repo env adds about the same again | 8–16 |
| Experiment outputs (adapters, tool cache, logs) | ~20 |
| **Total, everything** | **~85–95** |
| Minimum to start (step 1 + step 2 without the large tier) | ~45 |

Sizes are from the Hugging Face file listings (2026-11-04), excluding ONNX/TF/GGUF exports, which the script
skips. Check your server before downloading:

```bash
df -h .                                   # free space where the repo lives
df -h ${HF_HOME:-~/.cache/huggingface}    # free space where models go (often a different disk)
uv run --locked python scripts/step1_verify_assets.py --disk   # needed vs free, no downloads
du -sh ${HF_HOME:-~/.cache/huggingface}/hub/* 2>/dev/null | sort -h   # what is already cached
```

If the home disk is small, put models on a larger disk: `export HF_HOME=/big/disk/hf_cache`, and keep InsPLAD
there too (`--insplad-dir /big/disk/insplad`).

## Step 1: verify assets

### 1a. Download InsPLAD (manual, once)

ElecVQA-Bench does not redistribute images. Download InsPLAD from Mendeley Data
(<https://data.mendeley.com/datasets/5n3fjgvfyz/1>, "Download All"), then unzip so that you have:

```
data/raw/InsPLAD-det/
data/raw/supervised_fault_classification/
data/raw/unsupervised_anomaly_detection/
```

(Or keep it elsewhere and pass `--insplad-dir /path/to/that/folder`.)

### 1b. Run

```bash
uv run --locked python scripts/step1_verify_assets.py --all
# also fetch the 3-9B models (Kev-4B base, pplx-decider-flash, d1, Qwen3-VL-8B):
uv run --locked python scripts/step1_verify_assets.py --models --download-large
```

`--all` checks the environment and GPU; checks all candidate models on Hugging Face (licence, size, revision)
and downloads the small and medium ones; loads nano-jev and Laya and times one decision on the A10; clones
ElecVQA-Bench and computes per-tower statistics, verifying that every image path resolves in your InsPLAD copy;
and downloads and summarises the 4G/LTE traces.
Output goes to `results/step1/` (`summary.md` plus one JSON per check).

## Transferring results

**Small results (JSON, CSV, Markdown, logs; under ~50 MB in total): commit and push.**

```bash
git add results/step1
git commit -m "step1 results from A10"
git push origin step1-asset-verification
```

Then tell Claude "pushed step 1". Claude runs `git pull` and reads `results/step1/` directly.

**Large files (checkpoints, adapters, raw tool outputs): do not commit** (`.gitignore` blocks the usual types).
Either copy them to the laptop's project artifacts folder:

```bash
rsync -avz results/<step>/large/ \
  <laptop>:~/.local/share/openresearch/files/adaptive-tool-selection-and-orchestration-for-ed/<step>/
```

or upload them to a private Hugging Face dataset repo (`huggingface-cli upload <user>/fedorch-artifacts results/<step>/large`)
and share the repo name. Every script writes the summary statistics Claude needs into the small files, so large
files are only needed for re-analysis.

**Always include** the `env.json` that each step writes (git commit, GPU, package versions), so every result
can be traced to the exact code and machine.
