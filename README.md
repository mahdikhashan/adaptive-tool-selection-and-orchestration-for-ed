# Federated tool orchestration for edge-AI drone inspection

Target: FGCS special issue *Agentic Federated Learning for the Edge Intelligence* (deadline 2027-01-31).
Design: `docs/federated_learning_design.md`. Novelty check: `docs/novelty_check.md`.
Asset verification: `docs/asset_verification.md`.

## Workflow

Code is written on the laptop and pushed to GitHub. You pull it on the JKU AI-lab server, run the scripts,
and push back the **small** result files. Large files never go into git (see "Transferring results").

```
 laptop (Claude)              GitHub                 JKU server (you)
 write scripts  ── push ──▶  branch  ── pull ──▶   setup_server.sh; run scripts
 analyse results ◀── pull ──  branch  ◀── push ──   commit results/<step>/
```

## Server facts (JKU AI-lab student servers)

- SSH port **5792**; off campus you need the **VPN**. Example: `ssh -p 5792 khashan@student06.ai-lab.jku.at`.
- **Storage:** everything goes in `/system/user/studentwork/khashan` (**200 GB limit**). The home directory
  `/system/user/khashan` is limited to 20 GB: keep only dotfiles there. `scripts/server_env.sh` sends every
  cache (uv, Python installs, Hugging Face, torch, triton, pip, tmp) to studentwork.
- **GPUs are shared.** Book a GPU in the institute spreadsheet first, check `nvidia-smi`, then pin it with
  `export CUDA_VISIBLE_DEVICES=<id>`. Mark `can share` in the sheet if a job uses only part of a card.
- student06 currently has (from `nvidia-smi`, 2026-11-04): **GPU0 A10 24 GB** (Ampere, cc 8.6: bf16 ok),
  GPU1 V100 16 GB (cc 7.0: fp16, no bf16), GPU2–3 P40 24 GB (Pascal, cc 6.1: no bf16, slow fp16, and possibly
  unsupported by recent PyTorch builds). The plan assumes the **A10**. Step 1 tests each visible GPU (`kernel_ok`).
- Run long jobs in **tmux** so they survive logout: `tmux new -s fedorch`, detach with `Ctrl-b d`,
  re-attach with `tmux attach -t fedorch`.

## One-time setup

```bash
ssh -p 5792 khashan@student06.ai-lab.jku.at
cd /system/user/studentwork/khashan
git clone https://github.com/mahdikhashan/adaptive-tool-selection-and-orchestration-for-ed.git fedorch
cd fedorch && git checkout step1-asset-verification

tmux new -s fedorch                 # setup downloads ~6.4 GB, so run it inside tmux
bash scripts/setup_server.sh        # idempotent: re-run any time
```

`setup_server.sh`:
1. installs `uv` into `studentwork/khashan/bin`;
2. creates `.venv` from `uv.lock` and prints torch, CUDA and every visible GPU;
3. links `data/raw` and `data/external` to `studentwork/khashan/data/...`;
4. downloads InsPLAD from Mendeley (resumable, SHA-256 checked) and extracts **only**
   `supervised_fault_classification/` and `unsupervised_anomaly_detection/`. ElecVQA-Bench uses only these
   crops, never the full-size InsPLAD-det photos. The zip is then deleted.

If you cloned the repo into your home directory earlier, delete that copy (`rm -rf ~/adaptive-tool-selection-and-orchestration-for-ed`)
and use the studentwork clone. The ~3.3 GB of old Hugging Face downloads in `~/.cache/huggingface` can also go,
if you no longer need them.

**Every new shell:** `cd /system/user/studentwork/khashan/fedorch && source scripts/server_env.sh && export CUDA_VISIBLE_DEVICES=<booked GPU>`.

## Disk space (all in studentwork, 200 GB)

| What | GB |
|---|---|
| Small + medium decision models + SmolVLM (`--download`) | 3.0 |
| Large tier + cloud VLM (`--download-large`: GLiNER2.5-Decide 1.95, Kev-4B adapter 0.16, Qwen3.5-4B-Base 9.3, pplx-decider-flash 9.1, Qwen3-VL-8B 17.6) | 38.1 |
| Fallback backbones (only if a checkpoint fails) | 3.3 |
| InsPLAD: 6.4 zip while downloading, then the extracted crops (the zip is deleted) | ≤ 13 at peak |
| Python `.venv` + uv cache + Python install (the Kev repo env adds about the same again) | 10–20 |
| Experiment outputs (adapters, tool cache, logs) | ~20 |
| **Total, everything** | **~85–95 of 200** |

Check before and after downloads:

```bash
source scripts/server_env.sh
df -h $FEDORCH_WORK                                  # free space in studentwork
du -sh $FEDORCH_WORK/* | sort -h                     # what uses it
uv run --locked python scripts/step1_verify_assets.py --disk    # needed vs free, no downloads
```

## Step 1: verify assets

```bash
source scripts/server_env.sh && export CUDA_VISIBLE_DEVICES=0      # the A10, once it is booked and free
uv run --locked python scripts/step1_verify_assets.py --all
uv run --locked python scripts/step1_verify_assets.py --models --download-large   # ~38 GB, can run later
```

`--all` checks:
- **environment**: GPUs visible, compute capability, and whether a CUDA kernel actually runs on each (`kernel_ok`);
- **disk**: space needed vs free;
- **models**: every candidate on Hugging Face (licence, size, revision), downloading the small and medium ones;
- **smoke test**: loads nano-jev and Laya and times one decision on the GPU;
- **data**: clones ElecVQA-Bench, computes per-tower statistics, and checks that every image path resolves in InsPLAD;
- **trace**: downloads and summarises the 4G/LTE traces.

Output: `results/step1/summary.md` plus one JSON per check.

## Transferring results

**Small results (JSON, CSV, Markdown, logs; under ~50 MB in total): commit and push from the server.**

```bash
git add results/step1
git commit -m "step1 results from student06"
git push origin step1-asset-verification
```

Then tell Claude "pushed step 1". Claude pulls and reads `results/step1/` directly.

**Large files (checkpoints, adapters, raw tool outputs): never commit** (`.gitignore` blocks the usual types).
Pull them to the laptop with rsync over the institute's SSH port (VPN needed off campus):

```bash
# run on the laptop
rsync -avz -e "ssh -p 5792" \
  khashan@student06.ai-lab.jku.at:/system/user/studentwork/khashan/fedorch/results/<step>/large/ \
  ~/.local/share/openresearch/files/adaptive-tool-selection-and-orchestration-for-ed/<step>/
```

Or upload them to a private Hugging Face dataset repo (`hf upload <user>/fedorch-artifacts results/<step>/large`)
and share its name. Every script writes the summary statistics Claude needs into the small files, so large
files are only needed for re-analysis.

**Always include** the `env.json` each step writes (git commit, GPU, package versions), so every result can be
traced to the exact code and machine.
