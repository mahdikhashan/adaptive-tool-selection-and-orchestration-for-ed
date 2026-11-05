# Federated tool orchestration for edge-AI drone inspection

Target: FGCS special issue "Agentic Federated Learning for the Edge Intelligence". The deadline is 2027-01-31.

Related documents:
- `docs/federated_learning_design.md`: the design.
- `docs/novelty_check.md`: the novelty check.
- `docs/asset_verification.md`: the models and the datasets.

## Workflow

1. Claude writes the scripts on the laptop and pushes a branch to GitHub.
2. You pull the branch on the JKU server.
3. You run the scripts.
4. You commit the small result files and push them.
5. Claude pulls the results and analyses them.

Files of 1 GB or larger never go into git. See "Transfer results".

## Server

- Server: `student07`. GPU: the booked **GV100, 32 GB**.
- The GV100 has compute capability 7.0. It supports fp16 and fp32. It does not support bf16.
- On Linux, `uv.lock` installs torch 2.14.1 for CUDA 12.6. The torch 2.15 build for CUDA 13.2 has no kernels for compute capability 7.0.
- SSH port: 5792. You need the VPN when you are not on campus.
- Storage: put all files in `/system/user/studentwork/khashan`. The limit is 200 GB.
- Do not put data in the home directory `/system/user/khashan`. The limit is 20 GB.
- Other users also use the GPUs. Book a GPU in the institute spreadsheet before you use it.
- Use `tmux` for long jobs. A job in `tmux` continues after you log out.

## Setup (one time)

```bash
ssh -p 5792 khashan@student07.ai-lab.jku.at
cd /system/user/studentwork/khashan
git clone https://github.com/mahdikhashan/adaptive-tool-selection-and-orchestration-for-ed.git fedorch
cd fedorch
git checkout step1-asset-verification
tmux new -s fedorch
bash scripts/setup_server.sh
```

`setup_server.sh` does these steps:
1. It installs `uv` in `studentwork/khashan/bin` if `uv` is not available.
2. It makes the Python environment from `uv.lock`.
3. It downloads InsPLAD (6.4 GB) and extracts all three folders. ElecVQA-Bench uses all three folders.

You can run `setup_server.sh` again at any time. It does not repeat completed downloads.

If the clone in your home directory from student06 is still there, delete it.

## Every new shell

```bash
cd /system/user/studentwork/khashan/fedorch
source scripts/server_env.sh
nvidia-smi
export CUDA_VISIBLE_DEVICES=<index of the booked GV100>
```

`server_env.sh` sends all caches and data to `studentwork/khashan`. Use the `nvidia-smi` index of the GV100 that you booked.

## Disk space

| Item | GB |
|---|---|
| Models (7 checkpoints, without ONNX and GGUF files) | 39 |
| InsPLAD (zip during download, then the extracted files) | 13 at peak |
| Python environment and caches | 10 |
| Experiment outputs | 20 |
| **Total** | **about 82 of 200** |

To see the free space and the space that the models need, run:

```bash
df -h $FEDORCH_WORK
uv run --locked python scripts/step1_verify_assets.py disk
```

## Step 1: check the assets

```bash
uv run --locked python scripts/step1_verify_assets.py
```

The script runs six checks. Each check writes one JSON file in `results/step1/`.

| Check | Result |
|---|---|
| `env` | Python, torch, CUDA, package versions, and each visible GPU with a test kernel |
| `disk` | The space that the models need, and the free space |
| `models` | Downloads the seven checkpoints and records their revisions and licences |
| `smoke` | Loads nano-jev and Laya and measures one decision on the GPU |
| `data` | Clones ElecVQA-Bench, counts items per tower, and finds missing InsPLAD images |
| `trace` | Downloads the 4G/LTE traces and calculates the throughput of each trace |

To run only some checks, give their names: `... step1_verify_assets.py env smoke`.

## Step 2: build the tool cache

Step 2 trains two tools and runs five tools on all 39,228 tower14 items. Run the commands in this order:

```bash
git pull
uv sync --locked
uv run --locked python scripts/step2_train_classifier.py
uv run --locked python scripts/step2_train_smolvlm.py
for tool in classifier smolvlm_base smolvlm_local smolvlm_split cloud; do
  uv run --locked python scripts/step2_run_tools.py $tool --limit 200
done
```

The `--limit 200` run is a test. It shows the speed of each tool. Then run the full cache:

```bash
for tool in classifier smolvlm_base smolvlm_local smolvlm_split cloud; do
  uv run --locked python scripts/step2_run_tools.py $tool
done
uv run --locked python scripts/step2_summarize.py
```

`step2_run_tools.py` continues from the last completed item. If another user needs the GPU, stop the script with `Ctrl-c` and run the same command again later.

| Tool | Model | Place | Precision |
|---|---|---|---|
| `classifier` | ResNet-50, trained on the train towers | drone | fp16 |
| `smolvlm_base` | SmolVLM-256M, not fine-tuned | drone | fp32 |
| `smolvlm_local` | SmolVLM-256M with the LoRA adapter (EVLM recipe) | drone | fp32 |
| `smolvlm_split` | `smolvlm_local` with INT8 visual tokens | drone and cloud | fp32 |
| `cloud` | Qwen3-VL-8B, not fine-tuned | cloud | fp16 |

Each VLM reads the image at a maximum side of 512 px. Each VLM gives one probability for each answer letter in one forward pass. The tools do not generate text.

Outputs:
- `results/step2/tool_<tool>.jsonl`: one line for each item (prediction, correct, confidence, probabilities, compute time, payload).
- `results/step2/classifier_train.json`, `smolvlm_train.json`: training curves and validation results.
- `results/step2/summary.json`: accuracy, recall for each label, abstention rate, compute time and payload for each tool.
- `$FEDORCH_WORK/runs/step2/`: the classifier weights and the LoRA adapter. These files stay on the server.

Commit `results/step2` when all five tools are complete.

## Transfer results

Commit and push all result files that are smaller than 1 GB:

```bash
git add results/<step>
git commit -m "<step> results from student07"
git push origin step1-asset-verification
```

Then tell Claude "pushed step 1".

Git LFS stores these result types: `.safetensors`, `.pt`, `.bin`, `.npy`, `.parquet`, `.jsonl` (see `.gitattributes`). Git stores all other files (JSON, CSV, Markdown, logs).

Before you commit, find files that are 1 GB or larger:

```bash
find results -size +1G
```

Do not commit these files. Copy them to the laptop with `rsync`:

```bash
rsync -avz -e "ssh -p 5792" \
  khashan@student07.ai-lab.jku.at:/system/user/studentwork/khashan/fedorch/results/<step>/ \
  ~/.local/share/openresearch/files/adaptive-tool-selection-and-orchestration-for-ed/<step>/
```

GitHub limits LFS storage and LFS bandwidth for each account. Check the LFS usage in the GitHub settings ("Billing and licensing") before you push large adapters.

Each step writes `env.json`. This file records the git commit, the GPU, and the package versions. Always commit it.
