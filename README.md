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

Step 2 trains two tools and runs five tools on all 39,228 tower14 items.

The controller learns from the tool results on the train and val towers. A tool that trained on a tower gives results on that tower that are too good. Thus step 2 uses cross-fitting:

1. `src/fedorch/folds.py` divides the 28 train and val towers into 3 folds. Each fold has approximately 10,700 items.
2. Each training script trains 4 models: `fold0`, `fold1`, `fold2` and `full`.
3. Model `foldN` trains on the towers of the other two folds. It gives the results for the towers of fold N.
4. Model `full` trains on all 28 train and val towers. It gives the results for the 14 test towers.

Thus no tool result comes from a model that trained on the same tower. The classifier selects its best epoch on a separate set of towers in its training data.

Run the commands in this order:

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
| `classifier` | ResNet-50, cross-fitted (4 models) | drone | fp16 |
| `smolvlm_base` | SmolVLM-256M, not fine-tuned | drone | fp32 |
| `smolvlm_local` | SmolVLM-256M with a LoRA adapter (EVLM recipe), cross-fitted (4 adapters) | drone | fp32 |
| `smolvlm_split` | `smolvlm_local` with INT8 visual tokens | drone and cloud | fp32 |
| `cloud` | Qwen3-VL-8B, not fine-tuned | cloud | fp16 |

Each VLM reads the image at a maximum side of 512 px. Each VLM gives one probability for each answer letter in one forward pass. The tools do not generate text.

Outputs:
- `results/step2/tool_<tool>.jsonl`: one line for each item (model, prediction, correct, confidence, probabilities, compute time, payload).
- `results/step2/classifier_train_<model>.json`, `smolvlm_train_<model>.json`: towers, training curves and validation results for each model.
- `results/step2/summary.json`: accuracy, recall for each label, abstention rate, compute time and payload for each tool.
- `$FEDORCH_WORK/runs/step2/`: the 4 classifier weight files and the 4 LoRA adapters. These files stay on the server.

Commit `results/step2` when all five tools are complete.

## Step 3: replay the baselines

Step 3 does not need a GPU. It uses the step 2 cache in `results/step2/`.

```bash
uv run --locked python scripts/step3_replay.py
uv run --locked python scripts/step3_replay.py --jetson-factor 5 --out metrics_jetson5.json
uv run --locked python scripts/step3_replay.py --jetson-factor 35 --out metrics_jetson35.json
```

The results are in `results/step3/`. `docs/step3_results.md` explains them.

## Step 4: learned controllers (central training)

Each controller learns q(item, tool): the probability that the tool gives the correct answer. A constant rule in `src/fedorch/controller.py` then uses q, the risk weights (`docs/risk_weights.md`) and the latency to select the tool.

Run the two decision models on the GPU:

```bash
git pull
uv run --locked python scripts/step4_train_nanojev.py
uv run --locked python scripts/step4_train_laya.py
git add results/step4
git commit -m "step4 decision-model results"
git push origin step1-asset-verification
```

Each script writes `results/step4/q_<model>.jsonl` and a training summary. The weights stay in `$FEDORCH_WORK/runs/step4/`.

Claude runs the other parts on the laptop. These parts do not need a GPU:

```bash
uv run --locked python scripts/step4_train_numeric.py
uv run --locked python scripts/step4_replay.py
```

## Step 4b: decision models with the choice target

Step 4b gives every controller the same new target and the same new evidence. The target is one choice between `accept`, `smolvlm_local` and `cloud`. Its soft probabilities come from the realised risk of each option. The evidence adds the full classifier output (6 classes) and image signals (size, blur, brightness, contrast).

Run these commands on student07 in this order:

```bash
git pull
uv run --locked python scripts/step4b_extract_evidence.py
git add results/step4b/evidence.jsonl
git commit -m "step4b evidence"
git push origin step1-asset-verification
uv run --locked python scripts/step4b_train_nanojev.py
uv run --locked python scripts/step4b_train_laya.py
git add results/step4b
git commit -m "step4b decision-model results"
git push origin step1-asset-verification
```

Push the evidence file first. Claude then trains the MLP on the laptop while the decision models train on the GPU.

## Plan A: controllers with few labelled frames

Plan A trains each controller on 1%, 5%, 20% and 100% of the train-tower frames, with 3 seeds. All controllers use the setting from step 4b (temperature 1, 5 copies of each informative frame). Claude trains the MLP on the laptop. The 100% run of nano-jev with seed 0 is the step 4b run.

Run nano-jev and Laya on student07 in `tmux`:

```bash
git pull
bash scripts/planA_run_gpu.sh
git add results/planA
git commit -m "plan A decision-model runs"
git push origin step1-asset-verification
```

`planA_run_gpu.sh` skips each run that has its `p_*.jsonl` file. A run writes this file only at its end. Thus, after a stop, run the script again: it starts the missing runs only. Each run writes its log to `logs/planA_<run>.log`.

A small training set gets more epochs: nano-jev trains for at least 600 optimizer steps, and Laya for at least 300 optimizer updates. Claude then runs `scripts/planA_replay.py` on the laptop.

## Laya-vision zero-shot (branch `vision-fl`)

Laya-vision looks at each image and answers 4 questions: component, defect (yes or no), condition and image quality. The model is not fine-tuned. The pipeline has no other model. The design is in `docs/vision_fl_design.md`.

The branch has different packages (laya-vision as `laya`, `transformers==5.17.0`). Thus sync again after the checkout:

```bash
git fetch origin
git checkout vision-fl
uv sync --locked
```

Do a short test first, with 20 images:

```bash
uv run --locked python scripts/vision_run.py --limit 20 --latency-frames 5
rm -rf results/vision
```

Then do the full run in `tmux`:

```bash
bash scripts/vision_run_gpu.sh
git add results/vision
git commit -m "laya-vision zero-shot"
git push origin vision-fl
```

The logs are `logs/vision_zeroshot.log` and `logs/vision_latency_cpu.log`. After a stop, run the script again: it continues from the last image. Claude then runs `scripts/vision_summarize.py` on the laptop.

## Laya-vision fine-tuning (branch `vision-fl`)

This run follows the fine-tune guide of laya-vision (`site-docs/how-to/fine-tune.md`). The design is in `docs/vision_finetune.md`.

Do a short test first:

```bash
uv run --locked python scripts/vision_finetune.py --name smoke --per-class 3 --calib-images 2 --val-images 3 --passes 0.5 --limit 4 --latency-frames 2
rm -rf results/vision/smoke "$FEDORCH_WORK/models/vision/smoke"
```

Then do the full run in `tmux`:

```bash
bash scripts/vision_finetune_gpu.sh
git add results/vision/lastn_pc400
git commit -m "laya-vision fine-tune lastn_pc400"
git push origin vision-fl
```

The model is in `$FEDORCH_WORK/models/vision/lastn_pc400` (about 800 MB). Do not commit it. The log is `logs/vision_finetune.log`. If the prediction part stops, run the script again: the training starts again, but the prediction continues from the last image. Claude then runs `scripts/vision_summarize.py lastn_pc400` on the laptop.

## Laya-vision fine-tuning, round 2 (branch `vision-fl`)

Round 2 uses more training images and trains 2 models: `good300_lastn` and `good300_full`. The design is in `docs/vision_finetune_round2.md`.

```bash
git pull
uv sync --locked
bash scripts/vision_finetune_round2.sh
git add results/vision/good300_lastn results/vision/good300_full
git commit -m "laya-vision fine-tune round 2"
git push origin vision-fl
```

The script skips a model that has its `run.json`. The logs are `logs/vision_good300_lastn.log` and `logs/vision_good300_full.log`.

## Laya-vision federated learning (branch `vision-fl`)

The plan is in `docs/vision_fl_plan.md`. One script makes 8 runs (about 10 hours on the RTX 2080 Ti). It skips each run that has its `run.json`.

Do a short test first:

```bash
uv run --locked python scripts/vision_fl.py --name smoke --mode fedavg --drones 3 --rounds 2 --passes 0.02 --limit 12 --val-images 4 --latency-frames 2
rm -rf results/vision/smoke "$FEDORCH_WORK/models/vision/smoke"
```

Then do all runs in `tmux`:

```bash
tmux new -s fl
bash scripts/vision_fl_runs.sh
git add results/vision/fl_*
git commit -m "laya-vision federated runs"
git push origin vision-fl
```

The models are in `$FEDORCH_WORK/models/vision/fl_*` (about 800 MB each). Do not commit them. The logs are `logs/vision_fl_*.log`.

The script puts the temporary files of the data loaders in `/dev/shm` (local memory). On the network drive, Python cannot delete these files, and each data loader writes an error to the log.

## Transfer results

Commit and push all result files that are smaller than 1 GB:

```bash
git add results/<step>
git commit -m "<step> results from student07"
git push origin step1-asset-verification
```

Then tell Claude "pushed step 1".

Git LFS stores these result types: `.safetensors`, `.pt`, `.bin`, `.npy`, `.parquet` (see `.gitattributes`). Git stores all other files (JSON, CSV, Markdown, logs).

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
