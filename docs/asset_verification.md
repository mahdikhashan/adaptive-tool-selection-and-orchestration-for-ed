# Step 1: models and datasets

Date: 2026-11-04. Claude checked these facts on the laptop with the Hugging Face API, PyPI, GitHub and the dataset pages. The server run of `scripts/step1_verify_assets.py` must check the items in "Items to check on the server".

## Controller models

| Tier | Checkpoint | Size | Licence | Interface | Fine-tuning method |
|---|---|---|---|---|---|
| Small (main) | `sdmlai/nano-jev` v1.0 | 33.4M | MIT | MiniLM-L12 cross-encoder. It gives one score for each option. Input: `question: {q} option: {o}` and a context text. | `transformers` and PEFT LoRA |
| Medium | `convaiinnovations/laya` | 421M | Apache-2.0 | ModernBERT-large and a decision head. The head has an act/escalate output. | Laya fine-tuning notebook and PEFT LoRA |
| Large | `jaredpalmer/kev-4b` | 33.8M LoRA on `Qwen/Qwen3.5-4B-Base` | Apache-2.0 | A causal language model in prefill mode and a pointer head over the options | `kev.train --init_from jaredpalmer/kev-4b` from the Kev repository |
| Multimodal (optional) | `perplexity-ai/pplx-decider-flash-v1.2` | 4.5B | Apache-2.0 | A Qwen3.5-4B decision model. It accepts images up to 1.6 MP. | LoRA with the source package in the model repository |

Fallback backbones: `microsoft/MiniLM-L12-H384-uncased`, `answerdotai/ModernBERT-large`, `fastino/GLiNER2.5-Decide` (486M, Apache-2.0). The script records their revisions but does not download them.

Corrections to the first model table:
- The 33.4M "NanoJev" is `sdmlai/nano-jev`. `C-Tianyu/NanoJev` is a different Qwen3-0.6B model without a licence.
- GLiNER2.5-Decide has 486M parameters, not 340M.
- AnyJev has checkpoints only at 31B. SemIf has only MLX ports. The training-free baseline uses our own option-logit readout on `Qwen/Qwen3.5-4B-Base`.
- `LiquidAI/d1-3B` uses the LFM 1.0 licence. We do not use it. `pplx-decider-flash` is the multimodal option.

Facts from the model cards that change the design:
- nano-jev: accuracy decreases by 15 to 20 points on new datasets, and the calibration does not transfer. We must fine-tune and calibrate on our decision traces.
- Laya has an act/escalate head. We use this head for the accept or escalate decision.
- Kev-4B: the option order can change an answer. We put the options in a random order during training and evaluation.

## Perception tools

- `HuggingFaceTB/SmolVLM-256M-Instruct`: the onboard VLM. EVLM uses this model.
- `Qwen/Qwen3-VL-8B-Instruct`: the cloud VLM. It needs about 17.6 GB of GPU memory in fp16. It fits on the 32 GB GV100.

## Datasets

| Dataset | Source | Licence | Download |
|---|---|---|---|
| ElecVQA-Bench | `https://github.com/GhostDino/elecvqa-bench` | MIT for code. The metadata refers to InsPLAD. | The `data` check clones it |
| InsPLAD | Mendeley Data `5n3fjgvfyz`, file `InsPLAD_Dataset.zip`, 6.4 GB, SHA-256 `effd039e...0f489139` | See the InsPLAD repository | `scripts/get_insplad.py` downloads it and extracts all three folders |
| 4G/LTE traces (van der Hooft et al.) | `https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip` | Public dataset | The `trace` check downloads it |

### InsPLAD folders that ElecVQA-Bench uses

| Task | InsPLAD folder | Images |
|---|---|---|
| T1 grounding | `InsPLAD-det/` (full-size photos) | 10,559 |
| T2 binary | `supervised_fault_classification/`, `unsupervised_anomaly_detection/` | 9,997 |
| T3 multiclass | `supervised_fault_classification/`, `unsupervised_anomaly_detection/` | 36,370 |

All three folders are necessary. `get_insplad.py` extracts all three.

### ElecVQA-Bench tower14 split

| Split | Items | Towers | Items per tower (min / median / max) |
|---|---|---|---|
| train | 25,499 | 18 | 16 / 742 / 6,449 |
| val | 6,672 | 10 | 13 / 221 / 3,191 |
| test | 7,057 | 14 | 11 / 99 / 2,054 |

Labels in the train split: good 22,377, rust 1,009, defect (binary items) 1,002, corrosion 468, missing-cap 351, nest 273, torn-up 17, peeling-paint 2.

Consequences for the design:
- The prompts are in Chinese. Our tools use English. `src/fedorch/labels.py` changes each option to an English label. All labels change without errors.
- The 42 towers have very different sizes. Rare defects occur only on some towers. Thus the towers give non-IID clients.
- Client plan: the 28 train and val towers make 8 to 10 clients. Each client splits its items by `parent_id`. The 14 test towers are an unseen fleet for the global test and for the new-drone experiment (E6).
- Torn-up (17 items) and peeling-paint (2 items) are too rare for training. We merge them into "other-defect" or remove them. The paper reports this decision.
- About 88% of the items are "good". We use a class-weighted loss and report the recall of each class.

### 4G/LTE traces

The `trace` check found 40 traces (bicycle, bus, car, foot, train, tram). The traces have a total duration of about 5 hours. The mean throughput is 31.4 Mbps. The lowest trace mean is 14.1 Mbps. In the median trace, only about 1% of the seconds are below 2 Mbps.

Thus the traces do not give a low-bandwidth condition. We add outages and lower the bandwidth for some clients. The paper identifies this condition as constructed.

## Items to check on the server

1. The GV100 runs a CUDA kernel with the locked torch build (`env.json`, `kernel_ok`).
2. All seven checkpoints download (`models.json`).
3. nano-jev and Laya run on the GPU, and the script records the time of one decision (`smoke.json`).
4. All ElecVQA-Bench image paths exist in InsPLAD (`data.json`, `missing` is 0).
