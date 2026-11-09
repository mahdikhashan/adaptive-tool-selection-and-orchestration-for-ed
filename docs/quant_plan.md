# Plan: 4-bit laya-vision and SmolVLM-256M

Date: 2026-11-09. Branch: `vision-quant` (from `vision-fl`). Status: code complete, not run on the GPU.

## 1. Goal

Measure the accuracy, the GPU memory, the model size and the latency of 4 models in 4-bit format on 1 GPU (RTX 2080 Ti). Keep each 4-bit model on the server, so that we can push it to Hugging Face later. The results go into the paper.

## 2. Models

| Family | Model | Source |
|---|---|---|
| laya-vision | `base` | `thaitea/laya-vision`, revision `f2fe3c1` |
| laya-vision | `good300_full` | Our fine-tuned model, `$FEDORCH_WORK/models/vision/good300_full` |
| SmolVLM-256M | `base` | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| SmolVLM-256M | `lora` | The step 2 LoRA adapter `$FEDORCH_WORK/runs/step2/smolvlm_lora_full`, merged into the base weights |

## 3. Format

All linear layers use bitsandbytes 4-bit NormalFloat (nf4) weights with fp16 computation. The layers that are not linear (embeddings, normalization) use fp16. The script converts the weights to fp16 before the 4-bit conversion. No run uses fp32 weights in the backbone.

Exceptions:

- SmolVLM: bitsandbytes keeps `lm_head` in fp16, not in 4-bit.
- laya-vision: the decision head (8.5M parameters, outside the backbone) stays in fp32. The laya library changes the input of this head to fp32 (`h.float()` in `_readout`). Thus fp16 is not possible for the head without a change to the library. `run.json` records the data types of the backbone and of the head.

## 4. Storage

Each model goes to `$FEDORCH_WORK/models/quantized/<family>_<model>_nf4_fp16/`:

- laya-vision: `vlm_agent_config.json`, `processor/`, `head.safetensors` and `backbone_quantized/` (a Hugging Face model folder).
- SmolVLM: a Hugging Face model folder with the processor.

Each run loads the model again from this folder before the evaluation. Thus the evaluation tests the stored model.

## 5. Evaluation

| Family | Data | Metrics |
|---|---|---|
| laya-vision | 1,500 validation images (for the defect threshold) and all 6,092 test-tower images | Condition macro F1, component accuracy, defect AUROC (with 95% bootstrap intervals), ElecVQA-Bench test item accuracy and risk, latency for each image (4 questions) |
| SmolVLM | All 7,057 ElecVQA-Bench test items | Item accuracy (with a 95% bootstrap interval), risk, accuracy for each task, latency for each item (1 question) |

Both families record:

- the size on disk,
- the weight memory,
- the GPU memory after the load and the peak GPU memory during the evaluation,
- the number of predictions that are not finite.

`scripts/quant_table.py` writes `results/quant/q4_table.json` and `results/quant/q4_table.md`.

## 6. Scripts

- `scripts/quant_laya.py --model <base|good300_full>`
- `scripts/quant_smolvlm.py --model <base|lora>`
- `scripts/quant_table.py`
- `scripts/runs/q4_*.sh`: one script for each of the 4 runs.

## 7. GPU time (estimate)

A laya-vision run has 7,592 images, thus about 10 to 15 minutes. A SmolVLM run has 7,057 items, thus about 10 to 25 minutes. The 4 runs take about 1 to 1.5 hours on the RTX 2080 Ti.
