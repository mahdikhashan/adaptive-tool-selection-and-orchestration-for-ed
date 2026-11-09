# Plan: quantized laya-vision and SmolVLM-256M

Date: 2026-11-09. Branch: `vision-quant` (from `vision-fl`). Status: code complete, not run on the GPU.

## 1. Goal

Measure the accuracy, the latency and the memory of 2 models in 4 numeric formats. Keep each quantized model on the server, so that we can push it to Hugging Face later.

## 2. Models

| Family | Model | Source |
|---|---|---|
| laya-vision | `base` | `thaitea/laya-vision`, revision `f2fe3c1` |
| laya-vision | `good300_full` | Our fine-tuned model, `$FEDORCH_WORK/models/vision/good300_full` |
| SmolVLM-256M | `base` | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| SmolVLM-256M | `lora` | The step 2 LoRA adapter `$FEDORCH_WORK/runs/step2/smolvlm_lora_full`, merged into the base weights |

## 3. Formats

| Format | Weights | Computation |
|---|---|---|
| `fp32` | 32-bit float | fp32 (reference) |
| `fp16` | 16-bit float | fp16 |
| `int8` | bitsandbytes LLM.int8 | int8 matrix multiplication, outliers in fp16 |
| `nf4` | bitsandbytes 4-bit NormalFloat | fp32 |

The 4-bit format uses fp32 computation, because the GPUs have no bfloat16, and fp16 can give values that are not finite.

For laya-vision, the format applies to the backbone (vision encoder, connector, 20 language layers). The decision head stays in fp32, except in the `fp16` format. For SmolVLM, the format applies to all linear layers except `lm_head`.

## 4. Storage

Each model goes to `$FEDORCH_WORK/models/quantized/<family>_<model>_<format>/`:

- laya-vision: `vlm_agent_config.json`, `processor/`, `head.safetensors` and `backbone_quantized/` (a Hugging Face model folder).
- SmolVLM: a Hugging Face model folder with the processor.

Each run loads the model again from this folder before the evaluation. Thus the evaluation tests the stored model.

## 5. Evaluation

| Family | Data | Metrics |
|---|---|---|
| laya-vision | 1,500 validation images and the 6,092 test-tower images | Condition macro F1, component accuracy, defect AUROC, ElecVQA-Bench test item accuracy and risk, latency for each image (4 questions) |
| SmolVLM | The 7,057 ElecVQA-Bench test items | Item accuracy, risk, values that are not finite, latency for each item (1 question) |

Both families also record the size on disk, the weight memory and the peak GPU memory. The item accuracy is the common metric. The fp32 runs must give the earlier values: 0.948 (laya-vision fine-tuned), 0.956 (SmolVLM with LoRA) and 0.830 (SmolVLM base).

## 6. Scripts

- `scripts/quant_laya.py --model <base|good300_full> --format <fp32|fp16|int8|nf4>`
- `scripts/quant_smolvlm.py --model <base|lora> --format <fp32|fp16|int8|nf4>`
- `scripts/quant_summarize.py` writes `results/quant/summary.json` and prints a table.
- `scripts/runs/q_*.sh`: one script for each of the 16 runs.

## 7. GPU time (estimate)

A laya-vision run has 7,592 images at about 75 ms, thus about 10 minutes, plus the latency test. A SmolVLM run has 7,057 items, thus about 10 to 20 minutes. The 16 runs need about 4 GPU-hours. The 4-bit format can be slower than fp32.
