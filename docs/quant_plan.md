# Plan: quantized laya-vision and SmolVLM-256M

Date: 2026-11-09. Branch: `vision-quant` (from `vision-fl`). Status: code complete, not run on the GPU. Short version: only `int8` and `nf4`, on a small sample, for a direction.

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

## 5. Evaluation (short version)

The runs use only the quantized formats `int8` and `nf4`. The reference is the earlier fp32 predictions for the same images or items. Thus the runs do not run fp32 again.

| Family | Data | Metrics |
|---|---|---|
| laya-vision | 300 test-tower images: 150 with a defect, 150 good (seed 0) | Condition accuracy, component accuracy, defect AUROC, agreement with fp32, mean change of P(defect), latency for each image (4 questions) |
| SmolVLM | 150 ElecVQA-Bench test items: 75 good, 75 not good (seed 0) | Item accuracy, agreement with fp32, values that are not finite, latency for each item (1 question) |

Both families also record the size on disk, the weight memory and the peak GPU memory. The fp32 latency of laya-vision comes from an RTX 2080 Ti. The fp32 latency of SmolVLM comes from the GV100. Thus run laya-vision on the RTX 2080 Ti and SmolVLM on the GV100.

## 6. Scripts

- `scripts/quant_laya.py --model <base|good300_full> --format <fp32|fp16|int8|nf4>`
- `scripts/quant_smolvlm.py --model <base|lora> --format <fp32|fp16|int8|nf4>`
- `scripts/quant_compare.py` writes `results/quant/compare.json`.
- `scripts/runs/q_*.sh`: one script for each of the 8 runs.

## 7. GPU time (estimate)

Each run takes about 1 to 2 minutes, with the conversion. On 2 GPUs, the 8 runs take about 5 minutes.
