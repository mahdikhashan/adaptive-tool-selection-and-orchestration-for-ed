# Results: 4-bit laya-vision and SmolVLM-256M

Date: 2026-11-09. Plan: `docs/quant_plan.md`. GPU: RTX 2080 Ti. Format: nf4 weights, fp16 computation. Table: `results/quant/q4_table.md`. Figures: `figs/fig_q4_overview.pdf`, `figs/fig_q4_details.pdf`.

## 1. Results

| Model | Size on disk (MB) | Weights (MB) | GPU after load (MB) | Peak GPU (MB) | Test item accuracy | Median latency (ms) | p95 latency (ms) |
|---|---|---|---|---|---|---|---|
| laya-vision base | 189 | 175 | 187 | 222 | 0.812 | 83.1 (image, 4 questions) | 114.5 |
| laya-vision fine-tuned | 189 | 175 | 186 | 218 | 0.939 | 82.6 (image, 4 questions) | 119.7 |
| SmolVLM base | 232 | 216 | 230 | 262 | 0.846 (0.838 to 0.855) | 77.4 (item, 1 question) | 94.3 |
| SmolVLM LoRA | 232 | 216 | 341 | 373 | 0.865 (0.856 to 0.873) | 75.8 (item, 1 question) | 94.7 |

laya-vision on the 6,092 test-tower images:

| Model | Condition macro F1 | Component accuracy | Defect AUROC | Risk at the validation threshold |
|---|---|---|---|---|
| base | 0.180 (0.170 to 0.191) | 0.135 | 0.701 (0.677 to 0.724) | 717 |
| fine-tuned | 0.668 (0.647 to 0.689) | 0.870 | 0.967 (0.959 to 0.975) | 180 |

No prediction has a value that is not finite.

## 2. Data types

- laya-vision: 213 linear layers of the backbone are 4-bit. The other backbone values are fp16, except the rotary position frequencies (fp32 buffers). The decision head (34 MB) is fp32, because the laya library changes its input to fp32.
- SmolVLM: 283 linear layers are 4-bit. `lm_head` and the other values are fp16, except the rotary position frequencies (64 fp32 values).

## 3. Findings

1. Size: a 4-bit model needs 189 MB (laya-vision) or 232 MB (SmolVLM) on disk. At fp32, the weights are about 0.8 GB (201M parameters) and 1.0 GB (256M parameters). The peak GPU memory is less than 400 MB for all 4 models.
2. Latency: 4-bit does not make the models faster on the RTX 2080 Ti at batch size 1. The earlier fp32 run of fine-tuned laya-vision on an RTX 2080 Ti had a median of 74.0 ms. The 4-bit run has 82.6 ms. The SmolVLM fp32 latency came from a different GPU (GV100), thus we do not compare it.
3. laya-vision fine-tuned keeps most of its accuracy in 4-bit. The earlier fp32 values were item accuracy 0.948, macro F1 0.703, component accuracy 0.924 and defect AUROC 0.965. The 4-bit values are 0.939, 0.668, 0.870 and 0.967. Defect detection does not change. The macro F1 interval (0.647 to 0.689) does not include the fp32 value, thus the condition and component answers become a little worse.
4. SmolVLM LoRA loses most of its fine-tuning in 4-bit. The earlier fp32 item accuracy was 0.956 (base 0.830). The 4-bit item accuracy is 0.865 (base 0.846). The 4-bit LoRA model answers "good" for 98.4% of the items, but 84.9% of the items are good. On the defect question (T2), the accuracy is 0.696 against 0.928 in fp32.
5. A possible cause (not tested): the LoRA changes are small, and the 4-bit conversion of the merged weights rounds them back near the base weights. laya-vision had a full fine-tune with larger changes. A test: keep the LoRA adapter in fp16 on top of the 4-bit base model, as in QLoRA, without a merge.
6. The GPU memory after load of SmolVLM LoRA (341 MB) is larger than SmolVLM base (230 MB), although the stored weights have the same size. The conversion ran in the same process before the evaluation. Thus memory from the conversion can stay allocated. Run `scripts/runs/q4_smolvlm_lora.sh` again to measure it: the run has all predictions, thus it only loads the model.
