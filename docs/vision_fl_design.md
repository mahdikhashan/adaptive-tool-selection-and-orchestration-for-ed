# Laya-vision zero-shot classification (branch `vision-fl`)

Date: 2026-11-05. Status: code complete, not run on the GPU.

## 1. Idea

Laya-vision looks at each image of the dataset and classifies it. The model is not fine-tuned. The pipeline has no routing and no other model.

## 2. Model

- Checkpoint: `thaitea/laya-vision`, revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`.
- Code: `github.com/r33drichards/laya-vision`, commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`.
- Size: 201M parameters. The backbone is SmolVLM-256M with 20 of its 30 language layers.
- Output: one probability for each option, from one forward pass. The model does not generate text.
- Precision: fp32.
- Licence: the code is Apache 2.0. The weights are CC BY-NC-SA 4.0. Thus the weights are for research only.
- The model needs `transformers==5.17.0`. Version 5.22 gives an error in the image features.
- The Python package of laya-vision has the name `laya`. It replaces the text Laya on this branch. Thus the scripts that use the text Laya do not run on this branch.

## 3. Input and output

- Input: the image only. The input has no text about the component and no output of a different model.
- Laya-vision needs a question and a set of options. The question is the same for all images of one task:
  - T2: "Does the power line component in this image have a defect?"
  - T3: "Which condition does the power line component in this image have?"
- Options: the answer options of the dataset item, for example "good, no defect", "rust", "corrosion", "cannot judge".
- Prediction: the option with the highest probability.

## 4. Data

All 39,228 ElecVQA-Bench items (train 25,499, val 6,672, test 7,057). The model is not trained, so each split is a test set. The main result is the test split, for comparison with the other experiments.

## 5. Metrics

- Accuracy, overall and for each task (T2, T3).
- Recall for each label.
- Defect recall: the part of the defect frames with a defect answer.
- Abstention rate: the part of the answers that are "cannot judge".
- Risk per 1,000 frames, with the weights in `docs/risk_weights.md` (miss 10, false alarm 1, abstention 0.5).
- Time for each item on the GV100.

Values from step 2 on the test split, for comparison only (these models are not in this pipeline):

| Model | Accuracy | Fine-tuned |
|---|---|---|
| ResNet-50 classifier | 0.961 | yes |
| SmolVLM-256M with LoRA | 0.956 | yes |
| SmolVLM-256M base | 0.830 | no |
| Qwen3-VL-8B | 0.792 | no |

## 6. Files

- `src/fedorch/vision.py`: question, options and prediction.
- `scripts/vision_run.py`: prediction for all items (`--limit` for a short test).
- `scripts/vision_run_gpu.sh`: the full run on student07.
- `scripts/vision_summarize.py`: metrics on the laptop from `results/vision/zeroshot/scores.jsonl`.
