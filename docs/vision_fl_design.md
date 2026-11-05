# Laya-vision zero-shot classification (branch `vision-fl`)

Date: 2026-11-05. Status: code complete, not run on the GPU.

## 1. Idea

Laya-vision looks at each image of the dataset and answers 4 typed questions about it. The model is not fine-tuned. The pipeline has no routing and no other model.

## 2. Model

- Checkpoint: `thaitea/laya-vision`, revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`.
- Code: `github.com/r33drichards/laya-vision`, commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`.
- Size: 201M parameters. The backbone is SmolVLM-256M with 20 of its 30 language layers.
- Output: one probability for each option, from one forward pass. The model does not generate text.
- Precision: fp32.
- Licence: the code is Apache 2.0. The weights are CC BY-NC-SA 4.0. Thus the weights are for research only.
- The model needs `transformers==5.17.0`. Version 5.22 gives an error in the image features.
- The Python package of laya-vision has the name `laya`. It replaces the text Laya on this branch. Thus the scripts that use the text Laya do not run on this branch.

## 3. Input and questions

The input is the image only. The script calls `agent.predict` one time for each image, with 4 questions:

```python
agent.predict(
    {"image": image},
    {
        "component": {"type": "choice", "instructions": "Which power line component is in this image?",
                      "criteria": ["stockbridge damper", "preformed damper", "glass insulator", "polymer insulator",
                                   "shackle", "lightning rod suspension", "vari-grip clamp", "yoke",
                                   "yoke suspension", "spacer", "plate"]},
        "defect":    {"type": "noul", "instructions": "Does this power line component have a defect?"},
        "condition": {"type": "choice", "instructions": "Which condition does this power line component have?",
                      "criteria": ["good, no defect", "rust", "corrosion", "missing cap", "bird nest",
                                   "other defect (torn part or peeling paint)"]},
        "judgeable": {"type": "noul",
                      "instructions": "Is the image clear enough to judge the condition of the component?"},
    },
)
```

- The model encodes the image one time and uses it for the 4 questions.
- The probabilities use the temperatures of the checkpoint.
- No question is too long. The script records each answer that the model cuts (`truncated`).

## 4. Data and ground truth

- The unit is one image. ElecVQA-Bench has 39,228 items on 35,415 images. Some images have a T2 item and a T3 item.
- Ground truth for each image:
  - Component: the component folder of the image. The 7 shackle types become "shackle". Thus there are 11 components.
  - Condition: the class folder of the image (for example `rust`, `corrosão`, `missingcap`). Torn-up and peeling-paint become "other defect". This label exists for all 35,415 images, also for the images with only a T2 item.
  - Defect: the condition is not "good".
- The T2 and T3 labels of ElecVQA-Bench agree with the class folder on all images.
- The model is not trained. Thus each split is a test set. The main result is the test split (14 towers), for comparison with the other experiments.

## 5. Evaluation

The script `scripts/vision_summarize.py` compares the answers with the ground truth.

| Question | Metrics |
|---|---|
| `component` | Accuracy, balanced accuracy, macro F1, precision, recall and F1 for each class, confusion matrix |
| `condition` | The same metrics as `component` |
| `defect` | AUROC, average precision, expected calibration error (ECE). Precision, recall, F1, false-alarm rate and risk at the threshold 0.5 and at the threshold from the validation towers |
| `judgeable` | No ground truth. AUROC of P(judgeable) for a correct `condition` answer: a test of the value as a confidence signal |

Other results:

- Majority baseline for `component` and `condition`. The data is not balanced: 90% of the test images are "good". Thus the accuracy alone is not a good metric. Use the balanced accuracy and the macro F1.
- Defect from `condition`: a defect answer is a condition that is not "good". The script compares this answer with `defect` (agreement).
- Risk per 1,000 images: 10 for each missed defect, 1 for each false alarm (`docs/risk_weights.md`).
- The threshold for `defect` comes from the validation towers, with the lowest risk. The script applies it to the test towers.
- 95% bootstrap intervals on the test split, from 1,000 samples. The intervals are for these metrics: accuracy, macro F1 and balanced accuracy of `condition`. Accuracy of `component`. AUROC and recall of `defect`.
- Results for each component on the test split.
- A test with synthetic answers gave the expected values (for example, condition accuracy 0.832 against 0.833).

Values from step 2 on the test items, for comparison only (these models are not in this pipeline):

| Model | Accuracy | Fine-tuned |
|---|---|---|
| ResNet-50 classifier | 0.961 | yes |
| SmolVLM-256M with LoRA | 0.956 | yes |
| SmolVLM-256M base | 0.830 | no |
| Qwen3-VL-8B | 0.792 | no |

## 6. Latency

### 6.1 Measurement

- The script measures the time of each `agent.predict` call, one image at a time (batch 1), as on a drone. The time includes the read of the image from the disk, the preprocessing and the model.
- The full run records the time of each of the 35,415 images.
- After the full run, the script measures 500 test images again with 2 sets of questions: all 4 questions, and `condition` only. The first 10 images are a warm-up and the script does not count them.
- The script gives the mean, the median, p90, p95 and the maximum, in ms. It also records the peak GPU memory.
- A second run measures 100 images on the CPU of student07 (`--device cpu --latency-only`).

### 6.2 Comparison with EVLM

EVLM gives these values on an NVIDIA Jetson (8-core CPU, 16 GB, 1024 CUDA cores):

| EVLM value | Source |
|---|---|
| 6.448 s for each 4 s segment (histogram filter and VLM generation) | Table V |
| 1.4 s to 2.3 s inference time for each segment, for the intents that are not scene description | Fig. 8a |
| 1.9 ms for each frame for the histogram filter | Table IV |
| 3.45 key frames for each 4 s segment | Table V |

The summary script calculates:

- Segment time (measured) = 3.45 × the median time for one image, for all 4 questions and for `condition` only.
- Segment time on a Jetson (estimate) = 9 × the segment time on the GV100. The factor 9 comes from step 3. No Jetson measurement exists. Thus this value is an estimate.

Differences that the comparison must show:

1. The hardware is different. The GV100 is much faster than a Jetson.
2. EVLM generates text. Laya-vision does one forward pass and gives a label. It gives no description and no structured report.
3. EVLM measures video segments. This experiment measures single crops from ElecVQA-Bench.
4. The paper does not say if EVLM runs the VLM on each key frame. Thus the segment time of laya-vision assumes that it runs on each of the 3.45 key frames.

## 7. Files

- `src/fedorch/vision.py`: questions, ground truth, prediction, latency.
- `scripts/vision_run.py`: prediction for all images (`--limit` for a short test). After a stop, the script continues from the last image in `predictions.jsonl`.
- `scripts/vision_run_gpu.sh`: the full run and the CPU latency run on student07.
- `scripts/vision_summarize.py`: the evaluation on the laptop. It writes `results/vision/summary.json`.
