# Laya-vision fine-tuning, round 2 (branch `vision-fl`)

Date: 2026-11-06. Status: code complete, not run on the GPU.

## 1. Starting point

The first fine-tuned model (`lastn_pc400`, `docs/vision_finetune_results.md`) has 3 problems:

1. It reports too many defects, because the training data had about 79% defects.
2. It never gives 4 of the 11 components, because the training selection did not control the components.
3. It used only 1,793 images, 8% of the train-tower images.

## 2. Fix after the prediction: class-share correction

`scripts/vision_summarize.py` now also gives the results after a correction of the class shares. For each question, the script multiplies the probability of each class by the share of the class in the val towers. Then it divides the result by the share in the training images. Then it makes the sum of the probabilities equal to 1. The shares come from the labels, with 1 added to each count. The correction needs no new training.

Result for `lastn_pc400` on the test towers:

| Metric | Before the correction | After the correction | 95% interval (after) |
|---|---|---|---|
| condition accuracy | 0.642 | 0.856 | 0.847 to 0.865 |
| condition macro F1 | 0.303 | 0.451 | 0.427 to 0.474 |
| condition balanced accuracy | 0.592 | 0.506 | 0.479 to 0.534 |
| component accuracy | 0.465 | 0.526 | 0.513 to 0.538 |
| defect ECE | 0.405 | 0.090 | — |
| defect AUROC | 0.898 | 0.898 | 0.884 to 0.910 |

- The correction moves answers from the defect classes to "good". Thus the accuracy and the macro F1 increase, and the recall of the defect classes decreases.
- The correction does not change the order of the defect scores. Thus the AUROC and the risk at the val threshold do not change.

## 3. Round 2: more data and more trained layers

### 3.1 Data

- All defect images of the train towers: 2,249.
- Good images: up to 300 for each component, at random (seed 0): 2,858. Plate has only 140 good images, and spacer has 18.
- Total: 5,107 images, 44% with a defect. Each of the 11 components has at least 18 images, and 9 components have at least 300.
- The script keeps 300 images (900 questions) out of the training, for the temperatures. The training has 4,807 images (14,421 questions).

### 3.2 Two models

| Model | `freeze` | Trained parameters | Reason |
|---|---|---|---|
| `good300_lastn` | `last_n` (head and last 4 language layers) | 22.6M | Same training as `lastn_pc400`. Thus it shows the effect of the data. |
| `good300_full` | `full` (all parameters except the vision encoder) | more | The fine-tune guide gives `full` for "many thousands of questions, on a GPU" |

All other settings are the same as in `lastn_pc400`: 2.5 passes (about 4,500 steps), batch 8, learning rate 1e-4 and 2e-5, warm-up 25 steps, fp32.

### 3.3 Evaluation

The evaluation is the same as for `lastn_pc400`, with and without the class-share correction. The main comparison is the test towers.

## 4. Expected results

- `good300_lastn` against `lastn_pc400`: a higher component accuracy, because each component has training images.
- `good300_full` against `good300_lastn`: a higher accuracy, if the model can use the more trained parameters with this amount of data.
- The 2 models are 2 options of one decision. The better model on the val towers is the base for the next step.
