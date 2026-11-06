# Laya-vision fine-tuning, round 2: results

Date: 2026-11-07. Runs: `good300_lastn` and `good300_full`, student07, RTX 2080 Ti, fp32, commit ad37c82. The design is in `docs/vision_finetune_round2.md`. All values are in `results/vision/<run>/summary.json`.

## 1. Summary

1. Round 2 improves each metric against round 1. The full fine-tune is the best model.
2. The full model has a condition accuracy of 0.952 on the test images. This value is higher than the majority baseline (0.901).
3. On the test items of step 2, the full model has an accuracy of 0.948. The fine-tuned ResNet-50 has 0.961, and SmolVLM with LoRA has 0.956.
4. The latency does not change: 74 ms for each image.

## 2. Test towers, images (6,092), without the class-share correction

| Metric | Zero-shot | Round 1 | Round 2, last_n | Round 2, full | 95% interval (full) |
|---|---|---|---|---|---|
| condition accuracy | 0.834 | 0.641 | 0.924 | 0.952 | 0.947 to 0.957 |
| condition balanced accuracy | 0.181 | 0.592 | 0.705 | 0.734 | 0.715 to 0.752 |
| condition macro F1 | 0.180 | 0.303 | 0.637 | 0.703 | 0.685 to 0.720 |
| component accuracy | 0.141 | 0.465 | 0.789 | 0.924 | 0.917 to 0.931 |
| component balanced accuracy | 0.145 | 0.316 | 0.775 | 0.864 | — |
| defect AUROC | 0.714 | 0.898 | 0.959 | 0.965 | 0.956 to 0.973 |
| defect average precision | 0.225 | 0.618 | 0.812 | 0.862 | — |
| defect ECE | 0.178 | 0.405 | 0.094 | 0.058 | — |
| defect risk for each 1,000 images (val threshold) | 678 | 356 | 195 | 179 | — |

- The full model is better than the last_n model for each metric with an interval. The intervals do not overlap for the condition accuracy, the macro F1 and the component accuracy. They overlap for the defect AUROC.
- The training data now has 44% defects (round 1: 79%). Thus the class-share correction now changes the results only a little. For the full model, it changes the condition accuracy from 0.952 to 0.957 and the ECE from 0.058 to 0.019.

## 3. Test items (7,057), the unit of step 2

| Model | Accuracy | Risk for each 1,000 items |
|---|---|---|
| Laya-vision zero-shot | 0.830 | 1,186 |
| Laya-vision round 1 | 0.702 | 414 |
| Laya-vision round 2, last_n | 0.923 | 255 |
| Laya-vision round 2, full | 0.948 | 232 |
| ResNet-50, fine-tuned (step 2) | 0.961 | 177 |
| SmolVLM-256M with LoRA (step 2) | 0.956 | — |

- Laya-vision: a T2 item uses P(defect) ≥ 0.5. A T3 item uses the best condition among the options of the item.
- The risk of the step-2 classifier comes from `docs/risk_weights.md`, with the same weights. The threshold 0.5 for T2 is not tuned. A tuned threshold can change the risk of laya-vision.

## 4. Full model, details

- Condition recall: good 0.965, rust 0.834, corrosion 0.818, missing cap 0.783, bird nest 1.000, other defect 0 (5 test images).
- The largest error: 152 good images get the answer "rust". Then 35 rust images get "good", and 28 missing-cap images get "good".
- Component recall is at least 0.87 for 10 of the 11 components. The spacer has a recall of 0.06 (17 test images, 18 training images).
- The defect AUROC is at least 0.92 for each component with more than 50 defect images. Values: glass insulator 0.920, shackle 0.921, lightning rod suspension 0.999, vari-grip clamp 1.000, yoke suspension 0.993.

## 5. Training observations

- The full model trained for 24 minutes, and the last_n model for 20 minutes. Peak GPU memory: 3.6 GB (full) and 1.8 GB (last_n).
- After step 1,600, most batches of the full model had a loss of 0 and the maximum reward (0.750). Thus the model became very confident on the training questions.
- The temperatures from the held-out images are high (choice 4.13, yes/no 6.17 for the full model). They make the probabilities smaller and give a low ECE.
- The full model does not overfit the towers. The test results are near the val results: condition accuracy 0.952 against 0.935, defect AUROC 0.965 against 0.968.

## 6. Next steps

1. Use the full model as the base for the federated step.
2. To get to the accuracy of the step-2 tools, train on more good images and add more passes. Also tune the T2 threshold on the val towers.
