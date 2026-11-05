# Laya-vision fine-tuning: results

Date: 2026-11-06. Run: `lastn_pc400`, student07, RTX 2080 Ti, fp32, commit bb82c1c. The design is in `docs/vision_finetune.md`. All values are in `results/vision/lastn_pc400/summary.json`. The zero-shot values are in `results/vision/zeroshot/summary.json`.

## 1. Summary

1. Fine-tuning on 1,793 images (8% of the train-tower images) improves each metric that does not depend on the class shares.
2. The fine-tuned model reports too many defects. The training data has about 79% defects, and the test data has 10%. Thus the plain accuracy decreases.
3. The latency does not change.

## 2. Test towers (6,092 images)

| Metric | Zero-shot | Fine-tuned | 95% interval (fine-tuned) |
|---|---|---|---|
| condition accuracy | 0.834 | 0.642 | 0.630 to 0.653 |
| condition balanced accuracy | 0.181 | 0.592 | 0.567 to 0.618 |
| condition macro F1 | 0.180 | 0.303 | 0.288 to 0.321 |
| component accuracy | 0.142 | 0.465 | 0.453 to 0.477 |
| component balanced accuracy | 0.145 | 0.316 | — |
| defect AUROC | 0.714 | 0.898 | 0.884 to 0.910 |
| defect average precision | 0.225 | 0.618 | — |
| defect risk for each 1,000 images (threshold from val) | 678 | 356 | — |

The intervals of the 2 models do not overlap for the metrics in the table that have an interval.

## 3. Condition

- Recall: good 0.64, rust 0.55, corrosion 0.73, missing cap 0.74, bird nest 0.90, other defect 0 (5 test images).
- The model gives a defect class for 1,971 of the 5,491 good images. Most of these answers are "missing cap" (838) and "rust" (595).
- A defect answer from the condition question finds 89% of the defects (zero-shot: 20%). The precision is 0.21.

## 4. Defect (yes/no)

- The threshold from the val towers is 0.91. At this threshold: recall 0.80, precision 0.33, false-alarm rate 0.18.
- The ECE is 0.40 (zero-shot: 0.18). The temperatures came from the balanced training selection, so the probabilities are too high for the real class shares.
- The AUROC is at least 0.84 for each component with more than 50 defect images, except the glass insulator (0.75).

## 5. Component

- Recall: glass insulator 0.98, vari-grip clamp 0.80, shackle 0.59, yoke suspension 0.50.
- The model never gives preformed damper, yoke, spacer or plate. It gives polymer insulator for 11 images (510 true).
- The training selection has equal numbers for each condition, not for each component. This is a possible cause.

## 6. Other results

- The `judgeable` question was not trained. Its AUROC for a correct condition answer is now 0.29. Thus a high value now shows a wrong answer more often. Do not use this question.
- The training needed 7.2 minutes for 1,681 steps. Peak GPU memory: 1,755 MB.
- Latency: 73.1 ms median for each image with 4 questions (zero-shot: 72.0 ms).
- The val score in the training log decreased (accuracy 0.601 to 0.566 on 1,800 questions). This value mixes the 3 questions and uses the real class shares. Thus the decrease agrees with the lower plain accuracy above.

## 7. Next steps

1. Correct the class shares after the prediction: multiply each probability by the val share of its class, divided by the train share. This step needs no new training and runs on the laptop.
2. Train on a selection with equal numbers for each condition and for each component.
3. Train on more images, for example 800 for each condition, or all defect images.
