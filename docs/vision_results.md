# Laya-vision zero-shot: results

Date: 2026-11-05. Run: student07, NVIDIA GeForce RTX 2080 Ti, fp32, commit 8c89c32. Data: 35,415 images. No answer was cut. The design is in `docs/vision_fl_design.md`. All values are in `results/vision/summary.json`.

## 1. Summary

1. Laya-vision does not classify the condition of the components without fine-tuning. Its condition answers are almost equal to the majority baseline ("always good").
2. The yes/no defect question has a weak signal (AUROC 0.71).
3. The model does not identify the component.
4. The latency is low: 72 ms for each image on the RTX 2080 Ti. The 4 questions cost almost the same time as 1 question.

## 2. Test towers (6,092 images, 9.9% with a defect)

| Question | Metric | Laya-vision | Majority baseline | 95% interval |
|---|---|---|---|---|
| condition | accuracy | 0.834 | 0.901 | 0.824 to 0.843 |
| condition | balanced accuracy | 0.181 | 0.167 | 0.173 to 0.189 |
| condition | macro F1 | 0.180 | 0.158 | 0.172 to 0.190 |
| component | accuracy | 0.142 | 0.314 | 0.133 to 0.151 |
| component | balanced accuracy | 0.145 | 0.091 | — |
| defect | AUROC | 0.714 | 0.5 | 0.692 to 0.737 |
| defect | average precision | 0.225 | 0.099 | — |

The validation and train towers give almost the same values (condition macro F1 0.20 and 0.18, defect AUROC 0.71 and 0.70).

## 3. Condition

- Recall for each class: good 0.916, rust 0.146, corrosion 0.009, missing cap 0.016, bird nest 0, other defect 0.
- Of the 601 defect images, the model gives "good" for 480 images.
- The model gives "rust" for 321 good images.
- A defect answer from the condition question finds 20% of the defects (precision 0.21).

## 4. Defect (yes/no)

- At the threshold 0.5: recall 0.36, precision 0.26, false-alarm rate 0.11.
- At the threshold from the validation towers (0.19): recall 0.80, precision 0.14, false-alarm rate 0.53.
- Risk for each 1,000 images at this threshold: 678. "Always good" gives 987.
- The ECE is 0.18. Thus the probabilities are not calibrated for this data.
- The defect answer and the condition answer agree on 80% of the images.
- The AUROC is high for vari-grip clamps (0.88), lightning rod suspensions (0.84) and polymer insulators (0.85). It is low for glass insulators (0.56) and shackles (0.60).

## 5. Component

- The model gives "lightning rod suspension" for 2,237 images and "shackle" for 2,113 images. The true numbers are 210 and 977.
- Recall for yoke suspension (1,911 images) is 0. Recall for stockbridge damper (932 images) is 0.03.
- The best class is glass insulator (recall 0.32, precision 0.71).

## 6. Image quality (judgeable)

The AUROC of P(judgeable) for a correct condition answer is 0.53. Thus this answer has no value as a confidence signal.

## 7. Latency

| Device | Questions | Median for each image | p95 |
|---|---|---|---|
| RTX 2080 Ti, fp32 | 4 | 72.0 ms | 98.5 ms |
| RTX 2080 Ti, fp32 | condition only | 71.4 ms | 81.5 ms |
| student07 CPU, fp32 | 4 | 2,529 ms | 2,641 ms |
| student07 CPU, fp32 | condition only | 2,102 ms | 2,243 ms |

- The times include the read of the image from the disk and the preprocessing.
- The full run (35,415 images) gives the same median: 72.2 ms.
- Peak GPU memory: 880 MB.
- One 4 s segment has 3.45 key frames. Thus the GPU time for each segment is 0.25 s.

Comparison with EVLM (Jetson):

| Value | Time for each 4 s segment |
|---|---|
| EVLM, mission intent (Table V) | 6.448 s |
| EVLM, intents that are not scene description (Fig. 8a) | 1.4 s to 2.3 s |
| Laya-vision, RTX 2080 Ti (measured) | 0.25 s |
| Laya-vision, Jetson (estimate: 9 × the GPU time) | 2.2 s |

Limits of the comparison:

1. The run used an RTX 2080 Ti, not the GV100. The factor 9 came from the GV100. The fp32 speed of the two GPUs is similar, so the estimate stays near 2 s. It is not a measurement.
2. EVLM is fine-tuned and generates text. Laya-vision is not fine-tuned and gives only labels.
3. EVLM uses video segments. This run uses single crops.

## 8. Conclusion

- Zero-shot laya-vision is fast enough for the drone (estimate), but it is not accurate on this data. Its condition answers are not better than "always good".
- The defect question has some signal (AUROC 0.71). This signal is the best part to start from.
- A useful onboard model needs fine-tuning on the inspection data. The comparison values from step 2 show this: the fine-tuned ResNet-50 has 0.961 item accuracy, and SmolVLM with LoRA has 0.956.
