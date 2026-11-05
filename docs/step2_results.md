# Step 2 results: the tool cache

Date: 2026-11-05. Server: student07, GPU: Quadro GV100, fp16 for the cloud tool. Code: commit `3101ce9`.
Source files: `results/step2/summary.json` and `results/step2/tool_*.jsonl`.
The classifier and the SmolVLM LoRA are cross-fitted by tower. Each result comes from a model that did not train on the tower of the item.

## Tools on the 7,057 test-tower items

| Tool | Accuracy | Median compute (GV100) | Payload |
|---|---|---|---|
| `classifier` (ResNet-50) | 0.961 | 7.3 ms | 0 |
| `smolvlm_local` (LoRA) | 0.956 | 80.8 ms | 0 |
| `smolvlm_split` (INT8 tokens) | 0.957 | 80.2 ms | 36,868 B |
| `cloud` (Qwen3-VL-8B, zero-shot) | 0.792 | 126.9 ms | 35,421 B (JPEG) |
| `smolvlm_base` (zero-shot) | 0.830 | 61.4 ms | 0 |

The compute times are GV100 times. They are not Jetson times. The cloud time does not include the network.

## Findings

1. The classifier is the best single tool. It is also the fastest tool. This result agrees with ElecVQA-Bench: a vision backbone at a sufficient resolution equals the VLMs on these defects.
2. The tools make different errors. On the test towers, the classifier makes 279 errors. `smolvlm_local` gives the correct answer for 186 of these items (66.7%). `cloud` gives the correct answer for 135 (48.4%). One of the two tools is correct for 78.5% of the classifier errors.
3. The classifier confidence identifies its errors. The AUROC for error detection is 0.939. The calibration error (ECE, 5 bins) is 0.007.
4. A static-threshold cascade is a strong baseline. If the classifier confidence is below 0.8, the cascade sends the item to `smolvlm_local`. The cascade sends 6.2% of the test items. Its accuracy is 0.975. The classifier alone gets 0.961.
5. An oracle that selects the classifier or `smolvlm_local` for each item gets 0.987. With `cloud` as a third tool, the oracle gets 0.992. Thus a learned controller can add at most about 1.2 points over the static cascade with the two drone tools.
6. Most of the remaining gain is in the defect labels. The table below gives the recall on the test towers.

| Label | Items | Classifier | Oracle: classifier + local | Oracle: + cloud |
|---|---|---|---|---|
| good | 5,989 | 0.973 | 0.996 | 0.998 |
| defect (T2) | 534 | 0.906 | 0.957 | 0.989 |
| rust | 273 | 0.857 | 0.901 | 0.905 |
| missing-cap | 116 | 0.810 | 0.853 | 0.853 |
| corrosion | 100 | 0.960 | 0.990 | 0.990 |
| nest | 40 | 1.000 | 1.000 | 1.000 |
| other-defect | 5 | 0.000 | 0.600 | 1.000 |

7. The zero-shot cloud VLM is weaker than the drone tools. Its corrosion recall is 0.012 (all items). It also calls many good items defective. On the test towers, it is wrong on 1,322 items where the classifier is correct. But for the binary `defect` items, `cloud` increases the oracle recall from 0.957 to 0.989. Thus `cloud` is useful only for some items. The controller must learn which items.
8. INT8 visual tokens do not decrease the accuracy. `smolvlm_split` and `smolvlm_local` differ by less than 0.1 point on all items. This result answers the open accuracy question of the split paper for SmolVLM-256M.
9. At 512 px, the split payload (36,868 B) is almost equal to the JPEG payload (35,421 B). Thus the split saves almost no uplink bytes at this resolution.
10. SmolVLM-256M without fine-tuning answers "good" for almost all items. Its recall for each defect label is below 0.04. The EVLM fine-tuning is necessary.

## Consequences for the experiments

- The evaluation must use defect recall and the cost-weighted risk, not only accuracy. Accuracy hides the defect errors because 85% of the items are "good".
- The static-threshold cascade (classifier, then `smolvlm_local` below a confidence threshold) is a mandatory baseline.
- The headroom for a learned controller is small on accuracy (about 1 to 1.6 points). The headroom is larger on the defect labels (for example, T2 defect: 0.906 to 0.989).
