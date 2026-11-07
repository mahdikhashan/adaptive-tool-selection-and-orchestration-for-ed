# Results: federated round E (part 1)

Date: 2026-11-07. Status: 9 of 18 runs are complete. The plan is `docs/vision_fl_round_e_plan.md`. All runs start from the server model `fl_server_central`, use the 5 fleet towers and send int8 updates. All values are on the test towers (6,092 images). Each run has one seed. The intervals are 95% bootstrap intervals.

## 1. Round E1: update size (5 fleet drones, 2,553 images, 10 rounds)

| Run | Uplink of the run (MB) | Macro F1 | Macro F1, class-share correction | Component accuracy | Defect AUROC | Risk for each 1,000 images |
|---|---|---|---|---|---|---|
| Server model alone | 0 | 0.558 (0.532 to 0.584) | 0.556 | 0.894 | 0.903 (0.887 to 0.919) | 291 |
| `fl_fleet_warm_k5` (full, fp16, round A) | 11,475 | 0.636 (0.612 to 0.657) | 0.657 | 0.912 | 0.953 (0.942 to 0.963) | 189 |
| `fl_e1_full` | 5,735 | 0.638 (0.615 to 0.659) | 0.664 | 0.916 | 0.962 (0.953 to 0.970) | 216 |
| `fl_e1_layers` | 3,965 | 0.604 (0.579 to 0.624) | 0.630 | 0.902 | 0.956 (0.946 to 0.966) | 211 |
| `fl_e1_last4` | 1,130 | 0.537 (0.514 to 0.558) | 0.585 | 0.893 | 0.948 (0.939 to 0.957) | 243 |
| `fl_e1_lora8` | 505 | 0.515 (0.491 to 0.536) | 0.573 | 0.887 | 0.949 (0.940 to 0.958) | 241 |

Findings:

1. The int8 full update gives the same result as the fp16 full update with half the uplink. Thus int8 has no measurable cost from the fine-tuned start.
2. All 4 updates increase the defect AUROC above the server model (0.903). The AUROC of each update is in the interval of the fp16 full run.
3. The last-4 update and the LoRA update decrease the macro F1 below the server model. The class-share correction removes this decrease. The corrected values are 0.585 and 0.573, against 0.556 for the server model. Thus the cause is the class mix: the fleet images have 34% to 67% defects, but the test towers have 10% defects. These small updates move the model toward "defect". The recall of "good" falls from 0.95 to 0.87 or 0.88. The ECE increases from 0.063 to 0.14.
4. This explains the decrease of the validation accuracy in the logs. The validation images have about 90% "good", so the accuracy falls when the model gives "defect" more often. The full update and the 20-layer update do not have this problem.
5. Success criterion 1 has 3 limits: at most 1,150 MB, a macro F1 of at least 0.612 and an AUROC of at least 0.942. The last-4 update and the LoRA update satisfy the AUROC limit. They do not satisfy the macro F1 limit, also with the correction. Thus no update satisfies the criterion. The 20-layer update satisfies both limits with the correction, at 3,965 MB (35% of the fp16 full run).

## 2. Round E2: few natural labels (last-4 update, int8)

| Run | Labels for each drone | Macro F1 | Defect AUROC | Recall | False-alarm rate | Risk |
|---|---|---|---|---|---|---|
| Server model alone | 0 | 0.558 | 0.903 | 0.832 | 0.139 | 291 |
| Round C, fp16, no change | 50 | 0.571 | 0.836 | 0.684 | 0.021 | 331 |
| Round C, fp16, no change | 200 | 0.575 | 0.933 | 0.854 | 0.150 | 280 |
| `fl_e2_balance_n50` | 50 | 0.536 | 0.925 (0.913 to 0.937) | 0.872 | 0.232 | 335 |
| `fl_e2_balance_n200` | 200 | 0.523 | 0.944 (0.933 to 0.954) | 0.862 | 0.123 | 247 |
| `fl_e2_prox_n200` | 200 | 0.579 | 0.926 (0.912 to 0.938) | 0.860 | 0.163 | 285 |
| `fl_e2_slr_n200` | 200 | 0.580 | 0.921 (0.907 to 0.935) | 0.855 | 0.166 | 292 |

Findings:

1. With 50 labels, balanced sampling keeps the defect AUROC above the server model (0.925 against 0.903). Without it, round C gave 0.836. Thus this run satisfies success criterion 2. But the false-alarm rate increases to 0.232, and the macro F1 decreases to 0.536.
2. With 200 labels, balanced sampling gives the highest AUROC (0.944), but not the central result with the same labels (0.950 to 0.965). Success criterion 3 is not satisfied.
3. FedProx 0.1 and the server step 0.5 keep the macro F1 of the server model. They do not increase the AUROC above round C. The plain int8 runs (`fl_e2_plain_*`) are not complete. Thus the effect of int8 alone on these runs is not known yet.
4. The class-share correction does not correct balanced sampling, because it uses the label shares and not the sampled shares (50% defects).

## 3. Round E3: drones alone (component split, last-4 update, int8)

The 6 drones that train alone have a mean macro F1 of 0.481 (0.318 to 0.565). Their mean defect AUROC is 0.895 (0.827 to 0.931). The FedAvg runs of round E3 are not complete.

## 4. Runs that are not complete

`fl_e1_head`, `fl_e2_plain_n50`, `fl_e2_plain_n200`, `fl_e2_prox_n50`, `fl_e2_slr_n50`, `fl_e3_component`, `fl_e3_component_prox`, `fl_e3_dirichlet`, `fl_e3_iid`.

Figure: `figs/fig_round_e1.pdf`.
