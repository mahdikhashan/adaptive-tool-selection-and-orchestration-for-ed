# Results: federated round E

Date: 2026-11-08. Status: all 18 runs are complete. The plan is `docs/vision_fl_round_e_plan.md`. All runs start from the server model `fl_server_central`, use the 5 fleet towers and send int8 updates. All values are on the test towers (6,092 images). Each run has one seed. The intervals are 95% bootstrap intervals.

## 1. Round E1: update size (5 fleet drones, 2,553 images, 10 rounds)

| Run | Uplink of the run (MB) | Macro F1 | Macro F1, class-share correction | Component accuracy | Defect AUROC | Risk for each 1,000 images |
|---|---|---|---|---|---|---|
| Server model alone | 0 | 0.558 (0.532 to 0.584) | 0.556 | 0.894 | 0.903 (0.887 to 0.919) | 291 |
| `fl_fleet_warm_k5` (full, fp16, round A) | 11,475 | 0.636 (0.612 to 0.657) | 0.657 | 0.912 | 0.953 (0.942 to 0.963) | 189 |
| `fl_e1_full` | 5,735 | 0.638 (0.615 to 0.659) | 0.664 | 0.916 | 0.962 (0.953 to 0.970) | 216 |
| `fl_e1_layers` | 3,965 | 0.604 (0.579 to 0.624) | 0.630 | 0.902 | 0.956 (0.946 to 0.966) | 211 |
| `fl_e1_last4` | 1,130 | 0.537 (0.514 to 0.558) | 0.585 | 0.893 | 0.948 (0.939 to 0.957) | 243 |
| `fl_e1_lora8` | 505 | 0.515 (0.491 to 0.536) | 0.573 | 0.887 | 0.949 (0.940 to 0.958) | 241 |
| `fl_e1_head` | 425 | 0.547 (0.519 to 0.571) | 0.562 | 0.895 | 0.940 (0.930 to 0.950) | 270 |

Findings:

1. The int8 full update gives the same result as the fp16 full update with half the uplink. Thus int8 has no measurable cost from the fine-tuned start.
2. All 5 updates increase the defect AUROC above the server model (0.903). The head update has the smallest uplink (425 MB) and gives an AUROC of 0.940.
3. The last-4 update and the LoRA update decrease the macro F1 below the server model. The class-share correction removes this decrease. The corrected values are 0.585 and 0.573, against 0.556 for the server model. Thus the cause is the class mix: the fleet images have 34% to 67% defects, but the test towers have 10% defects. These small updates move the model toward "defect". The recall of "good" falls from 0.95 to 0.87 or 0.88. The ECE increases from 0.063 to 0.14.
4. This explains the decrease of the validation accuracy in the logs. The validation images have about 90% "good", so the accuracy falls when the model gives "defect" more often. The full update and the 20-layer update do not have this problem.
5. Success criterion 1 has 3 limits: at most 1,150 MB, a macro F1 of at least 0.612 and an AUROC of at least 0.942. The last-4 update and the LoRA update satisfy the AUROC limit. They do not satisfy the macro F1 limit, also with the correction. Thus no update satisfies the criterion. The 20-layer update satisfies both limits with the correction, at 3,965 MB (35% of the fp16 full run).

## 2. Round E2: few natural labels (last-4 update, int8)

| Run | Labelled defects in the fleet | Macro F1 | Defect AUROC | Recall | False-alarm rate | Risk |
|---|---|---|---|---|---|---|
| Server model alone | 0 | 0.558 | 0.903 | 0.832 | 0.139 | 291 |
| Round C, fp16, 50 labels | 37 | 0.571 | 0.836 | 0.684 | 0.021 | 331 |
| `fl_e2_plain_n50` | 37 | 0.564 | 0.839 | 0.696 | 0.024 | 322 |
| `fl_e2_balance_n50` | 37 | 0.536 | 0.925 | 0.872 | 0.232 | 335 |
| `fl_e2_prox_n50` | 37 | 0.566 | 0.843 | 0.702 | 0.026 | 317 |
| `fl_e2_slr_n50` | 37 | 0.562 | 0.840 | 0.704 | 0.027 | 317 |
| Round C, fp16, 200 labels | 149 | 0.575 | 0.933 | 0.854 | 0.150 | 280 |
| `fl_e2_plain_n200` | 149 | 0.580 | 0.938 | 0.834 | 0.114 | 267 |
| `fl_e2_balance_n200` | 149 | 0.523 | 0.944 | 0.862 | 0.123 | 247 |
| `fl_e2_prox_n200` | 149 | 0.579 | 0.926 | 0.860 | 0.163 | 285 |
| `fl_e2_slr_n200` | 149 | 0.580 | 0.921 | 0.855 | 0.166 | 292 |
| Same 200 labels on one server | 149 | 0.623 | 0.958 | 0.870 | 0.059 | 181 |

Findings:

1. The plain int8 runs give the same result as the fp16 runs of round C with half the uplink.
2. With 50 labels, only balanced sampling keeps the defect AUROC above the server model (0.925). Thus this run satisfies success criterion 2. The false-alarm rate increases to 0.232, and the risk does not decrease.
3. FedProx 0.1 and the server step 0.5 do not change the result with 50 labels. With 200 labels, they decrease the AUROC a little.
4. With 200 labels, balanced sampling gives the highest AUROC (0.944) and the lowest risk (247) of the federated runs. It does not reach the central result (0.950 to 0.965). Thus success criterion 3 is not satisfied.

## 3. Round E3: different image types (last-4 update, int8, 10 rounds)

| Run | Drones | Uplink (MB) | Macro F1 | Defect AUROC | Corrosion | Missing cap | Bird nest |
|---|---|---|---|---|---|---|---|
| Server model alone | 0 | 0 | 0.558 | 0.903 | 0.63 | 0.16 | 0.90 |
| `fl_e1_last4` (tower) | 5 | 1,130 | 0.537 | 0.948 | 0.65 | 0.55 | 0.94 |
| `fl_e3_iid` (random) | 6 | 1,356 | 0.535 | 0.932 | 0.66 | 0.49 | 0.96 |
| `fl_e3_component` | 6 | 1,356 | 0.529 | 0.923 | 0.62 | 0.39 | 0.94 |
| `fl_e3_component_prox` | 6 | 1,356 | 0.533 | 0.922 | 0.65 | 0.36 | 0.94 |
| `fl_e3_dirichlet` | 6 | 1,356 | 0.560 | 0.956 | 0.71 | 0.55 | 0.94 |
| Central, all towers | — | — | 0.703 | 0.965 | 0.82 | 0.78 | 1.00 |

Findings:

1. From the fine-tuned start, the split has a small effect. The macro F1 of the random, component and component-with-FedProx splits is in the same interval (0.529 to 0.535).
2. The Dirichlet split gives the best result (0.560, AUROC 0.956). Two large drones with balanced condition shares hold 85% of its images. This can be the cause, but the experiment does not show it.
3. After FedAvg with the component split, each defect type that only one drone has keeps at least half of its central recall. The values are corrosion 0.62 (limit 0.41), bird nest 0.94 (limit 0.50) and missing cap 0.39 (limit 0.39). Thus the run satisfies success criterion 4, but only just for missing cap. From the published checkpoint (round D), FedAvg lost all 3 types.
4. The 6 drones that train alone have a mean macro F1 of 0.481. Their mean defect AUROC is 0.895.

Figures: `figs/fig_round_e1.pdf`, `figs/fig2_uplink_all.pdf`, `figs/fig2_rounde3.pdf`.
