# Federated rounds B, C and D: results

Date: 2026-11-08. All 25 runs. The 2 round-C runs with 25 labels came later than the other runs. GPUs: RTX 2080 Ti, Quadro GV100, GTX 1080 Ti, fp32, commit 1269ab8. The plan is in `docs/vision_fl_round_b_plan.md`. All values are on the test towers (6,092 images), with one seed for each run. The values are in `results/vision/fl_{b,c,d}_*/summary.json`.

## 1. Summary

1. From the published checkpoint, the small updates (head, LoRA, last 4 layers) learn very little in 10 rounds. Only the updates with all language layers or the full model come near round A.
2. int8 updates and FedProx change nothing that the test can measure.
3. With 30 rounds, the last-4 update improves much (macro F1 0.281 to 0.488). Thus the small updates need more rounds, not a different method.
4. Round D shows no difference between the partitions. A drone that trains alone learns the defect type of its own components. FedAvg with the last-4 update loses these defect types for each partition. It also loses them for the random partition.
5. Round C: with few labels, only the fine-tuned start works. 25 or 50 labels for each drone give 18 or 37 defects in the fleet. With these labels, FedAvg decreases the defect AUROC of the server model (0.903 to 0.856 and 0.836). With 100 or 200 labels (69 or 149 defects), it increases the AUROC to 0.930 and 0.933. The same labels on one server give more (0.958) and never decrease the AUROC.

## 2. Round B: update size (6 tower drones, start from the published checkpoint)

| Run | Update (each drone and round) | Total uplink | Condition macro F1 | Component accuracy | Defect AUROC |
|---|---|---|---|---|---|
| Central `good300_full` (reference) | — | — | 0.703 | 0.924 | 0.965 |
| `fl_fedavg_k6`, full (round A) | 229.5 MB | 13.8 GB | 0.541 (0.513 to 0.570) | 0.761 | 0.939 |
| `fl_b_full_prox` | 229.5 MB | 13.8 GB | 0.526 (0.495 to 0.555) | 0.756 | 0.940 |
| `fl_b_layers` | 158.5 MB | 9.5 GB | 0.488 (0.460 to 0.514) | 0.707 | 0.931 |
| `fl_b_last4_r30` (30 rounds) | 45.2 MB | 8.1 GB | 0.488 (0.456 to 0.516) | 0.634 | 0.945 |
| `fl_b_last4` | 45.2 MB | 2.7 GB | 0.281 (0.258 to 0.304) | 0.321 | 0.838 |
| `fl_b_last4_prox` | 45.2 MB | 2.7 GB | 0.275 (0.257 to 0.295) | 0.331 | 0.834 |
| `fl_b_last4_int8` | 22.6 MB | 1.4 GB | 0.278 (0.257 to 0.301) | 0.320 | 0.838 |
| `fl_b_lora8` | 20.2 MB | 1.2 GB | 0.247 (0.232 to 0.262) | 0.307 | 0.828 |
| `fl_b_head` | 16.9 MB | 1.0 GB | 0.243 (0.229 to 0.256) | 0.293 | 0.792 |

Values in brackets are 95% bootstrap intervals. The script calculates the uplink. It does not measure it.

- The central model with the last-4 update (`good300_lastn`) has a macro F1 of 0.637. With FedAvg and 6 drones, the same update gives 0.281. With the full update, the loss is smaller (0.703 to 0.541).
- The val accuracy of the last-4 run is 0.657 after 10 rounds and 0.811 after 30 rounds. The curve still increases at round 30.
- int8 gives the same result as fp16 (macro F1 0.278 against 0.281, AUROC 0.838 against 0.838). Thus int8 halves the uplink with no measurable cost.
- FedProx (μ = 0.01) gives the same result as FedAvg for both updates.
- The update with all language layers but no embeddings and no connector loses 0.053 macro F1 against the full update (0.488 against 0.541). The intervals overlap.
- Possible cause of the slow small updates: in each round, the server takes the mean of 6 small changes. Thus the global model moves less in each round than a model that trains alone. The full model has more parameters that change, and it learns faster in each step. The 30-round run agrees with this explanation, but it does not prove it.

## 3. Round C: few labels (5 fleet drones, natural images with about 12% defects, last-4 update, 1 local pass for each round)

| Labels for each drone | FedAvg, base start | FedAvg, fine-tuned start | Same labels on one server | Each drone alone (mean, min to max) |
|---|---|---|---|---|
| Condition macro F1, 25 | 0.159 | 0.565 | — | — |
| Condition macro F1, 50 | 0.159 | 0.571 | 0.600 | 0.559 (0.519 to 0.582) |
| Condition macro F1, 100 | 0.162 | 0.582 | — | — |
| Condition macro F1, 200 | 0.207 | 0.575 | 0.623 | 0.551 (0.528 to 0.576) |
| Defect AUROC, 25 | 0.721 | 0.856 | — | — |
| Defect AUROC, 50 | 0.726 | 0.836 | 0.904 | 0.859 (0.831 to 0.940) |
| Defect AUROC, 100 | 0.767 | 0.930 | — | — |
| Defect AUROC, 200 | 0.838 | 0.933 | 0.958 | 0.902 (0.840 to 0.951) |

Reference without fleet labels: the server model `fl_server_central` has a macro F1 of 0.558, a component accuracy of 0.894 and a defect AUROC of 0.903.

- From the published checkpoint, few labels give almost nothing. With 50 and 100 labels, the macro F1 is near the majority baseline (0.158).
- From the fine-tuned start, 100 and 200 labels for each drone increase the defect AUROC from 0.903 to 0.930 and 0.933. The macro F1 changes little (0.558 to 0.575 or 0.582, the intervals overlap).
- Defects in the labels of the whole fleet: 18 (25 labels for each drone), 37 (50), 69 (100) and 149 (200). With 25 labels, the drones have 0, 2, 3, 6 and 7 defects.
- With 25 and 50 labels, the defect AUROC decreases to 0.856 and 0.836. With 50 labels, the defect risk increases from 291 to 331 for each 1,000 images.
- With few defects, FedAvg makes the model more conservative. With 25 labels, the false-alarm rate at the val threshold decreases from 0.139 to 0.044, and the defect recall decreases from 0.832 to 0.740.
- The same 37 defects on one server keep the AUROC of the server model (0.904). Thus the decrease comes from the average of drones with 0 to 15 defects each, not from the number of labels.
- FedAvg starts to help at about 70 defects in the fleet (100 labels for each drone, 6 to 27 defects on each drone).
- The same labels on one server are better than FedAvg: AUROC 0.958 against 0.933 with 200 labels, risk 181 against 280.
- FedAvg is better than the mean drone that trains alone with 200 labels: AUROC 0.933 against 0.902, macro F1 0.575 against 0.551.

## 4. Round D: different image types (6 drones, 5,107 images, start from the published checkpoint, last-4 update)

| Partition | Difference between drones (Jensen-Shannon) | Condition macro F1 | Component accuracy | Defect AUROC |
|---|---|---|---|---|
| Random (`fl_d_iid`) | 0.0004 to 0.0017 | 0.254 (0.243 to 0.267) | 0.334 | 0.856 |
| Tower (`fl_b_last4`) | 0.009 to 0.070 | 0.281 (0.258 to 0.304) | 0.321 | 0.838 |
| Component (`fl_d_component`) | 0.128 to 0.333 | 0.287 (0.260 to 0.318) | 0.331 | 0.837 |
| Component, FedProx | 0.128 to 0.333 | 0.285 (0.259 to 0.317) | 0.339 | 0.837 |
| Dirichlet 0.3 (`fl_d_dirichlet`) | 0.090 to 0.665 | 0.291 (0.264 to 0.318) | 0.477 | 0.833 |

- The intervals of all partitions overlap. Thus in this setting, the partition has no measurable effect.
- Recall for each defect type on the test towers:

| Defect type | Drone that trains alone and has the type | FedAvg, component | FedAvg, random | FedAvg, full model, tower (round A) |
|---|---|---|---|---|
| Corrosion (only on shackles) | 0.86 | 0.07 | 0.01 | 0.76 |
| Missing cap (only on glass insulators) | 0.79 | 0.02 | 0.03 | 0.37 |
| Bird nest (only on vari-grip clamps) | 0.94 | 0.12 | 0.00 | 0.67 |

- Each drone that trains alone learns its own defect type. FedAvg with the last-4 update does not learn these types, also with the random partition. Thus the cause is the slow learning of the small update (section 2), not the different data on the drones.
- A drone that trains alone does not learn the components of the other drones. Its component accuracy is 0.04 to 0.34 (mean 0.16). FedAvg with the component partition has 0.331.

## 5. Conclusion for the next runs

1. The best small update for the uplink is int8 (half size, no measurable cost).
2. A small update needs more rounds, or a fine-tuned start. In round C, the last-4 update from the fine-tuned start works.
3. Round D must be done again with an update that learns: the full model, or the last-4 update from the fine-tuned start, or 30 rounds. Then the effect of the partition can show.
4. Round B must be done again from the fine-tuned start. This is the realistic case: a central model, then small updates from the fleet.
