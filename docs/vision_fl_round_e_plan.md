# Plan: federated round E (fine-tuned start, int8, less data, less uplink)

Date: 2026-11-08. Status: code complete, not run on the GPU. The new options are `--balance` and `--server-lr` in `scripts/vision_fl.py`. The 18 scripts are `scripts/runs/fl_e*.sh`. Earlier results: `docs/vision_fl_results.md`, `docs/vision_fl_bcd_results.md`.

## 1. Goal

Get the accuracy of a federated fleet with less data and less uplink. All runs in this round:

- start from the fine-tuned server model `fl_server_central` (13 server towers),
- use only the 5 fleet towers, which the server model did not see,
- send int8 updates (one byte for each parameter), because round B showed no measurable cost for int8.

## 2. Starting point (test towers)

| Model | Condition macro F1 | Component accuracy | Defect AUROC | Uplink of the run |
|---|---|---|---|---|
| Server model alone (`fl_server_central`) | 0.558 | 0.894 | 0.903 | 0 |
| Fleet FedAvg, full model, fp16 (`fl_fleet_warm_k5`) | 0.636 (0.612 to 0.657) | 0.912 | 0.953 (0.942 to 0.963) | 11.5 GB |
| Central, all 18 towers (`good300_full`) | 0.703 | 0.924 | 0.965 | — |
| Round C, 50 labels for each drone, last 4, fp16 | 0.571 | 0.903 | 0.836 | 2.3 GB |
| Round C, 200 labels for each drone, last 4, fp16 | 0.575 | 0.914 | 0.933 | 2.3 GB |
| Round C, the same 200 labels on one server | 0.623 | 0.923 | 0.958 | — |

## 3. Data on the fleet

| Split | Drones | Images on each drone | Defects on each drone | Difference between drones (Jensen-Shannon) |
|---|---|---|---|---|
| Tower | 5 | 206 to 1,042 | 100 to 605 | 0.006 to 0.148 |
| Random | 6 | 399 to 400 | 191 to 213 | at most 0.002 |
| Component | 6 | 301 to 498 | 14 to 369 | 0.172 to 0.361 |
| Dirichlet 0.3 | 6 | 44 to 1,140 | 34 to 496 | 0.031 to 0.723 |

The fleet has 2,553 images (1,298 defects) and all 11 components.

## 4. Round E1: smallest update that keeps the accuracy

5 fleet drones (one tower each), 2,553 images, 10 rounds, int8.

| Run | Update | Update for each drone and round (int8) | Uplink of the run |
|---|---|---|---|
| `fl_e1_head` | Decision head | 8.5 MB | 0.42 GB |
| `fl_e1_lora8` | Head + LoRA rank 8 | 10.1 MB | 0.50 GB |
| `fl_e1_last4` | Head + last 4 language layers | 22.6 MB | 1.13 GB |
| `fl_e1_layers` | Head + 20 language layers | 79.3 MB | 3.96 GB |
| `fl_e1_full` | All parameters except the vision encoder | 114.7 MB | 5.74 GB |

Question: which update with less than 10% of the uplink of `fl_fleet_warm_k5` (11.5 GB) is in its 95% interval?

## 5. Round E2: fewer labels without the decrease

Round C showed that FedAvg makes the server model worse with few labels (defect AUROC 0.836 with 50 labels for each drone). The model becomes conservative: it gives "good" more often. These runs test 3 changes against this problem.

| Change | Rule | Reason |
|---|---|---|
| Balanced local sampling (`--balance`) | Each drone samples its defect images and its good images with the same probability | The drone sees its few defects more often, so the local update does not move the model toward "good" |
| Smaller server step (`--server-lr 0.5`) | New global weights = old weights + 0.5 × (mean of the drones − old weights) | The server model changes less in each round |
| Stronger FedProx (`--prox 0.1`) | Local loss + 0.1/2 × ‖w − w_global‖² | Each drone stays near the server model |

Runs: 5 fleet drones, natural labels (about 12% defects), update `last_n` 4 in int8, 1 local pass for each round, 10 rounds.

| Run | Labels for each drone | Change |
|---|---|---|
| `fl_e2_plain_n50`, `fl_e2_plain_n200` | 50, 200 | none (shows int8 against round C fp16) |
| `fl_e2_balance_n50`, `fl_e2_balance_n200` | 50, 200 | balanced local sampling |
| `fl_e2_slr_n50`, `fl_e2_slr_n200` | 50, 200 | server step 0.5 |
| `fl_e2_prox_n50`, `fl_e2_prox_n200` | 50, 200 | FedProx 0.1 |

Questions:

1. Does one change keep the defect AUROC at or above the server model (0.903) with 50 labels?
2. Does one change bring 200 labels near the central result with the same labels (0.958)?

## 6. Round E3: different image types, with an update that learns

6 fleet drones (5 for the tower split), 2,553 images, update `last_n` 4 in int8, 10 rounds.

| Run | Split | Method |
|---|---|---|
| `fl_e1_last4` (from E1) | tower | FedAvg |
| `fl_e3_iid` | random | FedAvg |
| `fl_e3_component` | component | FedAvg |
| `fl_e3_component_prox` | component | FedProx 0.1 |
| `fl_e3_dirichlet` | Dirichlet 0.3 | FedAvg |
| `fl_e3_component_local` | component | each drone alone |

Questions:

1. How much accuracy does each split cost against the random split, when the update can learn?
2. After FedAvg, does the fleet model find the defect types that only one drone has (corrosion, missing cap, bird nest)?

## 7. Success criteria

1. E1: an update has at most 1.15 GB of uplink (10% of 11.5 GB). Its condition macro F1 is at least 0.612, and its defect AUROC is at least 0.942. These are the lower limits of the 95% intervals of `fl_fleet_warm_k5`.
2. E2: with 50 labels for each drone, the defect AUROC is at least 0.903 (no decrease against the server model).
3. E2: with 200 labels, the defect AUROC is in the 95% interval of the central result with the same labels (0.950 to 0.965).
4. E3: the recall of each defect type that only one drone has is at least half of the central model.

## 8. Code changes

| Option of `scripts/vision_fl.py` | Effect |
|---|---|
| `--balance` | Each drone samples defect and good images with the same probability (the `dataset` key of the laya-vision stream) |
| `--server-lr` | Server step. 1.0 (default) is FedAvg |

All existing options keep their defaults. Each run gets its own script in `scripts/runs/`. All runs use `--val-images-held 1500` to save prediction time. The test towers stay complete.

## 9. GPU time (estimate)

| Round | Runs | Time |
|---|---|---|
| E1 | 5 | about 4 hours |
| E2 | 8 | about 7 hours |
| E3 | 5 | about 4.5 hours |

Total about 15 GPU-hours, so about 5 to 6 hours on 3 GPUs. The 3 rounds do not depend on each other, so all 18 runs can start at the same time. If E1 finds a smaller update than `last_n` 4, E2 and E3 can repeat with it.
