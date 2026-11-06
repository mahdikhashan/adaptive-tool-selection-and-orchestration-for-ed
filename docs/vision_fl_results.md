# Federated fine-tuning of laya-vision: results

Date: 2026-11-08. Runs: 8 runs (13 models), student07, RTX 2080 Ti, fp32, commit 9860747. The plan is in `docs/vision_fl_plan.md`. All values are in `results/vision/fl_*/summary.json` and `run.json`. Each run has one seed.

## 1. Summary

1. The federated training loop with 1 drone gives the same result as the central model `good300_full`. Thus the loop is not a cause of the differences below.
2. With the same total training, FedAvg becomes worse when the same data goes to more drones. With 6 and 18 drones, the val accuracy still increases after 10 rounds.
3. With 6 drones, FedAvg is better than the mean drone that trains alone, for each metric.
4. A fine-tuned start makes FedAvg on the fleet much better than a start from the published checkpoint. The fleet also improves the fine-tuned server model.
5. The full model update is 229.5 MB for each drone and round. One run needs 6.9 to 41.3 GB of uplink.

## 2. Number of drones (test towers, 6,092 images)

| Run | Drones | Condition macro F1 | Component accuracy | Defect AUROC | Item accuracy |
|---|---|---|---|---|---|
| `good300_full` (reference) | central | 0.703 (0.685 to 0.720) | 0.924 (0.917 to 0.931) | 0.965 (0.956 to 0.973) | 0.948 |
| `fl_central_k1` (control) | 1 | 0.697 (0.677 to 0.717) | 0.933 (0.927 to 0.939) | 0.959 (0.948 to 0.969) | 0.953 |
| `fl_fedavg_k3` | 3 | 0.624 (0.603 to 0.643) | 0.874 (0.866 to 0.882) | 0.955 (0.945 to 0.964) | 0.924 |
| `fl_fedavg_k6` | 6 | 0.541 (0.513 to 0.570) | 0.761 (0.751 to 0.771) | 0.939 (0.928 to 0.950) | 0.918 |
| `fl_fedavg_k18` | 18 | 0.474 (0.444 to 0.505) | 0.700 (0.688 to 0.711) | 0.933 (0.922 to 0.945) | 0.900 |
| `fl_local_k6` (mean of 6 drones) | 6, alone | 0.371 (min 0.278, max 0.525) | 0.621 (min 0.498, max 0.782) | 0.881 (min 0.837, max 0.916) | — |

Values in brackets are 95% bootstrap intervals, except for the local-only row (minimum and maximum of the 6 drone models).

- The control and `good300_full` have overlapping intervals for each of the 3 metrics.
- From 1 to 18 drones, the condition decreases from 0.697 to 0.474, and the component from 0.933 to 0.700. The defect AUROC decreases less, from 0.959 to 0.933.
- Val accuracy after the last round: 1 drone 0.938, 3 drones 0.897, 6 drones 0.839, 18 drones 0.826. The curves for 6 and 18 drones still increase at round 10.
- Possible cause: all runs have the same total number of training steps. In one round, each drone does only its part of the steps, and the server takes the mean of the drone updates. Thus the global model moves less in each round when there are more drones. A run with more rounds can show if the gap closes. This explanation is not tested.

## 3. FedAvg against local training (6 drones)

- FedAvg is better than the mean local model for each metric. Condition macro F1: 0.541 against 0.371. Component accuracy: 0.761 against 0.621. Defect AUROC: 0.939 against 0.881.
- FedAvg is better than the best local model for the defect AUROC (0.939 against 0.916). For the condition macro F1, the interval of the best local model (drone 1, 0.525) overlaps with the FedAvg interval. For the component accuracy, the best local model (drone 0, 0.782) is better than FedAvg (0.761).
- The local models with the fewest defect images are the worst: drone 3 (29% defects) has a macro F1 of 0.278.

## 4. Two start models (fleet: 5 towers, 5 drones, 2,553 images)

| Run | Start | Condition macro F1 | Component accuracy | Defect AUROC | Item accuracy |
|---|---|---|---|---|---|
| `fl_server_central` | Published checkpoint, central on 13 server towers | 0.558 (0.532 to 0.584) | 0.894 (0.887 to 0.902) | 0.903 (0.887 to 0.919) | 0.917 |
| `fl_fleet_base_k5` | Published checkpoint (not fine-tuned) | 0.513 (0.484 to 0.540) | 0.593 (0.581 to 0.605) | 0.932 (0.920 to 0.945) | 0.901 |
| `fl_fleet_warm_k5` | `fl_server_central` (fine-tuned) | 0.636 (0.612 to 0.657) | 0.912 (0.905 to 0.919) | 0.953 (0.942 to 0.963) | 0.937 |

- The fine-tuned start is better than the base start for each metric. The intervals do not overlap.
- The fleet improves the server model: macro F1 0.558 to 0.636, defect AUROC 0.903 to 0.953. The images of the fleet stay on the drones.
- The val accuracy of `fl_fleet_warm_k5` is 0.904 before the first round and 0.939 after the last round. The base start goes from 0.596 to 0.799.
- `fl_fleet_warm_k5` is near `good300_full`, which saw all images on one server: item accuracy 0.937 against 0.948, defect AUROC 0.953 against 0.965.

## 5. Communication

| Run | Drones | Total uplink (fp16) |
|---|---|---|
| `fl_fedavg_k3` | 3 | 6.9 GB |
| `fl_fedavg_k6` | 6 | 13.8 GB |
| `fl_fedavg_k18` | 18 | 41.3 GB |
| `fl_fleet_base_k5`, `fl_fleet_warm_k5` | 5 | 11.5 GB |

- The script calculates the values: 229.5 MB × drones × 10 rounds. The simulation does not send data on a network.
- For comparison: the 5,107 training images at about 35 KB each are about 0.18 GB. Thus the federation does not decrease the data volume. The reason for the federation is that the images stay on the drones.
- A smaller update (last 4 layers and head, 45 MB) is round B of the plan.

## 6. Other results

- Training time for each FedAvg run: 61 to 66 minutes, including the val evaluation after each round (central `good300_full`: 24 minutes). Each local drone model: 3 to 5 minutes of training.
- Latency does not change: about 74 ms for each image.

## 7. Limits

1. One seed for each run.
2. The drones are a simulation on one GPU. The script calculates the update size and the uplink. It does not measure them.
3. All runs have the same total training. This makes the comparison fair for the compute, but it gives fewer effective steps to FedAvg with many drones.

## 8. Next steps

1. FedAvg with 6 and 18 drones for 30 rounds, with the same local steps in each round. This shows if the gap to the central model closes.
2. Round B of the plan: smaller updates (last 4 layers and head, or head only) and FedProx.
3. More seeds for the main comparisons.
