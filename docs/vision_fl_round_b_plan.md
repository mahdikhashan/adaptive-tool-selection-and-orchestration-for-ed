# Plan: federated rounds B, C and D (smaller updates, few labels, different image types)

Date: 2026-11-08. Status: plan. No code and no run. Round A results: `docs/vision_fl_results.md`.

## 1. Starting point

| Run (round A) | Condition macro F1 | Component accuracy | Defect AUROC | Uplink for each drone and round |
|---|---|---|---|---|
| Central (`good300_full`) | 0.703 | 0.924 | 0.965 | — |
| FedAvg, 6 drones, full model (`fl_fedavg_k6`) | 0.541 | 0.761 | 0.939 | 229.5 MB |
| Fleet FedAvg, 5 drones, fine-tuned start (`fl_fleet_warm_k5`) | 0.636 | 0.912 | 0.953 | 229.5 MB |

## 2. Parts of the model and the size of an update

The script counted the parameters of laya-vision (201,161,347 in total). The size of an update is 2 bytes for each parameter (fp16).

| Part | Parameters | Trained in round A |
|---|---|---|
| Vision encoder (SigLIP) | 86,433,024 | no |
| Connector (vision to text) | 7,077,888 | yes |
| Token embeddings | 28,385,280 | yes |
| 20 language layers, 3,540,096 each | 70,801,920 | yes |
| Decision head (2 transformer layers, scorer, act head, type embedding) | 8,462,659 | yes |

The trained part is the part that each drone sends. These are the options:

| Option | Trained and sent | Parameters | Update (fp16) | Update (int8) |
|---|---|---|---|---|
| `head` | Decision head | 8.46M | 16.9 MB | 8.5 MB |
| `lora8` | Decision head + LoRA rank 8 on the 7 linear layers of all 20 language layers | 10.09M | 20.2 MB | 10.1 MB |
| `last1` | Head + last language layer | 12.00M | 24.0 MB | 12.0 MB |
| `last4` | Head + last 4 language layers | 22.62M | 45.2 MB | 22.6 MB |
| `layers` | Head + all 20 language layers (no embeddings, no connector) | 79.27M | 158.5 MB | 79.3 MB |
| `full` (round A) | All parts except the vision encoder | 114.73M | 229.5 MB | 114.7 MB |

- LoRA: each linear layer gets 2 small matrices (rank 8). One language layer then has 81,408 new parameters. The decision head is 84% of the `lora8` update.
- The token embeddings are 25% of the `full` update. The images and the questions use a fixed vocabulary, so the embeddings probably need no training.
- int8: the drone sends the change of the weights in 8 bits (one scale for each tensor). This halves the update. The script must test the effect on the accuracy.
- The server sends the same size back to each drone.
- For comparison: EVLM sends 3.75 MB for a 10-minute mission. The `head` update in int8 (8.5 MB) is about 2 missions of EVLM data.

## 3. Round B: smaller updates and FedProx

All runs use the same drones and data as `fl_fedavg_k6`. They have 6 drones, split by tower, and 5,107 images. They start from the published checkpoint and use 10 rounds with 2.5 passes in total. Thus each run compares directly with `fl_fedavg_k6` (full, 0.541).

| Run | Update | Size for each drone and round | Question |
|---|---|---|---|
| `fl_b_head` | `head` | 16.9 MB | Is the head alone enough? |
| `fl_b_lora8` | `lora8` | 20.2 MB | Does LoRA give the accuracy of more layers at the size of the head? |
| `fl_b_last4` | `last4` | 45.2 MB | The update in the request |
| `fl_b_layers` | `layers` | 158.5 MB | What do the embeddings and the connector add? |
| `fl_b_last4_int8` | `last4`, int8 | 22.6 MB | What does int8 cost? |
| `fl_b_full_prox` | `full`, FedProx μ = 0.01 | 229.5 MB | Does FedProx help the uneven drones? |
| `fl_b_last4_prox` | `last4`, FedProx μ = 0.01 | 45.2 MB | The same with the smaller update |
| `fl_b_last4_r30` | `last4`, 30 rounds, the same local steps in each round as in 10 rounds | 45.2 MB | Does the gap to the central model close with more rounds? |

- FedProx adds μ/2 × ‖w − w_global‖² to the local loss. It keeps the local weights near the global weights.
- The run `fl_b_last4_r30` has 3 times more training than the other runs. It answers the open question of round A (`docs/vision_fl_results.md`, section 2).
- Result for each run: the same metrics as round A, and accuracy against the total uplink.
- The best update on the val towers, with a small uplink, becomes the update of rounds C and D.

## 4. Round C: few labels, as in a real drone operation

### 4.1 Setting

- A drone flies one mission and an operator labels some of the crops. The plan gives each drone a label budget n: 25, 50, 100 or 200 images.
- The drones are the 5 fleet towers (one tower for each drone). These towers are not in the training data of the server model (`fl_server_central`).
- The images come from all images of the drone's tower, at random. Thus the class shares are natural: about 12% defects (10,962 images on the 5 towers). With n = 25, a drone has about 3 defect images.
- Each drone trains 1 pass over its own labels in each round, for 10 rounds.
- The update is the winner of round B (provisionally `last4`).
- The val and test towers do not change. To save time, the prediction uses 1,500 val images and all 6,092 test images.

### 4.2 Runs

| Run | Start | n for each drone |
|---|---|---|
| `fl_c_base_n{25,50,100,200}` | Published checkpoint | 25, 50, 100, 200 |
| `fl_c_warm_n{25,50,100,200}` | Server model (`fl_server_central`) | 25, 50, 100, 200 |
| `fl_c_warm_local_n{50,200}` | Server model, each drone alone | 50, 200 |
| `fl_c_warm_central_n{50,200}` | Server model, the labels of all 5 drones on one server | 50, 200 |

Reference without fleet labels: `fl_server_central` (macro F1 0.558, defect AUROC 0.903).

### 4.3 Questions

1. How many labels does each drone need before FedAvg improves the server model?
2. Is the gap between the base start and the fine-tuned start larger with few labels?
3. With few labels, does FedAvg come near the central model with the same labels?

## 5. Round D: drones with different image types

### 5.1 Partitions

All runs use the 5,107 images of round A and 6 drones. Only the division of the images between the drones changes.

| Partition | Rule | Drones (images) |
|---|---|---|
| `iid` | At random | 6 drones with the same mix |
| `tower` (round A) | By tower | 637 to 1,097 images |
| `component` | Each drone inspects other components | see below |
| `dirichlet0.3` | The condition shares of each drone come from a Dirichlet distribution with α = 0.3 | different defect types on each drone |

`component` partition:

| Drone | Components | Images | Defect types |
|---|---|---|---|
| 1 | shackle | 1,113 | rust, corrosion |
| 2 | vari-grip clamp | 874 | rust, bird nest |
| 3 | glass insulator, spacer | 701 | missing cap |
| 4 | lightning rod suspension, plate | 686 | rust, other defect |
| 5 | yoke suspension, yoke | 801 | rust |
| 6 | polymer insulator, stockbridge damper, preformed damper | 932 | rust, other defect |

- In the dataset, each defect type occurs only on some components. Corrosion occurs only on shackles, missing cap only on glass insulators, and bird nest only on vari-grip clamps. Thus in the `component` partition, only one drone sees each of these defect types.
- The script reports the difference between the class shares of each drone and the global shares (Jensen-Shannon divergence), for each partition.

### 5.2 Runs

| Run | Partition | Method |
|---|---|---|
| `fl_d_iid` | `iid` | FedAvg |
| `fl_d_component` | `component` | FedAvg |
| `fl_d_component_prox` | `component` | FedProx μ = 0.01 |
| `fl_d_dirichlet` | `dirichlet0.3` | FedAvg |
| `fl_d_component_local` | `component` | Each drone alone |

The update is the winner of round B. The run with the `tower` partition and the same update comes from round B.

### 5.3 Questions

1. How much accuracy does each partition cost against `iid`?
2. Can a drone recognize a defect type that it never saw, after FedAvg? The script reports the recall of each defect type on the test towers.
3. Does FedProx help the `component` partition?

## 6. Code changes

| Option of `scripts/vision_fl.py` | Values |
|---|---|
| `--share` | `full`, `layers`, `last_n` (with `--n-last`), `head`, `lora` (with `--lora-rank`) |
| `--quantize` | `none`, `int8` |
| `--partition` | `tower`, `iid`, `component`, `dirichlet` (with `--alpha`) |
| `--budget` | number of labelled images for each drone, from the natural images of its tower |
| `--local-passes` | passes for each round (round C) |
| `--val-images-held` | number of val images in the final prediction |

- LoRA uses `peft` (already a dependency) on the 7 linear layers of the language layers. The head trains without LoRA.
- All existing options keep their defaults. Thus the round-A runs stay reproducible.

## 7. GPU time (estimate, RTX 2080 Ti)

| Round | Runs | Time |
|---|---|---|
| B | 8 | about 9 hours (the 30-round run needs about 2.5 hours) |
| C | 12 | about 4 hours (small data, so the prediction uses most of the time) |
| D | 5 | about 5 hours |

Rounds C and D wait for the result of round B, because they use its best update.
