# Plan: federated fine-tuning of laya-vision on a fleet of drones

Date: 2026-11-07. Status: plan. No code and no run.

## 1. Goal

Each drone has the images of its own towers. The drones fine-tune one laya-vision model together. The images stay on the drones. Only model weights go to a server.

Questions:

1. How near is the federated model to the central model (`good300_full`)?
2. Is the federated model better than a model that one drone trains alone?
3. How do the number of drones and the different data on each drone change the result?
4. How much data must each drone send, and which part of the model must it send?

## 2. References

| Reference | Role | Test result (images) |
|---|---|---|
| Zero-shot laya-vision | Lower limit, no training | condition macro F1 0.180, component accuracy 0.141, defect AUROC 0.714 |
| `good300_full` | Upper limit: the same images on one server | condition macro F1 0.703, component accuracy 0.924, defect AUROC 0.965 |
| Local only | Each drone trains alone | to measure |

The federated runs start from the published checkpoint, not from `good300_full`. The model `good300_full` saw the images of all drones, so it cannot be the start of a federated run.

## 3. Data on each drone

The drones use the same 4,807 training images as `good300_full`. The script divides them by tower (`tower_buckets` in `src/fedorch/folds.py`). Thus the union of the drone data is the central data, and the comparison is fair.

| Drones | Images on each drone | Defect share on each drone | Components on each drone |
|---|---|---|---|
| 3 | 1,605, 1,605, 1,597 | 0.48, 0.51, 0.33 | 10, 11, 11 |
| 6 | 1,097, 1,049, 648, 645, 731, 637 | 0.42 to 0.57 | 8 to 11 |
| 18 (one tower for each drone) | 1 to 1,097 | 0 to 0.68 | 1 to 11 |

- With 18 drones, the data is very different between drones. For example, 4 drones have no defect image, and only 4 drones have missing-cap images.
- The val towers and the test towers do not change. They are not drone data. The server uses them only for the evaluation.
- Each drone keeps 6% of its images out of the training. The drones send the logits and the labels of these images to the server, and the server fits the temperatures. The drones do not send these images.
- The class-share correction needs the class counts of the training images. Each drone sends its counts, and the server adds them.

## 4. Method (FedAvg)

1. The server sends the global weights to all drones.
2. Each drone trains on its own questions for a part of a pass. It uses AdamW, a new optimizer in each round, and the same loss as `good300_full`.
3. Each drone sends its weights to the server.
4. The server calculates the mean of the weights. The weight of each drone is its number of training questions.
5. Steps 1 to 4 are one round. The plan uses 10 rounds.

Settings:

| Setting | Value | Reason |
|---|---|---|
| Trained part | `full`, 114.7M parameters | The same as the best central model |
| Local training | 0.25 passes over the drone data in each round | 10 rounds × 0.25 = 2.5 passes, the same as `good300_full` |
| Learning rate | head 1e-4, layers 2e-5, cosine decrease over the rounds | The same values as `good300_full` |
| Batch | 8 | The same as `good300_full` |
| Precision | fp32 | The RTX 2080 Ti and the GV100 have no native bf16 |

The total number of training questions is the same as for `good300_full`. Thus a difference in the result comes from the federation, not from more training.

The simulation runs all drones one after the other on one GPU. It keeps only one model on the GPU and a sum of the weights in the CPU memory. The averaging is the same as the `FedAvg` strategy of Flower. A version with Flower is an option for the paper.

The script uses its own short training loop with the parts of laya-vision (`make_item`, `vlm_loss`). The `train` function of laya-vision starts a new learning-rate schedule in each call, and it uses bf16 on CUDA.

## 5. Experiments

The experiments are in rounds. Each round starts from the best result of the round before it.

### Round A: number of drones

| Run | Drones | Method |
|---|---|---|
| `fl_local_k6` | 6 | Each drone trains alone for 2.5 passes over its own data. No averaging. |
| `fl_fedavg_k3` | 3 | FedAvg |
| `fl_fedavg_k6` | 6 | FedAvg |
| `fl_fedavg_k18` | 18 | FedAvg |

The local-only run gives 6 models. The evaluation gives the mean, the minimum and the maximum of the 6 models on the test towers.

### Round B: which part each drone sends

This round uses the number of drones from round A with the largest gap to the central model. A large gap shows the effect of the change better.

| Run | Sent part | Parameters | Size for each round and each drone (fp16) |
|---|---|---|---|
| `full` (from round A) | All parameters except the vision encoder | 114.7M | 229 MB |
| `last_n` | Head and last 4 language layers | 22.6M | 45 MB |
| `head` | Decision head only | about 8.5M (calculated, not measured) | about 17 MB |

The round also tests FedProx (μ = 0.01). FedProx adds a term that keeps the local weights near the global weights. It can help when the data is different between drones.

### Round C: participation and rounds

- 50% of the drones in each round, at random.
- 20 rounds with 0.125 passes for each round. This keeps the total training the same.

## 6. Metrics

- The same evaluation as before (`scripts/vision_summarize.py`): condition, component and defect metrics on the test towers. The metrics have 95% bootstrap intervals, with and without the class-share correction. Also the item accuracy against the step-2 tools.
- For each round: the accuracy on 600 val images (1,800 questions). This gives a learning curve for each run.
- Communication: the MB that each drone sends in each round, and the total MB of the run.
- Time: the training time for each round and for the run.

## 7. Success criteria

1. FedAvg with 6 drones is in the 95% interval of `good300_full` for the condition macro F1 and the defect AUROC.
2. FedAvg is better than each local-only model.
3. The results show the gap between FedAvg and the central model for 3, 6 and 18 drones.
4. Round B shows how much accuracy a smaller update costs.

## 8. GPU time (estimate)

- `good300_full` needed 24 minutes for 4,507 steps on the RTX 2080 Ti. A FedAvg run has the same number of steps. The changes of the weights add about 1 to 2 minutes for each round. The val evaluation adds about 1 minute for each round.
- Thus one FedAvg run needs about 45 to 55 minutes, and the prediction of the val and test images needs 15 minutes more.
- Round A: about 4 hours. Round B: about 3 hours. These values are estimates.
- CPU memory: one copy of the 114.7M weights in fp32 is about 460 MB. The sum and the global weights need about 1 GB.

## 9. Risks

1. **Small drones.** With 18 drones, 9 drones have fewer than 50 images. Their weight in the average is small, so they change the model only a little. Their local training needs at least 1 step in each round.
2. **Different classes on each drone.** A drone without defect images trains the model toward "good". FedAvg can then be worse than the central model. FedProx in round B is a test for this risk.
3. **Strong confidence.** The central full model reached a loss of 0 on many batches. Local training on a small drone can do this faster. The temperatures from the held-out images correct the probabilities, but not the order of the answers.
4. **Uplink.** 229 MB for each round is too much for the 4G link during a mission. The plan sends the update after the mission, from the ground station. This agrees with the learning flow in `docs/architecture.md`.

## 10. Files to make

- `scripts/vision_fl.py`: one run (`--drones`, `--mode fedavg|local`, `--share full|last_n|head`, `--prox`, `--rounds`, `--participation`).
- `scripts/vision_fl_round_a.sh`: the 4 runs of round A.
- The outputs go to `results/vision/<run>/`, in the same format as the other runs. The evaluation script needs no change, except for the 6 models of the local-only run.
