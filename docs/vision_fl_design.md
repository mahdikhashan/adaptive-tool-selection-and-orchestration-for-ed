# Vision decision model with federated learning (branch `vision-fl`)

Date: 2026-11-05. Status: code complete, not run on the GPU.

## 1. Idea

A vision-language decision model reads only the image of a frame. It answers typed questions about the frame. A fleet of drones fine-tunes the model with federated learning (FL). The setup has no network traces and no cloud. All tools run on the drone.

## 2. Model

- Checkpoint: `thaitea/laya-vision`, revision `f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc`.
- Code: `github.com/r33drichards/laya-vision`, commit `9e1e2419d855ad3e1a2af4d4bd1ef6be5418842c`.
- Size: 201M parameters. The backbone is SmolVLM-256M with 20 of its 30 language layers.
- Output: one probability for each option, from one forward pass. The model does not generate text.
- Licence: the code is Apache 2.0. The weights are CC BY-NC-SA 4.0. Thus the weights are for research only.
- The model needs `transformers==5.17.0`. Version 5.22 gives an error in the image features.
- The Python package of laya-vision has the name `laya`. It replaces the text Laya on this branch. Thus the scripts of step 4, step 4b and plan A that use the text Laya do not run on this branch.

## 3. Input and questions

The state is the image only. The state has no classifier output, no evidence text and no telemetry.

Each frame gives 2 questions of the type `choice`:

| Question | Instructions | Options | Target |
|---|---|---|---|
| `answer` | The ElecVQA question, for example "Does the damper stockbridge have a defect?" | The answer options of the item, for example "good, no defect", "defect", "cannot judge" | 1 for the true label, 0 for the other options |
| `route` | "The drone must answer: {question} Which onboard tool gives the correct answer for this image?" | "accept the answer of the onboard defect classifier", "ask the onboard vision-language model" | softmax(−(risk + cost) / T), T = 1, cost 0 and 0.1 |

- The `answer` question makes the model an onboard tool.
- The `route` question makes the model an onboard router between the classifier and SmolVLM with LoRA.
- The risk of an answer comes from `docs/risk_weights.md`: miss 10, false alarm 1, abstention 0.5.
- The tool answers for the train towers come from the cross-fitted cache of step 2.

## 4. Training

- Trainable part: the decision head and the last 4 language layers (`freeze=last_n`, `n_last=4`). This is 22.6M parameters.
- Loss: the loss of laya-vision (`vlm_loss`, RLCD with soft cross-entropy).
- Precision: fp32. The GV100 has no bf16. The training loop of laya-vision uses bf16 autocast, so `src/fedorch/vision.py` has its own loop.
- Optimizer: AdamW, learning rate 1e-4 for the head and 2e-5 for the layers, batch 8.
- Each mode uses the same total number of optimizer steps (3,000).

## 5. Modes

| Mode | Description |
|---|---|
| `zeroshot` | No training. |
| `central` | One model on all train-tower frames. This is the upper bound for FL. |
| `fedavg` | K drones. Each drone has a set of the 18 train towers. In each round, each drone trains for 3,000 / (10 × K) steps from the global weights. The server averages the trainable weights, weighted by the number of frames. 10 rounds. |
| `local` | K drones. Each drone trains alone for 3,000 / K steps. There is no averaging. |

The runs are `zeroshot`, `central`, `fedavg` with K = 3, 6 and 18, and `local` with K = 6. With K = 18, each drone has one tower.

The assignment of the towers to the drones balances the number of frames (`tower_buckets` in `src/fedorch/folds.py`). The tower sizes are different, so the drones are not identical (non-IID).

## 6. Evaluation

The evaluation uses the 7,057 test-tower frames. It uses no replay, because the setup has no network. Late answers do not occur on the edge path in step 3, so the risk is the same as in the replay.

1. **Model as a tool:** the option with the highest probability is the answer. The script gives the risk per 1,000 frames, the accuracy and the defect recall.
2. **Model as a router:** the drone escalates to SmolVLM if p(onboard VLM) − p(classifier) is at least a threshold. The threshold comes from quantiles of the validation scores, with the lowest validation risk.

References on the same test frames, without the network:

| Policy | Risk per 1,000 frames |
|---|---|
| classifier only | 177.0 |
| SmolVLM with LoRA only | 188.3 |
| static cascade θ = 0.95 | 155.7 |
| edge oracle (best of the 2 tools) | 91.1 |

## 7. Questions that the runs answer

1. Can the vision decision model, with only the image, decrease the risk below the static cascade (155.7)?
2. Is FedAvg near the central model? How does the gap change with the number of drones (3, 6, 18)?
3. Is FedAvg better than a drone that trains alone?
4. What is the cost of the FL uplink? The script records the size of the update: 45 MB in fp16 for each drone in each round.

## 8. Files

- `src/fedorch/vision.py`: questions, targets, training loop, prediction, FedAvg.
- `scripts/vision_run.py`: one run (`--mode`, `--drones`, `--rounds`, `--steps`, `--seed`, `--limit`).
- `scripts/vision_run_gpu.sh`: all runs. The script skips a run that has `train.json`.
- `scripts/vision_summarize.py`: metrics on the laptop from `results/vision/*/scores.jsonl`.
