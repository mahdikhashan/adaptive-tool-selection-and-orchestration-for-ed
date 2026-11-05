# Plan A results: controllers with few labelled frames

Date: 2026-11-05. Status: complete.
Source files: `results/planA/metrics.json` and `results/planA/p_*.jsonl`. Figure: `planA-low-data/figures/learning-curve.html` in the artifacts directory, with the data in `planA-low-data/data/learning-curve.csv`.

## Question

One drone has few labelled frames. Can a pretrained decision model learn a better controller than the MLP from few labelled frames?

## Method

1. Each controller trains on a random part of the train-tower frames: 1%, 5%, 20% or 100%. 1% is about 255 frames.
2. Each size has 3 training seeds (0, 1, 2). For one seed, all controllers use the same frames.
3. All controllers use the choice target with the step 4b setting: temperature 1, 5 copies of each informative frame.
4. nano-jev trains for at least 600 optimizer steps. Laya trains for at least 300 optimizer updates. The MLP stops early on the validation loss.
5. Laya trains only at 5% and 20%, with seed 0, because one Laya run needs much GPU time.
6. We tune the escalation threshold of each run on the full validation towers.
7. The replay uses the weak regime, 5 network seeds and the 7,057 test-tower frames.

## Results

Risk per 1,000 frames, mean ± standard deviation over the training seeds:

| Training frames | Choice MLP | nano-jev (33M) | Laya (421M, 1 seed) |
|---|---|---|---|
| 1% | 172.6 ± 10.5 | 172.5 ± 3.3 | — |
| 5% | 156.6 ± 9.6 | 153.7 ± 13.5 | 168.0 |
| 20% | 131.0 ± 11.0 | 144.5 ± 24.7 | 165.8 |
| 100% | 113.7 ± 1.5 | 128.7 ± 9.9 | 150.0 |

For comparison: the classifier alone has 177.0. The static cascade has 155.7. The static cascade needs no labels for training. It needs only the validation towers to select its threshold.

Paired difference (nano-jev − MLP) for each seed. A negative value means that nano-jev is better:

| Training frames | Seed 0 | Seed 1 | Seed 2 | Mean | nano-jev better |
|---|---|---|---|---|---|
| 1% | +5.5 | −15.8 | +9.8 | −0.2 | 1 of 3 |
| 5% | +8.7 | +5.1 | −22.4 | −2.8 | 1 of 3 |
| 20% | +13.6 | −4.4 | +31.2 | +13.5 | 1 of 3 |
| 100% | +13.0 | +2.8 | +29.2 | +15.0 | 0 of 3 |

## Findings

1. With few labelled frames, nano-jev does not beat the MLP. At 1% and 5%, the two controllers are equal within the seed variation (mean difference −0.2 and −2.8 risk points). At 20% and 100%, the MLP is better by 13.5 and 15.0 risk points.
2. Laya is worse than both controllers at every size.
3. At 1%, no controller learns a useful decision. The risk of each controller (about 172) is near the risk of the classifier alone (177.0).
4. At 5%, the learned controllers are equal to the static cascade (155.7). Thus a drone with fewer than about 1,300 labelled frames gets no advantage from a learned controller.
5. The learned controllers start to beat the static cascade between 5% and 20% of the training frames. This is between about 1,300 and 5,100 labelled frames.
6. nano-jev has a large seed variation at 20% (standard deviation 24.7). One seed escalates only 3.1% of the frames and has a risk of 177.8. Thus nano-jev training is less stable than MLP training.
7. At 1%, nano-jev has a smaller seed variation than the MLP (3.3 against 10.5). But it also escalates fewer frames (1.6% against 9.3%). It stays near the classifier.

## Limits

- The MLP uses the validation loss for early stopping. nano-jev and Laya train for a fixed number of steps. This difference can help the MLP.
- Laya has only one seed for each size.
- The frames are a random sample of all train towers. A real drone sees a small number of towers. A sample by tower is a harder test.

## Consequences

- The data-efficiency argument for the decision models does not hold for this task and this evidence.
- One drone alone needs more than about 5% of the train-tower frames before a learned controller helps. This supports federated learning: the fleet together has more labelled frames than one drone.
- The controller for the federated step is the choice MLP. The decision models stay as centralised ablations, unless plan F (free-text intents) or plan B (unseen components) shows an advantage.
