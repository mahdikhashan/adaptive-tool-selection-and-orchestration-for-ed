# Feedback model for the federated controller

Date: 2026-11-08. Code: `src/fedorch/feedback.py`, `scripts/fc2_feedback.py`.

## 1. Clients

A client is a region. A region is an operator with its own review team and its own ground station. Its images and labels stay in the region. The code makes 5 regions from the 18 train towers with `tower_buckets`. Two regions have one large tower each. The other three regions have 5 or 6 towers.

The server holds the val towers. It uses them to select the decision margin and the cascade threshold. It does not use them to train the controller.

## 2. Actions

The controller selects one action for each frame:

1. `accept`: keep the answer of the onboard classifier.
2. `smolvlm_local`: ask the onboard VLM.
3. `cloud`: send the frame to the cloud VLM.

## 3. What each action shows to the learner

| Action | Feedback |
|---|---|
| `accept` | No label. An audit can send the frame to the operator with probability λ. |
| `smolvlm_local` | No label. An audit can send the frame to the operator with probability λ. |
| `cloud` | The answer of the cloud VLM. The operator reviews the frame with probability ρ = 0.5. |

When a frame goes to the ground station, the station runs all tools on it. If the operator gives the true label, the learner thus gets the risk of all 3 actions for that frame.

The cloud VLM is the weakest tool (accuracy 0.792, corrosion recall 0.012). Its answer is a noisy label.

## 4. Behaviour policy

In round 0, each region uses the static cascade. After round 0, each region uses the action with the highest probability from its current controller. In each round, a uniform random action replaces the selected action with probability ε = 0.05. The code records the probability of each action.

## 5. Learners

| Learner | Training data |
|---|---|
| `full` | The true label for each frame. This is the upper limit. |
| `cloud_oracle` | The operator labels, and the cloud answer as a true label for the other cloud frames. |
| `truth_naive` | The operator labels only, with weight 1. |
| `truth_ipw` | The operator labels only, with weight 1 / P(label). P(label) = π(cloud) ρ + (1 − π(cloud)) λ. The minimum P(label) is 0.02. |
| `multifid` | The operator labels with the IPW weight. For the other cloud frames, a soft target. The soft target uses P(true label given cloud answer). The code counts this probability from the frames that have both labels. |

## 6. Rounds and federation

Each region sees its frames in tower order, in 10 rounds. After each round, the controller trains on all the labels that the region has so far.

1. `fed`: 20 FedAvg steps per round. Each region does 10 Adam steps per FedAvg step.
2. `local`: each region trains alone with 200 Adam steps per round.

## 7. Evaluation

The learner sees only the feedback in section 3. The evaluation uses all cached tool answers and the true labels of the test towers. The replay uses the weak network and 5 network seeds.
