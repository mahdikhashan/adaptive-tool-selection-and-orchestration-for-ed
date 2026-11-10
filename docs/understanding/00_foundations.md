# Foundations: the concepts that all routes use

This file explains the basic ideas of the project. The reader is a first-year master student in AI with a background in mechanical engineering. Each concept has a link to a concept from mechanical engineering. The three route files (`01`, `02`, `03`) use these concepts and do not explain them again.

## 1. The problem in one paragraph

A drone flies along a power line and takes photos of components: insulators, shackles, dampers, clamps. For each photo, the system must say "good" or "defect" (and sometimes the type of defect). The drone has small AI models on board. A large AI model is available in the cloud, but the radio link is slow and sometimes down. The project asks: which model must the drone use for each photo, and what can a fleet of drones learn together?

Mechanical analogy: think of a quality inspection line in a factory. A fast visual check looks at each part. A slow ultrasonic test looks at the parts that the visual check cannot decide. A senior engineer looks at the hardest parts. The plant manager must decide which part goes to which test, with a limited time per part.

## 2. Classification

A classifier is a function that takes an input $x$ (here, an image) and gives a class $y$ (here, "good" or a defect type). Most modern classifiers give a probability for each class, not only one class.

- **Binary classification**: two classes, for example "defect: yes or no".
- **Multi-class classification**: more than two classes, for example "good, rust, corrosion, missing cap, bird nest, other".

The project uses both. The final decision is binary (report a defect or not), but some models also give the defect type.

## 3. The logistic function and the softmax

A model first calculates a real number for each class, the **logit** $z$. The logistic function changes one logit into a probability:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

For $K$ classes, the **softmax** changes $K$ logits into $K$ probabilities that add to 1:

$$p_k = \frac{e^{z_k / T}}{\sum_j e^{z_j / T}}$$

$T$ is the **temperature**.

Mechanical analogy: this is the Boltzmann distribution from thermodynamics. The logit has the role of a negative energy, and $T$ has the role of the temperature. A high $T$ makes the probabilities more equal (a hot gas fills all states). A low $T$ puts almost all probability on the best class (a cold system stays in the lowest energy state). The project uses this "temperature" to calibrate models (section 7).

## 4. Logistic regression

Logistic regression is the simplest classifier: $p = \sigma(w^\top x + b)$. It learns one weight for each input feature. The training minimises the **cross-entropy** (also called log loss):

$$L = -\frac{1}{n}\sum_i \left[y_i \log p_i + (1-y_i)\log(1-p_i)\right]$$

The cross-entropy is large when the model gives a low probability to the true class. A regularisation constant $C$ limits the size of the weights, which prevents overfitting.

Mechanical analogy: logistic regression is like a linear regression of a calibration curve, but the output is a probability between 0 and 1. The regularisation is like a stiffness that keeps the weights near zero when the data does not support large weights.

The project uses logistic regression to combine the answers of several models (route 03, "fusion").

## 5. Neural networks used in the project

| Network | What it is | Where the project uses it |
|---|---|---|
| Multilayer perceptron (MLP) | Layers of $h = \tanh(Wx + b)$, about 7,000 weights | Controllers and PPO policies |
| ResNet-50 | A convolutional network for images, 50 layers with skip connections | The fast classifier on the drone |
| Transformer | Layers of self-attention: each token is a weighted mean of all tokens | Inside all vision-language models |
| Vision-language model (VLM) | An image encoder plus a language transformer | SmolVLM-256M, laya-vision, Qwen3-VL-8B |

Mechanical analogy for the MLP: it is a general nonlinear transfer function with adjustable coefficients, like a response surface in design optimisation. Training adjusts the coefficients to fit the data.

Mechanical analogy for attention: each token calculates a weighted average of the other tokens. The weights depend on how similar the tokens are. This is like a weighted average of sensor readings, where the weight of each sensor depends on how relevant it is to the current reading.

## 6. Risk: the cost function of the project

Accuracy is not a good metric here, because about 85% of the items are "good". A model that always says "good" has an accuracy of 85% but it finds no defect. Thus the project uses a **risk** with costs:

| Answer | True state | Cost |
|---|---|---|
| "good" | defect (a miss) | 10 |
| "defect" | good (a false alarm) | 1 |
| "cannot judge" or "send to the operator" | any | 0.5 |
| correct answer | any | 0 |

The project reports the risk per 1,000 items. Lower is better.

Mechanical analogy: this is a safety factor in the cost function. A missed crack in a load-carrying part costs much more than an unnecessary maintenance visit. The weights 10, 1 and 0.5 are assumptions of the project (`docs/risk_weights.md`).

**The best threshold.** Let a model give a calibrated defect probability $q$. Then the expected cost of "good" is $10q$, and the expected cost of "defect" is $1-q$. "Defect" is better when $1 - q < 10q$, that is $q > 1/11 \approx 0.09$. Thus the system reports a defect already at 9% probability. Deferring to the operator is better than both answers when $\min(10q, 1-q) > 0.5$, that is for $q$ between 0.05 and 0.5.

## 7. Calibration

A model is **calibrated** if its probabilities are correct frequencies. Of all items with $q = 0.3$, about 30% must have a defect. The **expected calibration error (ECE)** measures the mean gap between the probability and the true frequency.

The threshold of section 6 is correct only for calibrated probabilities. Two methods make probabilities calibrated:

- **Temperature scaling or Platt scaling**: fit one or two numbers on held-out data, for example $T$ in the softmax.
- **Prior correction**: change the probabilities when the share of defects in the new data is different from the training data (section 8).

Mechanical analogy: calibration is the same as the calibration of a strain gauge. A gauge with a good resolution but an offset gives correct ranks of the loads but wrong values. A threshold on the wrong values gives wrong decisions.

## 8. Label shift and the EM prior correction

The share of defects changes from tower to tower. In this data, the validation towers have 19.6% defects and the test towers have 15.1%. A model trained on one share gives probabilities that are too high or too low on the other share.

Bayes' rule gives the correction. Let $\pi_v$ be the training share and $\pi$ the new share:

$$q = \frac{p\,\pi/\pi_v}{p\,\pi/\pi_v + (1-p)(1-\pi)/(1-\pi_v)}$$

The new share $\pi$ is not known, because the new data has no labels. The EM procedure (Saerens et al., 2002) estimates it in a loop:

1. Start with $\pi = \pi_v$.
2. Correct all probabilities with the current $\pi$.
3. Set $\pi$ to the mean of the corrected probabilities.
4. Do steps 2 and 3 again until $\pi$ does not change.

Mechanical analogy: this is like successive substitution in a nonlinear solver. The estimate of the share and the corrected probabilities depend on each other, and the iteration finds the point where they agree.

## 9. Train, validation and test data, and towers

The data (ElecVQA-Bench, `tower14` split, built on InsPLAD images) has three parts:

| Part | Towers | Items | Use |
|---|---|---|---|
| Train | 18 | 25,499 | Fit the models |
| Validation | 10 | 6,672 | Select settings, fit thresholds and the fusion |
| Test | 14 | 7,057 | Report the results, one time only |

The split is **by tower**: the images of one tower are in one part only. This is important because images of the same tower look similar. A split by image would let the model learn the tower, not the defect.

Mechanical analogy: this is like a validation on a different test rig. A model that works only on its own calibration rig is not useful.

**Cross-validation by tower**: the project must sometimes fit and test on the same data without leakage. It divides the towers into 5 groups, fits on 4 groups and predicts the fifth group. The predictions are **out-of-fold**.

## 10. Federated learning (FedAvg)

In federated learning, each client (a drone or an operator region) keeps its data. The clients train copies of one model, and a server combines the copies. **FedAvg** (McMahan et al., 2017) does this in rounds:

1. The server sends the global weights $\theta$ to each client.
2. Each client $d$ trains on its own data and gets $\theta_d$.
3. The server calculates the weighted mean $\theta \leftarrow \sum_d n_d\,\theta_d / \sum_d n_d$, where $n_d$ is the data size of client $d$.

The **uplink** is the data that the clients send: the size of the model update times the number of clients and rounds. For a drone, the uplink is the expensive resource.

FedAvg becomes worse when the clients have different data (**non-IID** data). **FedProx** adds a term that keeps each client near the global weights.

Mechanical analogy: several laboratories measure the same material property with different specimens. Each laboratory sends its fitted coefficients, not its raw data. The coordinator takes a weighted mean. If the laboratories test very different specimens, the mean can be a bad fit for all of them.

## 11. Reinforcement learning: MDP, policy, reward

Reinforcement learning (RL) learns decisions by trial and error. The problem is a **Markov decision process (MDP)**:

| Element | Meaning | In this project |
|---|---|---|
| State $s$ | What the agent knows now | Current defect probability, time left, bandwidth |
| Action $a$ | What the agent can do | Call a tool, report, defer |
| Reward $r$ | The feedback after an action | Minus the risk of the answer |
| Policy $\pi(a \mid s)$ | The probability of each action in each state | A small MLP |
| Episode | One sequence from start to end | One image, or 40 images in a row |

The agent maximises the expected sum of rewards. A **constrained MDP** adds limits, for example a time budget or a maximum number of deferrals. The project enforces the limits with an **action mask**: an action that breaks a limit gets probability zero.

Mechanical analogy: the MDP is the state-space model of control engineering. The state is the state vector, and the action is the control input. The reward is the negative of the cost in an optimal control problem (like LQR). The action mask is like actuator saturation: some inputs are not physically possible.

## 12. PPO, advantage and GAE

PPO (proximal policy optimisation, Schulman et al., 2017) improves the policy step by step:

1. Run the policy and collect episodes.
2. For each action, estimate the **advantage** $\hat A$: was the action better or worse than the average action in that state? A second network, the **critic** $V(s)$, predicts the average value.
3. Increase the probability of actions with a positive advantage and decrease it for a negative advantage.

PPO limits the size of each step. The ratio $\rho = \pi_{\text{new}}(a \mid s) / \pi_{\text{old}}(a \mid s)$ stays between 0.8 and 1.2:

$$L = \mathbb{E}\left[\min\left(\rho \hat A,\ \mathrm{clip}(\rho, 0.8, 1.2)\,\hat A\right)\right]$$

**GAE** (generalised advantage estimation) uses the one-step errors $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$. The advantage is their sum with the weight $(\gamma\lambda)^l$ for step $t+l$.

Mechanical analogy: the clip in PPO is like the under-relaxation factor in an iterative CFD or FEM solver. A full step can make the solution unstable, so the solver takes only a part of the step. GAE is like an exponential smoothing filter on a noisy signal: it reduces the noise but adds a small bias.

## 13. Imitation learning

In imitation learning (behaviour cloning), the policy copies the actions of an expert. The training is a normal classification: the state is the input, and the expert's action is the label. The project uses a hand-made rule as the expert, and then starts PPO from the copy (a **warm start**).

Mechanical analogy: an apprentice first copies the moves of an experienced welder. Later, the apprentice improves the technique by trial and error. If the trials are noisy, the apprentice can also lose the good habits. An **anchor** term in the loss keeps the policy near the expert.

## 14. Model size, LoRA and quantisation

- **Full fine-tuning** changes all weights. **LoRA** adds a small low-rank correction $W + BA$ to some weight matrices and trains only $A$ and $B$.
- **Quantisation** stores the weights with fewer bits: fp32 (32 bits), fp16 (16 bits), int8 (8 bits), nf4 (4 bits).

Mechanical analogy for LoRA: a large structure gets a small set of added stiffeners. The stiffeners change the behaviour in some directions, and the rest of the structure stays the same. Quantisation is like a sensor with a lower resolution. The coarse steps are acceptable if the important differences are larger than the steps.

## 15. Metrics in the project

| Metric | Meaning | Good value |
|---|---|---|
| Risk per 1,000 items | Cost from section 6 | Low |
| Recall | Share of the true defects that the model finds | High |
| False-alarm rate | Share of the good items that the model calls defective | Low |
| AUROC | Probability that a random defect gets a higher score than a random good item | 1.0 is perfect, 0.5 is random |
| Macro F1 | Mean of the F1 score of each class, so rare classes count as much as frequent classes | High |
| ECE | Calibration error | Low |
| Uplink | Bytes that the drones send | Low |
| Latency | Time for each item | Low |

## 16. Further reading in the repository

- `docs/risk_weights.md`: why the weights are 10, 1 and 0.5.
- `docs/architecture.md`: the software architecture.
- `docs/research_direction.md`: the anchor papers and the research question.
