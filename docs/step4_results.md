# Step 4 results: learned controllers (central training)

Date: 2026-11-05. Status: complete. The replay calibrates each estimator on the validation towers (Platt scaling, one map for each tool).
Source files: `results/step4/numeric_train.json`, `results/step4/metrics.json`. Each replay value is the mean of 5 seeds on the 7,057 test-tower items. Jetson factor: 9. Risk weights: 10 / 1 / 0.5 (`docs/risk_weights.md`).

## Method

1. Each controller first runs the classifier on the frame.
2. A quality estimator gives q(item, tool) for `classifier`, `smolvlm_local` and `cloud`. q is the probability that the tool gives the correct answer.
3. The estimator uses only data that the drone has before the escalation: the task type, the component, and the classifier output.
4. A constant rule calculates the expected risk of each tool: (1 − q) × wrong-answer cost, plus λ × latency. The wrong-answer cost is 10 × P(defect) + 1 × P(good), from the classifier probabilities.
5. The rule rejects a tool that cannot meet the deadline or that has no link. Then it selects the tool with the lowest value.
6. The estimators train on the train towers. We tune λ on the validation towers for each regime.

## Quality estimators

AUROC on the test towers (how well q separates correct and wrong answers of each tool):

| Estimator | classifier | smolvlm_local | cloud |
|---|---|---|---|
| Logistic regression | 0.957 | 0.847 | 0.841 |
| MLP (2 × 64) | 0.964 | 0.853 | 0.874 |
| nano-jev (33M, LoRA) | 0.961 | 0.842 | 0.866 |
| Laya (421M, full fine-tuning) | 0.961 | 0.843 | 0.858 |

After calibration, the ECE of each estimator is between 0.009 and 0.040 on the test towers.

## Replay (weak regime)

| Policy | Risk / 1,000 | Defect recall | False alarm rate | Escalation rate | Uplink MB / 1,000 items |
|---|---|---|---|---|---|
| Risk rule + true q (upper bound) | 62.9 | 0.952 | 0.002 | 0.031 | 0.17 |
| **Risk rule + MLP q** | **136.4** | 0.907 | 0.015 | 0.061 | 0.88 |
| Risk rule + Laya q | 173.9 | 0.876 | 0.010 | 0.072 | 1.83 |
| Risk rule + nano-jev q | 176.0 | 0.878 | 0.011 | 0.018 | 0.19 |
| Static cascade, θ = 0.95 | 155.7 | 0.896 | 0.015 | 0.144 | 0 |
| Classifier only | 177.0 | 0.888 | 0.026 | 0 | 0 |
| Risk rule + logistic q | 175.1 | 0.881 | 0.012 | 0.015 | 0.15 |

## Findings

1. The risk rule with the MLP estimator is the best deployable controller. Its risk is 12% less than the static cascade (136.4 against 155.7). It escalates 6.1% of the frames. The static cascade escalates 14.4%. The result is almost the same in the three regimes (135.5 to 136.4).
2. With the true q, the same rule gives the oracle result. Thus the rule is correct. The gap to the upper bound comes from the estimator.
3. The decision models (nano-jev and Laya) do not decrease the risk. Their risk (171 to 180) is almost equal to the classifier alone (177.0).
4. The AUROC of each tool is almost the same for the MLP, nano-jev and Laya (difference 0.01 or less). Thus a good AUROC for each tool does not give a good controller.
5. The rule uses the difference between the tools for one frame. nano-jev ranks the frames where the onboard VLM helps better than the MLP (AUROC 0.815 against 0.732 on the test towers). But the highest-ranked frames of nano-jev also contain harmful escalations: defect frames that the classifier answers correctly and that the VLM changes to "good". Each such frame costs 10 risk points. On the validation towers, escalation of the top 2% of nano-jev frames increases the risk from 235 to 272.
6. The text input of the decision models has less information than the MLP input. It contains only the two highest classifier answers, with probabilities rounded to 2 decimals. 96% of the validation states have the same text as a training state. nano-jev gives only 638 different q values for the classifier on 7,057 test frames. The MLP gives 3,948.
7. The logistic estimator is worse than the static cascade. The relation between the features and the tool correctness is not linear.
8. The tuned λ is 0 for the MLP. Thus the deadline and link checks control the latency, and the latency term does not change the decision.

## Figure

`step4-controllers/figures/escalation-budget.html` (artifacts directory) shows the risk against the share of escalated frames for each estimator. The data are in `step4-controllers/data/escalation-budget.csv`. This view is an offline replay without the queue and without the network.

## Next

- Give the decision models a target that matches the decision. Use one choice question with the options accept, onboard VLM and cloud. Use the risk of each option as the soft target. A decision model selects an option. It does not estimate three separate probabilities.
- Give the decision models the full evidence: all option probabilities with more decimals, the entropy, and the image-quality signals.
- Train the MLP on the same target, so that the comparison stays fair.
