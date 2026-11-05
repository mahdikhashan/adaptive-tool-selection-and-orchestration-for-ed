# Step 4 results: learned controllers (central training)

Date: 2026-11-05. Status: the numeric estimators are complete. The decision models (nano-jev, Laya) are not complete yet.
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

## Replay (weak regime)

| Policy | Risk / 1,000 | Defect recall | False alarm rate | Escalation rate | Uplink MB / 1,000 items |
|---|---|---|---|---|---|
| Risk rule + true q (upper bound) | 62.9 | 0.952 | 0.002 | 0.031 | 0.17 |
| **Risk rule + MLP q** | **137.8** | 0.907 | 0.015 | 0.057 | 0.71 |
| Static cascade, θ = 0.95 | 155.7 | 0.896 | 0.015 | 0.144 | 0 |
| Classifier only | 177.0 | 0.888 | 0.026 | 0 | 0 |
| Risk rule + logistic q | 181.2 | 0.878 | 0.013 | 0.018 | 0.39 |

## Findings

1. The risk rule with the MLP estimator is the best deployable controller. Its risk is 11.5% less than the static cascade (137.8 against 155.7). It escalates 5.7% of the frames. The static cascade escalates 14.4%.
2. The result is almost the same in the three regimes (risk 137.4 to 137.8).
3. With the true q, the same rule gives the oracle result. Thus the rule is correct. The gap to the upper bound (137.8 against 62.9) comes from the estimator.
4. The logistic estimator is worse than the static cascade. The relation between the features and the tool correctness is not linear.
5. The tuned λ is 0 for the MLP. Thus the deadline and link checks control the latency, and the latency term does not change the decision.

## Next

- The GPU runs of nano-jev and Laya add `q_nanojev.jsonl` and `q_laya.jsonl`. The same replay then compares them with the MLP.
