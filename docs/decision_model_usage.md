# How the experiments use the decision models

Date: 2026-11-05. This document lists each use of nano-jev and Laya: the input, the expected output, the training target and the result.

## 1. The two models and their interface

| Model | Checkpoint | Input | Output |
|---|---|---|---|
| nano-jev | `sdmlai/nano-jev` v1.0, 33.4M | Pairs of text: `question: {question} option: {option}` and a context text | One score (logit) for each pair |
| Laya | `convaiinnovations/laya`, 421M | A state text and a set of typed questions | For a `choice` question: one probability for each option. For a `noul` question: the probability of "yes" |

Neither model generates text. Each call gives numbers for the options that we supply.

## 2. The example frame

All examples use the test frame `test-01095`:

- Task: defect type question. Component: stockbridge damper. True label: rust.
- The classifier answers "good" with probability 0.506. This answer is wrong.
- `smolvlm_local` answers "rust" (correct). `cloud` answers "rust" (correct).
- Thus the correct decision is to escalate.

## 3. Use 1: smoke test (step 1)

- **Purpose:** check that the models load and run on the GPU. The models are not fine-tuned.
- **Script:** `scripts/step1_verify_assets.py` (`smoke`).
- **State text:** "Intent: detect rust or corrosion on clamps. Corrosion is critical. Abstain is allowed. Evidence: classifier says rust p=0.58, good p=0.31. Crop 212x180 px. Blur 0.21."
- **Question:** "Which action must the drone do for this frame?"
- **Options:** `accept_classifier`, `interpret_local`, `interpret_cloud`, `abstain`, each with a short description.
- **Output:** one probability for each option.
- **Result:** nano-jev needs 6.5 ms and Laya needs 23.9 ms for one decision on the GV100. The zero-shot answers do not agree (nano-jev: `abstain`, Laya: `accept_classifier`). The test does not use these answers.

## 4. Use 2: correctness of each tool (step 4)

- **Purpose:** estimate q(frame, tool), the probability that the tool gives the correct answer.
- **Scripts:** `scripts/step4_train_nanojev.py`, `scripts/step4_train_laya.py`. Text functions: `evidence` and `TOOL_TEXT` in `src/fedorch/features.py`.
- **Context (state) text for the example frame:**

  > Task: defect type question. Component: damper stockbridge. Classifier answer: good with probability 0.51. Second answer: rust with probability 0.49.

- **nano-jev input:** 3 pairs, one for each tool. The first text of a pair is:

  > question: Will this tool give the correct answer for this frame? option: ask the onboard vision-language model

  The other options are "accept the answer of the onboard defect classifier" and "ask the cloud vision-language model".

- **Laya input:** 3 `noul` questions on the same state, for example:

  > Will 'ask the onboard vision-language model' give the correct answer for this frame?

- **Target:** 1 if the tool answer is correct, 0 if not. For the example: classifier 0, onboard VLM 1, cloud 1.
- **Loss:** nano-jev: binary cross-entropy on each logit, LoRA rank 8, 2 epochs. Laya: its own `rlcd` loss, full fine-tuning, 2 epochs.
- **Expected output:** a high q for the tools that are correct.
- **Output for the example (before calibration):**

  | Estimator | classifier | onboard VLM | cloud |
  |---|---|---|---|
  | MLP | 0.467 | 0.719 | 0.339 |
  | nano-jev | 0.679 | 0.515 | 0.587 |
  | Laya | 0.959 | 0.956 | 0.949 |

- **Decision:** the risk rule in `src/fedorch/controller.py` calibrates q on the validation towers. Then it selects the tool with the lowest (1 − q) × wrong-answer cost + λ × latency.
- **Result:** risk 176.0 (nano-jev) and 173.9 (Laya). The MLP has 136.4. The decision models do not decrease the risk relative to the classifier alone (177.0).

## 5. Use 3: choice of the action (step 4b)

- **Purpose:** select the action directly: `accept`, `smolvlm_local` or `cloud`.
- **Scripts:** `scripts/step4b_train_nanojev.py`, `scripts/step4b_train_laya.py`. Text functions: `evidence_text` and `ACTION_TEXT` in `src/fedorch/decision.py`.
- **Context (state) text for the example frame:**

  > Task: defect type question. Component: damper stockbridge. Classifier classes: good 0.494, rust 0.482, corrosion 0.007, missing-cap 0.004, nest 0.007, other-defect 0.005. Answer options: good 0.506, rust 0.494, cannot-judge 0.000. Log-odds of good: -0.02. Image: 255 x 255 px, blur 699, brightness 0.67, contrast 0.18.

- **Question:** "Which action gives the lowest risk for this frame?"
- **Options:** "accept the answer of the onboard defect classifier", "ask the onboard vision-language model", "ask the cloud vision-language model".
- **nano-jev input:** 3 pairs, `question: … option: …` with the context text. A softmax over the 3 logits gives the probabilities.
- **Laya input:** 1 `choice` question with the 3 options as criteria.
- **Target:** a soft distribution, softmax(−(risk + cost) / T). The risk of an option is the risk of its tool answer (`docs/risk_weights.md`). The cost is 0, 0.1 and 0.2. For the example with T = 1: accept 0.000, onboard VLM 0.525, cloud 0.475.
- **Loss:** soft cross-entropy. nano-jev: LoRA rank 8, 3 epochs. Laya: `rlcd` with the soft target as `gold` probabilities, 2 epochs. Both use 5 copies of each informative training frame (selected on validation).
- **Expected output:** a higher probability for an escalation than for `accept` when the classifier is wrong and a VLM is correct.
- **Decision:** escalate when p(tool) − p(accept) is at least a threshold. We tune the threshold on the validation towers from quantiles of the scores of the same model (`choice_rule` in `src/fedorch/decision.py`).
- **Output and decision for the example:**

  | Model | accept | onboard VLM | cloud | Gap | Threshold | Decision |
  |---|---|---|---|---|---|---|
  | MLP | 0.213 | 0.414 | 0.372 | +0.201 | −0.017 | escalate (correct) |
  | nano-jev | 0.363 | 0.363 | 0.273 | 0.000 | −0.050 | escalate (correct) |
  | Laya | 0.363 | 0.348 | 0.289 | −0.014 | +0.153 | accept (wrong) |

  The outputs of the decision models stay near the prior. The MLP separates the options more.

- **Result:** risk 114.2 (MLP), 125.3 (nano-jev), 150.0 (Laya). The static cascade has 155.7.

## 6. Use 4: few labelled frames (plan A)

- **Purpose:** test whether the decision models learn better than the MLP from few labelled frames.
- **Input, target, loss and decision:** the same as use 3. The only difference is the size of the training set: 1%, 5%, 20% or 100% of the train-tower frames, with 3 seeds.
- **Training length:** nano-jev trains for at least 600 optimizer steps. Laya trains for at least 300 optimizer updates.
- **Result:** at 1% and 5%, nano-jev and the MLP are equal within the seed variation. At 20% and 100%, the MLP is better. Laya is worse at each size. See `docs/planA_results.md`.

## 7. Use 5: parser of operator intents (plan F)

- **Purpose:** change a free-text operator intent into a set of facts. The models are not fine-tuned. The test uses only Laya.
- **Script:** the zero-shot test in plan F. The output file is `results/planF/laya_zero_shot_parse.json`.
- **State text:** the intent, for example: "Please make sure that no general defects goes unreported. False reports waste crew time, so keep them rare."
- **Questions:** one `noul` question for each focus item, for example "Does the operator ask to give priority to general defects?". One more `noul` question: "Does the operator ask to avoid false alarms or false reports?"
- **Expected output:** "yes" (probability above 0.5) for each item in the focus set, and "yes" for the strict question if the intent is strict.
- **Output for this intent:** strict 0.843 (correct). General defects 0.247 (wrong, the true answer is yes). The other defect types are between 0.001 and 0.260.
- **Result:** the strict flag is correct for 56 of 56 intents. The exact focus set is correct for 13 of 56 intents.
- **nano-jev in plan F:** not trained. The value check showed that a perfect reader of the intent gains at most about 11 risk points. This value is equal to the gap between nano-jev and the MLP.

## 8. Summary

| Use | Step | Models | Target | Fine-tuned | Best decision-model risk | MLP risk |
|---|---|---|---|---|---|---|
| Smoke test | 1 | nano-jev, Laya | none | no | — | — |
| Correctness of each tool | 4 | nano-jev, Laya | correct or not, for each tool | yes | 173.9 (Laya) | 136.4 |
| Choice of the action | 4b | nano-jev, Laya | soft choice from the risk | yes | 125.3 (nano-jev) | 114.2 |
| Few labelled frames | plan A | nano-jev, Laya | soft choice from the risk | yes | equal to the MLP at 1% and 5% | — |
| Parser of intents | plan F | Laya | none | no | not a controller | — |

All risk values are for the weak regime on the 7,057 test-tower frames.
