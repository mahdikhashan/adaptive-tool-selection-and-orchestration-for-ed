# Findings, steps 1 to 4b, plan A and plan F

Date: 2026-11-05. This document collects all findings to this date. Each section gives the source files. The detailed documents are `docs/asset_verification.md`, `docs/step2_results.md`, `docs/step3_results.md`, `docs/step4_results.md`, `docs/step4b_results.md`, `docs/planA_results.md` and `docs/planF_results.md`.

Common settings: 7,057 test-tower frames (14 towers). Risk weights 10 / 1 / 0.5 (`docs/risk_weights.md`). Replay values are the mean of 5 seeds in the weak network regime, with the Jetson factor 9, unless a finding gives a different setting.

## 1. Data and infrastructure (step 1)

Sources: `results/step1/*.json`, `docs/asset_verification.md`.

1. All controller checkpoints exist with open licences: nano-jev (33.4M, MIT), Laya (421M, Apache-2.0), Kev-4B (Apache-2.0) and pplx-decider-flash (4.5B, Apache-2.0).
2. ElecVQA-Bench gives tower-grouped splits: 18 train, 10 validation and 14 test towers. A tower has 11 to 6,449 items. About 88% of the items are "good". Rare defects occur only on some towers. Thus the towers give non-IID clients for federated learning.
3. The ElecVQA-Bench prompts are in Chinese. `src/fedorch/labels.py` changes all options to English labels without errors.
4. The InsPLAD archive contains three inner zip files. All 56,926 image paths of ElecVQA-Bench exist after the extraction.
5. The 40 LTE traces have a mean throughput of 31.4 Mbps. Only about 1% of the seconds are below 2 Mbps. Thus the network regimes need scaled bandwidth and added outages.
6. The PyPI build of torch 2.15 (CUDA 13.2) has no kernels for the GV100 (compute capability 7.0). Torch 2.14.1 for CUDA 12.6 runs on it.
7. On the GV100, nano-jev needs 6.5 ms and Laya needs 23.9 ms for one decision.

## 2. Tools (step 2)

Sources: `results/step2/summary.json`, `results/step2/tool_*.jsonl`, `docs/step2_results.md`. The tools are cross-fitted by tower.

8. The ResNet-50 classifier is the best single tool: accuracy 0.961 at 7.3 ms (GV100). The fine-tuned SmolVLM-256M gets 0.956 at 80.8 ms.
9. The zero-shot cloud VLM (Qwen3-VL-8B) is weak: accuracy 0.792. Its corrosion recall is 0.012. It is correct on many binary defect questions.
10. SmolVLM-256M without fine-tuning answers "good" for almost all frames. Its recall is below 0.04 for each defect label.
11. INT8 visual tokens decrease the accuracy by less than 0.1 point. At 512 px, the split payload (36,868 B) is almost equal to the JPEG payload (35,421 B).
12. The tools make different errors. SmolVLM is correct on 66.7% of the 279 classifier errors. The cloud VLM is correct on 48.4%.
13. The classifier confidence identifies its errors (AUROC 0.939). The calibration error is small (ECE 0.007).

## 3. Baselines in the replay (step 3)

Sources: `results/step3/metrics*.json`, `docs/step3_results.md`.

14. The static cascade (classifier, then SmolVLM below confidence 0.95) has a risk of 155.7. The classifier alone has 177.0. The cascade escalates 14.4% of the frames.
15. The oracle has a risk of 62.9 and escalates only 3.1% of the frames. Thus the problem is to select the correct frames, not to escalate more frames.
16. The EVLM pipeline (SmolVLM on each frame) has a risk of 188.3. Cloud on each frame has 648.5. In the poor regime, about 43% of the cloud answers are late.
17. With the Jetson factor 5, the results do not change. With the factor 35, onboard escalation is not possible, and the cascade becomes the classifier.

## 4. Learned controllers with a correctness target (step 4)

Sources: `results/step4/metrics.json`, `docs/step4_results.md`.

18. A rule on a learned correctness estimate q(frame, tool) with an MLP gives a risk of 136.4. This is 12% less than the static cascade.
19. The same rule with the true q gives the oracle result. Thus the rule is correct, and the gap comes from the estimator.
20. nano-jev (176.0) and Laya (173.9) with the correctness target are not better than the classifier alone. Logistic regression gives 175.1.
21. The AUROC of q for each tool is almost the same for the MLP, nano-jev and Laya (difference 0.01 or less). Thus a good AUROC for each tool does not give a good controller.
22. nano-jev ranks the frames where escalation helps well (AUROC 0.815, against 0.732 for the MLP). But its highest-ranked frames also contain harmful escalations of defect frames that the classifier answers correctly.

## 5. Learned controllers with a choice target (step 4b)

Sources: `results/step4b/metrics.json`, `results/step4b/metrics_t1.json`, `docs/step4b_results.md`.

23. A choice target is better than a correctness target. The MLP that learns the choice between accept, SmolVLM and cloud has a risk of 117.0. The MLP with the correctness target has 136.4.
24. The new evidence (6 classifier classes and image signals) does not help the MLP. With the correctness target, the MLP risk is 140.5 to 144.8 with the new evidence.
25. A threshold on a constant scale does not work for nano-jev. Its outputs stay near the prior, and the constant margins give no escalation (risk 177.0). A threshold from the quantiles of its own validation scores gives 137.9.
26. We did a grid search on the validation towers. The MLP and nano-jev select the same setting: soft-target temperature 1 and 5 copies of each informative training frame.
27. With this setting, the MLP has a risk of 114.2 and escalates 10.8% of the frames. nano-jev has a risk of 125.3 and escalates 29.9%. Thus nano-jev is 10% worse than the MLP, but it is 20% better than the static cascade.
28. A sharp soft target (temperature 0.25) is worse for both models.
29. Laya with the selected setting (temperature 1, 5 copies) has a risk of 150.0 and escalates 2.6% of the frames. With the first setting (no copies) it has 159.1. Thus Laya is better than the static cascade, but worse than nano-jev and the MLP.

## 6. Controllers with few labelled frames (plan A)

Sources: `results/planA/metrics.json`, `docs/planA_results.md`. 3 training seeds for each size (Laya: 1 seed).

30. With 1% and 5% of the train-tower frames, nano-jev and the MLP are equal within the seed variation. The mean paired difference is −0.2 and −2.8 risk points. nano-jev is better on 1 of 3 seeds.
31. With 20% and 100%, the MLP is better than nano-jev by 13.5 and 15.0 risk points.
32. Laya is worse than both controllers at every size (168.0 at 5%, 165.8 at 20%).
33. At 1% (about 255 frames), no controller learns a useful decision. The risk is about 172. The classifier alone has 177.0.
34. At 5%, the learned controllers are equal to the static cascade (155.7). They start to beat it between 5% and 20%, that is with more than about 1,300 to 5,100 labelled frames.
35. nano-jev is less stable than the MLP. At 20%, its standard deviation is 24.7 risk points.

## 7. Free-text operator intents (plan F)

Sources: `results/planF/*.json`, `docs/planF_results.md`. The intents are synthetic.

36. Without an escalation budget, the intent has almost no value for the decision. A perfect reader of the intent changes the risk by −3.1 to +0.5 risk points.
37. With a budget of 2% and defect-type intents, a perfect reader decreases the risk by 5% to 8%. The reader must combine the intent with the score in code. An MLP that gets the true intent as numbers does not learn this combination.
38. A bag-of-words MLP fails on new wording. Its risk is higher than the classifier alone.
39. Zero-shot Laya parses the strict flag correctly for all 56 intents. It parses the exact focus set for only 13 of 56 intents.
40. Plan F cannot show an advantage for nano-jev. The largest gain from the intent (11 risk points) is equal to the gap between nano-jev and the MLP.

## 8. Conclusions to this date

41. The best deployable controller is the choice-target MLP (risk 114.2). It is 27% better than the static cascade. Its gap to the oracle is 51 risk points.
42. With the same facts, a decision model does not beat the MLP. This is true with all labels and with few labels. The larger model (Laya, 421M) is worse than the smaller model (nano-jev, 33M). This agrees with arXiv 2610.00346: a fine-tuned classifier beat the decision models when task labels exist.
43. One drone alone needs more than about 5% of the train-tower frames before a learned controller beats the static cascade. The fleet together has more labelled frames. Thus federated learning has a clear purpose: it gives each drone the data that it does not have alone.
44. The controller for the federated step is the choice MLP. The decision models stay as centralised ablations. Plan F did not show an advantage. Plan B (unseen components) is the remaining test. A decision model can be useful as a parser of operator intents, with a rule in code. This is future work, and it needs real operator intents.

## 9. Limits

- The Jetson factor 9 is an estimate. No measurement on a Jetson exists.
- We constructed the network regimes from downlink traces.
- The risk weights are assumptions.
- The step 4 and step 4b controller results use one training seed. Plan A uses 3 training seeds. The replay uses 5 network seeds.
- In plan A, the MLP uses the validation loss for early stopping. The decision models train for a constant number of steps.
