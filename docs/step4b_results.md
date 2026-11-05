# Step 4b results: decision models with the choice target

Date: 2026-11-05. Status: complete except the Laya run with the selected setting.
Source files: `results/step4b/metrics.json` (grid and rank threshold), `results/step4b/metrics_t1.json` (first run with constant margins).
Each value is the mean of 5 network seeds on the 7,057 test-tower frames, weak regime, Jetson factor 9.

## Changes from step 4

1. **Target.** One choice between `accept`, `smolvlm_local` and `cloud`. The soft target is softmax(−(risk + cost) / T). The risk of an option is the realised risk of the tool answer. The cost is 0, 0.1 and 0.2.
2. **Evidence.** The 6 classifier class probabilities, the log-odds of "good", and the image size, blur, brightness and contrast. The decision models read this as text (about 130 tokens). The MLP reads the same facts as numbers.
3. **Threshold.** The controller escalates when p(tool) − p(accept) is more than a threshold. We tune the threshold on the validation towers. The candidates are quantiles of the validation scores of the same model.
4. **Grid.** Temperature T in {1, 0.25} and copies of each informative training frame in {1, 5}. An informative frame is a frame where the risks of the options differ. We select the setting of each model by the validation risk only.

## Results

| Controller | Setting | Risk / 1,000 | Defect recall | Escalation rate |
|---|---|---|---|---|
| Oracle | — | 62.9 | 0.952 | 0.031 |
| **Choice MLP** | T 1, 5 copies (selected) | **114.2** | 0.922 | 0.108 |
| Choice MLP | T 1, 1 copy | 117.0 | 0.921 | 0.066 |
| Choice nano-jev | T 1, 5 copies (selected) | 125.3 | 0.921 | 0.299 |
| Correctness MLP (step 4) | — | 136.4 | 0.907 | 0.061 |
| Choice nano-jev | T 1, 1 copy | 137.9 | 0.911 | 0.211 |
| Choice MLP | T 0.25, 1 copy | 139.7 | 0.900 | 0.045 |
| Choice nano-jev | T 0.25, 5 copies | 144.7 | 0.906 | 0.246 |
| Static cascade | θ = 0.95 | 155.7 | 0.896 | 0.144 |
| Choice Laya | T 1, 1 copy | 159.1 | 0.892 | 0.235 |
| Choice nano-jev | T 0.25, 1 copy | 189.3 | 0.876 | 0.243 |

## Findings

1. The choice target is the largest improvement. It decreases the MLP risk from 136.4 to 117.0.
2. The new evidence does not decrease the risk of the MLP.
3. With constant margins, nano-jev does not escalate (risk 177.0) and Laya escalates 1.2% (risk 166.4). Their outputs stay near the prior. The rank-based threshold removes this problem.
4. The MLP and nano-jev select the same setting. The temperature 0.25 is worse for both.
5. nano-jev is 10% worse than the MLP (125.3 against 114.2). It escalates about 3 times more frames. It escalates only to the onboard VLM.
6. The results in the good and poor regimes are within 3 risk points of the weak regime.

## Open item

- Laya with T 1 and 5 copies.
