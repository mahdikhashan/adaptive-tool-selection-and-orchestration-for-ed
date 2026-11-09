# Tool selection with PPO: results (November 2026)

This document records all experiments on the branch `agentic-ppo-v2`.
There are two questions:

1. Can a learned policy pick perception tools for an inspection drone, under a time budget?
2. Can a fleet train this policy with federated learning?

## Summary

1. A VLM that reads tool outputs and writes the report gets worse with tools. This is true for fine-tuned laya-vision and for base laya-vision. It is true when PPO trains only the scorer, and when PPO trains the 8.65M-parameter decision head.
2. Tools help when the system fuses their own answers. A logistic fusion of the four tools gives a test risk of 107.1 per 1,000 items. The best single tool gives 177.0. A label-free threshold (EM prior correction) recovers almost all of the threshold gap.
3. Test-time zoom does not help. The missed defects are small source crops. Zoom adds no pixels.
4. PPO tool selection learns a budget-aware policy. It gets the risk of "all tools" (105.4) with about 30% less time. It does not beat a simple uncertainty cascade (106.3).
5. FedAvg PPO is equal to central PPO and has less variance across seeds. One update is 6,916 parameters (28 KB in fp32).
6. Drones with different components and networks do not change result 4. Personalized FedAvg (FedAvg and then local fine-tuning) is the most stable learned method. It is equal to the cascade in risk and uses 15% less time.
7. The oracle that knows the best tool subset for each item gets 65.0. The gap to the oracle (about 40 points) is not a problem of the RL algorithm. The policy state does not tell which tool will correct which item.

## Setup

- Data: ElecVQA-Bench `tower14` on InsPLAD crops. Val towers: 6,672 items. Test towers: 7,057 items.
- Tools:
  - ResNet-50 classifier, fine-tuned.
  - laya-vision, fine-tuned (`good300_full`).
  - SmolVLM-256M with LoRA.
  - Qwen3-VL-8B in the cloud, zero-shot.
- Risk per 1,000 items: 10 for a missed defect, 1 for a false alarm, 0.5 for an abstention.
- Times: GPU times multiplied by 9 for onboard tools. The factor 9 is an estimate for a Jetson.
- Network: the Qwen call sends a JPEG over an LTE trace. The `weak` regime has 10% of the trace bandwidth and six 20 s outages per hour.

## Experiments

### 1. Pilot: fine-tuned laya reads tool evidence

- Run: the quick stage of `scripts/runs/ppo_v2_gpu.sh`, about 1,000 images, on the VM.
- PPO learned to call no tool. The val risk was 318, equal to laya alone.
- Each tool changed more correct laya answers to wrong answers than wrong answers to correct answers (about 1.5 times more).
- The controller trained on images that laya saw in fine-tuning. Train risk was 122 and val risk was 318.
- The result files of this pilot stay on the VM.

### 2. Which tool answer to trust

- Data: the step-2 item cache. Fit on val towers, score on test towers.
- Selector 133, fusion 123, best single tool (ResNet) 177, cascade 156, oracle 38 to 41.

### 3. Three-question report from all tools

- Files: `scripts/report3_tools.py`, `scripts/report3_check.py`, `results/report3/check.json`.
- 2,000 val images and 2,000 test images. Each tool answers the three laya questions (component, defect, condition).
- Component: laya 0.872, fusion 0.905, oracle 0.925.
- Defect: ResNet alone 105.5 is best. Fusion 119.0.
- SmolVLM fails outside its training prompt (component accuracy 0.14). Qwen answers "defect" for almost all images (false-alarm rate 0.95).

### 4. Base laya as the agent (PPO on the scorer)

- Files: `scripts/agent_cache.py`, `src/fedorch/laya_agent.py`, `scripts/agent_ppo.py`, `results/agent_ppo/seed0/summary.json`.
- Base laya-vision in 4-bit reads the image and the tool outputs, picks the next action, and writes the report. Tools: zoom (2/3 crops), ResNet, Qwen. PPO trains the option scorer on frozen cached vectors. 800 train images, 400 test images, budgets 1, 3 and 6 s, seed 0.
- Report cost per 1,000 at round 10, 1 s budget:

| Method | Report cost | Defect recall | Steps |
|---|---|---|---|
| Zero-shot | 1,668 | 0.57 | 1.0 |
| PPO, report only | 1,505 | 0.77 | 1.0 |
| Central PPO with tools | 1,715 | 0.50 | 4.9 |
| FedAvg PPO with tools | 1,628 | 0.55 | 6.7 |

- With tools, the report is worse than without tools. Component accuracy stays near 0.15.

### 5. Separate scorers for action and report

- Files: `results/agent_ppo/split_seed0/summary.json`.
- No improvement. Report only: 1,592. Central: 1,765 (1 s) and 1,645 (3 s). FedAvg: 1,665 (1 s) and 1,825 (3 s).
- The policy does not learn to stop. It uses about 7 steps, which is the maximum.

### 6. PPO trains the laya decision head (option 2)

- Files: `scripts/agent_cache_h.py`, `src/fedorch/laya_head_agent.py`, `scripts/agent_ppo_head.py`, `results/agent_ppo_head/seed0/summary.json`.
- PPO trains the head transformer and two scorers (8.65M parameters) on cached backbone outputs.
- Mean of the last 3 rounds: no tools 1,461. Central with tools 1,518 (1 s) and 1,572 (3 s). FedAvg 1,712 at all budgets.
- Central PPO learns to stop. Steps go from 6 to 1.2 at 1 s and to 2.0 at 3 s. This is the first budget-dependent behavior.
- FedAvg learns slowly. Uplink: 8.65 MB per drone per round in int8, 432 MB for 10 rounds. Time: 74 min for FedAvg, 14 min for central.
- Figure: `artifacts/agentic-laya/figures/head_ppo.svg`.

### 7. Gap analysis of item-level fusion

- Test towers: 7,057 items with 1,068 defects. Fusion 145, oracle 25.
- Most of the gap is false alarms: 725 false alarms against 30 misses.
- The val threshold is too low for the test towers. The val defect share is 0.196 and the test share is 0.151.
- 31 defects get no flag from any tool. 16 of the 30 misses are rust on `polymer-insulator-lower-shackle`.

### 8. Fusion with a label-free threshold

- Files: `scripts/fusion_check.py`, `results/fusion/check_good300_full.json`.
- Logistic fusion of P(defect) from the four tools, with task and component. Cross-validation on val towers chose C = 0.03 and the simple features.

| Method (test towers) | Risk per 1,000 | 95% CI |
|---|---|---|
| ResNet alone | 177.0 | 148 to 206 |
| laya alone | 261.0 | 225 to 294 |
| Fusion, threshold tuned on val | 115.8 | 100 to 133 |
| Fusion, Bayes threshold 1/11 | 113.8 | 99 to 130 |
| Fusion, Bayes after EM prior correction | 107.1 | 91 to 124 |
| Fusion, EM per region | 117.5 | 100 to 137 |
| Fusion, test-optimal threshold (labels) | 105.4 | 89 to 124 |
| Oracle best tool (labels) | 44.8 | 30 to 60 |

- EM estimates a test defect share of 0.154. The true share is 0.151.

### 9. Zoom check

- Files: `scripts/zoom_check.py`, `scripts/zoom_analyze.py`, `results/zoom/analysis.json`.
- ResNet on the full image and on 5 overlapping 2/3 crops. Test risk (image level): full image 110.0. Best zoom rule 126.9. Other rules 142 to 194.
- Lower-shackle recall: 0.80 for the full image, 0.68 to 0.81 with zoom.
- Missed lower-shackle defects have a median of 62k pixels. Found defects have a median of 110k pixels.
- Decision: zoom is not a PPO action. Image size is a state feature.

### 10. PPO tool selection with fused reports

- Files: `src/fedorch/route_env.py`, `scripts/route_ppo.py`, `results/route/summary_lambda0.01.json`, `results/route/summary_lambda0.03.json`.
- Environment:
  - ResNet runs first.
  - Actions: call laya, call SmolVLM, call Qwen, or report.
  - The report is the fusion of the answers so far, with the EM threshold.
  - A tool is available only if it fits in the remaining budget.
  - Reward: minus the risk, minus lambda times the time used divided by the budget.
- Training: on val items, with out-of-fold fusion. Test: on test items. Three seeds, 30 rounds of 2,048 episodes.
- Actor: MLP 64-64 with 6,916 parameters. FedAvg uses five drones from tower groups.
- With lambda 0.2, PPO learns to call no tool. The time penalty is larger than the mean gain.

| Method, budget 3 s | Risk | Time (s) | Qwen calls |
|---|---|---|---|
| ResNet only | 146.9 | 0.07 | 0 |
| All tools that fit | 105.4 | 1.81 | 0.92 |
| Uncertainty cascade | 106.3 | 1.23 | 0 |
| Central PPO, lambda 0.01 | 105.9 ± 1.5 | 1.26 | 0.58 |
| FedAvg PPO, lambda 0.01 | 105.7 ± 1.8 | 1.79 | 0.92 |
| Central PPO, lambda 0.03 | 104.8 ± 3.4 | 1.21 | 0.61 |
| FedAvg PPO, lambda 0.03 | 105.7 ± 1.8 | 1.31 | 0.53 |
| Oracle subset (labels) | 65.0 | 0.11 | 0.02 |

- At 0.75 s: central 129.5 ± 8.7, FedAvg 123.4 ± 0.8, cascade 124.3.
- At 0.25 s only Qwen fits. Qwen makes the risk worse (150.6 against 146.9).
- Figure: `figs/fig_route.pdf`.

### 11. Drones with different components and networks

- Files: `scripts/route_drones.py`, `results/route/drones_components_lambda0.01.json`, `results/route/drones_towers_lambda0.01.json`.
- Five drones. Split `components`: each drone inspects one component group (glass, polymer, hardware, yoke, rod and grip). Split `towers`: each drone has a tower group. Network regimes per drone: poor, good, weak, poor, good.
- The fusion and the EM prior are the same for the full fleet. EM per drone failed: it estimated 12% defects for the hardware drone, and the true share is 0.8%.
- New baselines: a cascade tuned on val with one rule for the fleet, and a cascade tuned per drone. The grid has the tool order, the margin and the maximum number of tools.
- New methods: local PPO (each drone alone) and FedAvg with 10 rounds of local fine-tuning.

| Split `components`, mean of 3 seeds | 0.75 s | 1.5 s | 3 s |
|---|---|---|---|
| Static cascade (margin 0.1) | 126.5 | 109.3 | 106.3 |
| Cascade tuned per drone | 127.4 | 120.2 | 117.1 |
| Central PPO | 138.1 ± 8.3 | 116.1 ± 6.1 | 107.7 ± 1.4 |
| Local PPO | 131.7 ± 6.9 | 110.0 ± 2.0 | 107.1 ± 2.3 |
| FedAvg PPO | 126.5 ± 0.0 | 113.0 ± 5.4 | 110.8 ± 4.6 |
| FedAvg + local fine-tune | 126.6 ± 0.1 | 112.0 ± 2.6 | 107.9 ± 0.9 |
| Oracle subset (labels) | 105.1 | 73.6 | 66.8 |

- At 3 s, FedAvg with fine-tuning uses 1.05 s per item. The static cascade uses 1.23 s.
- The cascade tuned per drone is worse than the static cascade. The val and test defect shares of a drone are different (glass: 0.35 on val, 0.20 on test).
- The split `towers` gives the same order of methods.
- Figure: `figs/fig_drones.pdf`.

## Conclusions

1. Do not use a VLM as the agent that reads tool outputs. Use the tools' own answers and fuse them.
2. The label-free threshold is necessary. The defect share changes between towers.
3. PPO and FedAvg PPO learn a correct, budget-aware policy, but a one-line cascade is as good in this setting.
4. Federated training of the policy costs KB per round. Federated training of the perception model costs MB to GB per round.
5. To beat the cascade, the policy needs a per-item signal that predicts which tool corrects the item.

## Next steps

1. Add per-item features: disagreement between tools, image size per component, and the tool reliability per component.
2. Run the PPO with Flower (Ray backend) on the VM, with the 4-bit laya times.
3. Add the three-question report to the environment.
