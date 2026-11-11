# tools_v3: results of the Qwen inspection agent

Date: 2026-11-11. Branch: `agentic-ppo-v2`. Plan: `tool_corpus_plan.md`. Tests for later: `tools_v3_future_tests.md`.

## 1. Summary

- A small language model (Qwen3-0.6B) is the agent. It selects tools step by step and then writes a report with three fields: component, defect (yes or no), condition.
- Without training, the agent calls no tool and always writes "no defect". No report is fully correct.
- After SFT, 88% of the reports are fully correct. The agent calls one tool (laya-vision) and then writes the report.
- GRPO after SFT gives the same result as SFT. The model is almost certain after SFT, thus GRPO gets almost no learning signal.
- The best single tools are better than the agent on defect and condition. The agent does not call them.
- The time budgets (4, 10 and 30 s) do not change the result. The agent uses approximately 0.35 s for each image.
- The results with thinking (0.6B, 1.7B and 4B) are not complete.

## 2. Set-up

### 2.1 Data

| Item | Value |
|---|---|
| Source | InsPLAD component crops from the ElecVQA-Bench split `tower14` |
| Training images for the tools | the 18 training towers |
| Images for the agent data (val) | 5,624 images from 10 towers |
| Test images | 6,092 images from 14 towers. The agent test uses 1,500 of them (seed 0). |
| Components | 11 types |
| Conditions | good, rust, corrosion, missing cap, bird nest, other defect |
| Defect share on the test towers | 9.9% (601 of 6,092) |

### 2.2 Tools

There are 61 tools. 13 tools run on the drone, 48 tools run in the cloud.

| Group | Examples | Location | What the tool gives |
|---|---|---|---|
| Image classifiers | ResNet-18/50/101, EfficientNet-B0/B3, MobileNet-V2, ConvNeXt-T/S, ViT-B, DeiT-S, Swin-T, YOLO | drone and cloud | condition (thus also defect) |
| Variants | test-time flip, five crops, re-cut from the full photo | mostly cloud | condition or defect |
| Component specialists | ResNet-50 and ConvNeXt-T for each of 5 component groups | cloud | condition, only for their group |
| Feature probes | DINOv2-S/B, CLIP: linear condition classifier and one detector for each defect type | mostly cloud | condition or one defect type |
| Zero-shot | CLIP-L | cloud | condition |
| Vision-language models | laya-vision (full, 4-bit, re-cut), SmolVLM-256M (3 variants), Qwen3-VL-8B (7 questions) | drone and cloud | laya: all three fields. Others: defect or one defect type. |
| Image checks | quality, exposure, size | drone | a score, no answer |

Only the laya-vision tools give the component.

### 2.3 Time and network

- A tool on the drone uses its desktop GPU time multiplied by 9 (our estimate for a Jetson Orin).
- A tool in the cloud uses its GPU time, a round trip of 50 ms, and the upload of the image. The drone sends the image only one time for each image (a re-cut is a different image). The upload follows an LTE trace (weak network). A cloud tool is not possible when the link is down.
- Budgets: 4 s, 10 s, 30 s and no limit. An episode has a maximum of 8 steps. At step 8 the agent must write the report.

### 2.4 The agent

- Qwen3-0.6B (text only). The agent does not see the image. It sees the tool results as text.
- At each step, a BM25 retriever gives 15 tool candidates and "report". The agent writes JSON: `{"name": "SELECT", "ids": [tool]}` or `{"name": "REPORT", "component": ..., "defect": ..., "condition": ...}`.
- The method follows Zhou et al. (2026): retrieval, then selection, trained with SFT and GRPO. We do not use their plan stage.

### 2.5 Training targets

- Tool step: the gold tool is the tool whose own answer adds the most correct fields that are not correct yet. If two tools are equal, the faster tool is the gold tool. If no tool adds a correct field, the gold action is "report".
- Report step: the target is the true labels.
- Data: 61,461 steps from 22,496 episodes (5,624 val images, 4 budgets, 30% random exploration). Gold actions: report 30,511, laya 24,577, laya_recut 2,933, resnet 1,642, resnet18 804. Each other tool is the gold action in less than 300 steps.
- SFT: TRL `SFTTrainer`, LoRA r=16 on all linear layers, 12,000 steps, 1 epoch, learning rate 1e-4.
- GRPO: TRL `GRPOTrainer`, starts from the SFT adapter, 400 steps, 8 answers for each prompt, beta 0.02, learning rate 2e-5. Reward: 1 for the gold tool. For a report, the reward is the share of correct fields. An answer that is not valid gets 0.
- Base: Qwen3-0.6B with no training and the same instructions.

## 3. Results without thinking (Qwen3-0.6B)

1,500 test images, one seed, greedy decoding.

### 3.1 Accuracy

| Method | Budget | Component | Defect | Condition | All three | Time (s) | Upload (kB) | Tool steps |
|---|---|---|---|---|---|---|---|---|
| base | 4 s | 0.037 | 0.895 | 0.000 | 0.000 | 0.00 | 0.0 | 0.0 |
| SFT | 4 s | 0.919 | 0.961 | 0.958 | 0.881 | 0.35 | 34.1 | 1.0 |
| SFT | 10 s | 0.917 | 0.961 | 0.958 | 0.880 | 0.37 | 34.1 | 1.0 |
| SFT | 30 s | 0.915 | 0.959 | 0.957 | 0.876 | 0.41 | 33.7 | 1.0 |
| SFT + GRPO | 4 s | 0.919 | 0.961 | 0.958 | 0.881 | 0.35 | 34.1 | 1.0 |
| SFT + GRPO | 10 s | 0.917 | 0.961 | 0.958 | 0.880 | 0.37 | 34.1 | 1.0 |
| SFT + GRPO | 30 s | 0.915 | 0.959 | 0.957 | 0.876 | 0.41 | 33.7 | 1.0 |

The results without a time limit are not in the table. A fault in the code gave them an infinite time (see section 5).

Other values for SFT at 4 s: defect recall 0.885, recall of good images 0.970. Macro F1: component 0.832, condition 0.884.

### 3.2 Single tools

Accuracy of each tool alone on all 6,092 test images. A tool that gives no component has no component value.

| Tool | Component | Defect | Condition |
|---|---|---|---|
| laya (cloud) | 0.924 | 0.947 | 0.952 |
| laya_q4 (drone) | 0.870 | 0.944 | 0.938 |
| resnet18 (drone) | - | 0.981 | 0.979 |
| deit_small (drone) | - | 0.982 | 0.980 |
| convnext_recut (cloud) | - | 0.988 | 0.988 |
| At least one tool is correct | 0.970 | 1.000 | 1.000 |

### 3.3 Which tools the agent calls

| Method | Budget | laya | laya_q4 | Other tools |
|---|---|---|---|---|
| SFT and SFT + GRPO | 4 s | 94.7% | 5.3% | 0% |
| SFT and SFT + GRPO | 30 s | 94.3% | 5.7% | 0% |

The agent calls laya_q4 (on the drone) when the link is down.

### 3.4 Interpretation

1. **SFT teaches the format and one good tool.** The base model calls no tool and writes the same report every time. Its defect accuracy of 0.895 is only the share of good images. SFT teaches the JSON format and the step "call laya, then report".
2. **The agent copies laya, but it corrects some answers.** The component accuracy is equal to laya (0.92). Defect and condition are slightly better than laya (0.961 against 0.947, 0.958 against 0.952). The agent uses only the laya output, thus it learned to change some laya answers.
3. **The agent does not call the best tools.** convnext_recut alone gets 0.988 on defect and condition. The gold rule prefers laya, because laya gives three fields at the same time. Thus the training data rarely shows a second tool.
4. **The budgets are too large to have an effect.** The agent uses approximately 0.35 s. All budgets are 4 s or more.

## 4. GRPO without thinking and with thinking

| | Without thinking | With thinking |
|---|---|---|
| Steps, answers for each prompt | 400, 8 (2 prompts each step) | 200, 8 (1 prompt each step) |
| Answer length | approximately 22 tokens | approximately 150 tokens |
| Groups with no learning signal (first half, second half) | 0.91, 0.95 | 0.62, 0.71 |
| Mean reward (first half, second half) | 0.861, 0.903 | 0.845, 0.811 |
| KL to the SFT model | approximately 0.002 | approximately 0.06 |
| Largest change of a LoRA weight | 0.001 | not measured |

- Without thinking, the SFT model is almost certain (entropy approximately 0.002). The 8 answers are almost always the same. Thus the reward has no spread and GRPO does not learn. The GRPO adapter changed by a maximum of 0.001. This change does not change the most probable answer at any step. Thus the evaluation results are equal to SFT.
- With thinking, the 8 answers are different. GRPO gets more learning signal and the model moves more. But the training reward does not increase. The evaluation of `grpo_think` is necessary.

## 5. Faults found

| Fault | Effect | Status |
|---|---|---|
| With no time limit, a cloud tool was possible when the link was down. The code accepted an infinite cost when the budget was also infinite. | 347 of 30,950 gold tool steps (1.1%) select a tool that cannot run. The time without a limit is infinite in the evaluation. | Code corrected in commit a23f30f. The data and the models are not made again yet. Use only the 4, 10 and 30 s results. |
| The value report counts a tool answer that is not given as an error. | The defect detectors (for example `head_rust`) show an accuracy of approximately 0.01 to 0.05. | Not corrected. It has no effect on the agent. |

## 6. Thinking (teacher reasons from Haiku 5.5)

- Haiku 5.5 wrote one reason (80 to 150 words) for each of 4,000 balanced steps. 3,780 reasons passed the checks (median 102 words). The checks reject a reason with an incorrect length. They also reject a reason that does not name the selected tool. A reason that names a tool that is not a candidate, or refers to the true answer, is also rejected.
- Each training example is one step. The reasons of different steps of one image are not connected. At each step, the agent writes a new reason. The reasons of earlier steps are not in the next prompt.
- SFT with thinking: 0.6B (loss 1.74 to approximately 0.89), 1.7B (loss 1.90 to 0.81), 4B on the A10 (running).
- Evaluations of `base_think`, `sft_think` and `grpo_think`: running. The results go to `results/tools_v3/select/eval_think*.json`.

## 7. Files

| File | Content |
|---|---|
| `results/tools_v3/select/eval.json` | Evaluation without thinking |
| `results/tools_v3/value_report.json` | Accuracy of each tool alone |
| `results/tools_v3/select/*_train.json` | Training logs of SFT and GRPO |
| `results/tools_v3/select/sft_qwen_think.jsonl` | SFT data with teacher reasons |
| `scripts/runs/tools_v3_all.sh` | All stages |
