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

## 7. Run without laya-vision and with a component tool

### 7.1 The component tool

Without laya-vision, no tool gives the component. We counted the 6,092 test images where at least one tool without laya is correct. The share is 0.000 for the component, 1.000 for the defect and 0.999 for the condition. Thus we added a new tool:

| Tool | Location | Answer | GPU time | Component accuracy (6,092 test images) |
|---|---|---|---|---|
| `component_resnet18` | drone | component only | 2.8 ms | 0.996 |
| laya (for comparison) | cloud | all three fields | 73 ms | 0.924 |

The retriever puts `component_resnet18` in position 1 of 59 tools. This is true at the first step and after a condition result. Its description is "ResNet-18 component classifier: names the component type. It gives no defect or condition".

### 7.2 Training data

Settings: `TV3_EXCLUDE=laya,laya_q4,laya_recut`, `TV3_RUN=nolaya`. The data uses the corrected cost check (commit a23f30f).

| Gold action | Steps |
|---|---|
| report | 29,768 |
| `component_resnet18` | 29,102 |
| `resnet` (ResNet-50, drone) | 20,004 |
| `resnet18` (drone) | 3,562 |
| `dinov2_probe` | 1,009 |
| `clip_probe` | 482 |
| all other tools | less than 300 each |
| `convnext_recut` (most accurate tool) | 12 |

- An episode has approximately 3.8 steps (with laya: 2.7). The agent learns two tools: one condition classifier and the component tool.
- The gold rule selects the fastest tool that is correct on this image. ResNet-50 on the drone is correct on approximately 97% of the images, thus it is the gold tool on most images. Test 13 in `tools_v3_future_tests.md` describes this problem and the solutions.

### 7.3 Results

1,500 test images, one seed, greedy decoding. The same test images as section 3.

| Agent | Budget | Component | Defect | Condition | All three | Defect recall | Time (s) | Upload (kB) | Tool steps |
|---|---|---|---|---|---|---|---|---|---|
| With laya, SFT (section 3) | 4 s | 0.919 | 0.961 | 0.958 | 0.881 | 0.885 | 0.354 | 34.1 | 1.0 |
| Without laya, base | 4 s | 0.019 | 0.346 | 0.003 | 0.001 | 0.452 | 0.140 | 19.2 | 0.4 |
| Without laya, SFT | 4 s | 0.994 | 0.967 | 0.965 | 0.959 | 0.879 | 0.089 | 0.0 | 2.0 |
| Without laya, SFT | 10 s | 0.994 | 0.963 | 0.960 | 0.954 | 0.866 | 0.089 | 0.0 | 2.0 |
| Without laya, SFT | 30 s | 0.994 | 0.964 | 0.961 | 0.955 | 0.873 | 0.089 | 0.0 | 2.0 |
| Without laya, SFT | no limit | 0.994 | 0.965 | 0.963 | 0.957 | 0.866 | 0.089 | 0.0 | 2.0 |
| Without laya, SFT + GRPO | all | equal to SFT | | | | | | | |

Tools that the SFT agent calls (4 s): `component_resnet18` on 100% of the images, `resnet` on 94.7%, `resnet18` on 5.3%. All three tools are on the drone. Thus there is no upload.

GRPO training without laya, first half and second half: the mean reward is 0.961 and 0.969. The groups with no learning signal are 0.95 and 0.99. The KL to the SFT model is 0.0011 and 0.0003. The entropy is 0.0019 and 0.0005.

### 7.4 Interpretation

1. **The agent without laya is better and faster.** 95.9% of the reports are fully correct (with laya: 88.1%). The time is 0.089 s for each image (with laya: 0.354 s) and the upload is zero.
2. **Most of the gain is the component.** The component tool is correct on 0.994 of the images, laya on 0.919. Defect and condition increase by less than 1 point.
3. **The tool list has a larger effect than the training method.** The same SFT recipe gives 7.8 more points with a different tool list. GRPO gives no change with both tool lists.
4. **The agent is still not at the best possible result.** It calls ResNet-50 for defect and condition, not `convnext_recut` (0.988). The cause is the gold rule with hindsight (test 13).
5. **Defect recall does not increase** (0.879 against 0.885). ResNet-50 misses a similar share of defects as laya.
6. **The time without a limit is now correct.** The corrected cost check (commit a23f30f) gives 0.089 s.
7. **The untrained agent with all 62 tools** is also in `eval_nolaya.json` (method name `qwen3-0.6b base`, without `_nolaya`). It is a different case and not part of this comparison.

## 8. Thinking at three model sizes, and new tools without new training

Date: 2026-11-12. We use 500 test images (the first 500 of the 1,500 images in section 3). The budgets are 4, 10, 30 s and no limit. We use one seed, greedy decoding and up to 384 new tokens for each step. The tool list has 62 tools: the 61 tools of section 3 and `component_resnet18`. We trained the thinking agents before `component_resnet18` existed. Thus this tool was never a gold action in their training data.

### 8.1 All 62 tools

All three fields correct:

| Model | Method | 4 s | 10 s | 30 s | No limit | Tool steps (4 s) | Main tools (4 s) |
|---|---|---|---|---|---|---|---|
| 0.6B | SFT, thinking | 0.926 | 0.856 | 0.898 | 0.910 | 1.89 | component_resnet18 85%, laya_recut 75%, laya 22% |
| 0.6B | SFT + GRPO, thinking | 0.876 | 0.874 | 0.878 | 0.880 | 1.00 | laya 95%, laya_q4 5% |
| 0.6B | base, thinking | 0.002 | 0.002 | 0.000 | 0.000 | 0.73 | resnet 28% (53% reports not valid) |
| 1.7B | SFT, thinking | 0.876 | 0.856 | 0.840 | 0.834 | 1.25 | laya 50%, laya_recut 48% |
| 1.7B | SFT + GRPO, thinking | 0.876 | 0.874 | 0.876 | 0.874 | 1.00 | laya 95%, laya_q4 5% |
| 1.7B | base, thinking | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | none (all reports not valid) |
| 4B | SFT, thinking | 0.842 | 0.850 | 0.856 | 0.866 | 1.15 | laya_recut 72%, laya 21% |
| 4B | SFT + GRPO, thinking | 0.874 | 0.880 | 0.880 | 0.872 | 1.01 | laya 95%, laya_q4 5% |
| 4B | base, thinking | 0.000 | 0.000 | 0.000 | 0.000 | 0.00 | none (all reports not valid) |

For comparison (section 3, 1,500 images): Qwen3-0.6B SFT without thinking gets 0.881 at 4 s.

### 8.2 Laya removed at test time

We trained the agents of section 3 and section 8.1 with laya. For this test, `TV3_EXCLUDE=laya,laya_q4,laya_recut` removes the laya tools only in the evaluation. 59 tools remain.

All three fields correct:

| Model | Method | 4 s | 10 s | 30 s | No limit | component_resnet18 called (4 s) | Component (4 s) | Condition (4 s) |
|---|---|---|---|---|---|---|---|---|
| 0.6B | SFT | 0.334 | 0.346 | 0.330 | 0.348 | 0% | 0.348 | 0.898 |
| 0.6B | SFT + GRPO | 0.754 | 0.728 | 0.756 | 0.586 | 72% | 0.774 | 0.956 |
| 0.6B | SFT, thinking | 0.340 | 0.244 | 0.408 | 0.602 | 73% | 0.718 | 0.482 |
| 0.6B | SFT + GRPO, thinking | 0.492 | 0.424 | 0.158 | 0.068 | 64% | 0.634 | 0.714 |
| 1.7B | SFT, thinking | 0.762 | 0.662 | 0.718 | 0.596 | 84% | 0.830 | 0.846 |
| 1.7B | SFT + GRPO, thinking | 0.924 | 0.910 | running | running | 100% | 0.988 | 0.942 |
| 4B | SFT, thinking | 0.394 | 0.498 | running | running | 60% | 0.600 | 0.736 |
| 4B | SFT + GRPO, thinking | running | | | | | | |

Reference: the agent trained without laya (section 7, 1,500 images) gets 0.959 at 4 s.

### 8.3 Interpretation

1. **With all tools, thinking adds little.** The best thinking agent is 0.6B SFT (0.926 at 4 s). It is 4.5 points better than SFT without thinking at 4 s. Its result changes from 0.856 to 0.926 between budgets that do not limit it. Thus the budget text in the prompt changes its plan.
2. **A larger model does not help.** With thinking SFT, 0.6B is equal to or better than 1.7B and 4B.
3. **GRPO with thinking gives the same policy at all sizes.** The 0.6B, 1.7B and 4B agents call laya on 95% of the images. They call one tool and get approximately 0.875. This is the policy of SFT without thinking.
4. **The untrained thinking models do not give a report.** The 1.7B and 4B base models give no valid report on any image. The thinking is longer than 384 tokens.
5. **GRPO makes the agent better when the tool list changes.** With laya removed, the GRPO agents are much better than their SFT agents. The 0.6B agent without thinking gets 0.754 against 0.334. The 1.7B agent with thinking gets 0.924 against 0.762. With the full tool list, GRPO did not change the result (section 4).
6. **The 1.7B GRPO agent with thinking uses a new tool without new training.** It calls `component_resnet18` on 100% of the images, then ResNet-50 on 81%. It gets 0.924 at 4 s. The agent trained without laya gets 0.959.
7. **The SFT agents trust the wrong tool for the condition.** They find the component tool. But the 0.6B SFT agent with thinking gets only 0.48 to 0.66 for the condition.
8. **The 0.6B agents are not stable across budgets without laya.** For example, the 0.6B GRPO agent with thinking gets 0.07 to 0.49.

Limits: one seed, 500 images (an uncertainty of approximately 3 points), results marked "running" are not available yet.

## 9. Files

| File | Content |
|---|---|
| `results/tools_v3/select/eval.json` | Evaluation without thinking |
| `results/tools_v3/value_report.json` | Accuracy of each tool alone |
| `results/tools_v3/select/*_train.json` | Training logs of SFT and GRPO |
| `results/tools_v3/select/sft_qwen_think.jsonl` | SFT data with teacher reasons |
| `scripts/runs/tools_v3_all.sh` | All stages |
