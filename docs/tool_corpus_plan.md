# Plan: a larger tool corpus and laya-vision as a tool selector (Plan, Retrieve, Select)

Date: 2026-11-11. Branch: `agentic-ppo-v2`. Source of the method: Zhou et al., "Training Small Language Models for Transferable Tool Selection" (arXiv 2610.31483).

## 1. Goal

With 4 similar tools, a threshold rule is as good as PPO (`docs/tool_selection_results.md`). This plan tests two changes:

1. A corpus of about 20 different tools, so that the choice of the tool has value.
2. laya-vision selects a tool from its text description, as in the paper. Fusion still gives the final answer. laya does not read the tool outputs.

Hypotheses:

- H1. The new tools add information. Fusion of all tools has a lower risk than the current 107 per 1,000 items.
- H2. laya, trained first by supervised fine-tuning (SFT) and then by GRPO, beats the tuned rules under the same time budget.
- H3. The trained selector can use a new tool from its description alone, without training.

## 2. The tool corpus

Each tool has four items: answers on all val and test items, the measured GPU time, the upload size and a short text description.

| Group | Tools | Source | Count |
|---|---|---|---|
| Component specialists | ResNet-50 for glass insulators, polymer insulators, hardware, yokes, rods and clamps | New training with `scripts/step2_train_classifier.py` on each group | 5 |
| Defect detectors | One-against-rest heads for rust, missing cap, corrosion and bird nest | New heads on ResNet-50 features | 4 |
| High-resolution re-cut | The component, cut again from the full InsPLAD-det photo with context, then ResNet and laya | New, if phase 0 succeeds | 2 |
| Test-time variants | ResNet on 5 crops and the full image | Present: `results/zoom/resnet_views.jsonl` | 1 |
| laya-vision | fp32 and 4-bit (a pair with a different cost), component question | Present: `results/vision/good300_full`, `q4_laya_good300_full` | 3 |
| Detection and quality | YOLO classification, image quality | Present in `src/fedorch/ppo_v2_tools.py` | 2 |
| SmolVLM | LoRA model on the drone | Present: `results/step2/tool_smolvlm_local.jsonl` | 1 |
| Cloud prompts | Qwen3-VL-8B with one question for each defect type | New cache run | 4 |
| Edge server | A larger model at the ground station, with a medium delay | Same answers as laya fp32, different cost | 1 |
| Operator | Defer | Present | 1 |

Total: about 24 tools.

## 3. Phases

### Phase 0: feasibility check (VM, 1 day)

1. Find the full photo for each crop. The crop name contains the DJI frame number. Match the crop to the photo with template matching on a smaller copy of the photo.
2. Test 200 crops. Success: 90% or more of the crops have a match with a clear score.
3. Count the training images for each component group and each defect type.

Gate G0: if the match fails, remove the re-cut tools and continue with the other tools.

### Phase 1: build the tool cache (VM, 3 to 4 GPU days)

1. Train the 5 component specialists and the 4 defect heads on the train towers. Use cross-fitting by tower, as in step 2.
2. Run all new tools on the val and test items (13,729 items). Record the answer, the time and the upload.
3. Run the 4 Qwen prompts on the val and test items. The estimate is about 2 hours on the GV100.
4. Write `results/tools_v3/tool_<name>.jsonl` and `results/tools_v3/tools.json` (descriptions and costs).

### Phase 2: value check (laptop, CPU, 1 day)

1. Fusion of all tools, with the EM threshold.
2. The oracle with the cheapest correct tool for each item.
3. A table of the tools that are correct when ResNet is wrong, for each component.

Gate G1 (H1): continue only if one of these conditions is true:

- Fusion of all tools has a risk below 102 (5 points better than 107).
- Cheap specialists are correct on a large part of the items where ResNet is wrong.

If not, the new tools add no information. Then the selection cannot have value, and the plan stops here. The result is still useful for the paper.

### Phase 3: gold tools and SFT of the laya selector (VM, 2 days)

1. Plan step: laya answers one choice question for each item. The question is "Which checks does this component need?" The options are the defect types of the component.
2. Select step: one laya choice question for each check. The options are the descriptions of the K = 8 best tools from a BM25 search over the tool descriptions. The state text contains the time left, the link state and the scores so far as short text.
3. Gold tool for each check: the cheapest tool that makes the fused answer correct within the budget. The cache gives this without a teacher model.
4. Cache the laya backbone outputs for each state, as in `scripts/agent_cache_h.py`. Limit the depth to 3 steps.
5. SFT of the laya decision head on the gold tools.

### Phase 4: GRPO (VM or laptop with the cache, 2 days)

1. For each image, run G = 8 attempts with different tool choices. The advantage of an attempt is its reward minus the mean reward of the group. Thus no critic is necessary.
2. Reward: minus the risk of the fused answer, minus λ times the time used divided by the budget.
3. A term that keeps the policy near the SFT model.
4. Compare these methods under the same budgets:

| Group | Methods |
|---|---|
| Classical | Uncertainty cascade with defer, paced cascade |
| Deep RL, numbers | PPO with an MLP on the new corpus |
| Deep RL, laya | laya zero-shot, laya after SFT, laya after SFT and GRPO |
| Federated | FedAvg of the laya scorer or head, 5 drones |

### Phase 5: transfer (laptop and VM, 2 days)

1. New tool: remove one tool from the training. Add it at test time with its description only.
2. New component group: train on 4 groups, test on the fifth.
3. New network: train in the weak regime, test in the poor regime.

### Phase 6: paper (from 2026-12-15)

Write the results into `paper_2col.tex`. The target is the submission on 2027-01-24.

## 4. Time plan

| Dates | Phase |
|---|---|
| 2026-11-11 to 11-12 | Phase 0 |
| 2026-11-13 to 11-19 | Phase 1 |
| 2026-11-20 | Phase 2 and gate G1 |
| 2026-11-21 to 11-30 | Phases 3 and 4 |
| 2026-12-01 to 12-07 | Phase 5 |
| 2026-12-08 to 12-14 | Extra seeds, figures |
| 2026-12-15 to 2027-01-24 | Paper |

## 5. Risks

| Risk | Effect | Action |
|---|---|---|
| The crops do not match the full photos | No re-cut tools | Continue without them (gate G0) |
| The specialists repeat the errors of ResNet | No new information | Stop at gate G1 and report the result |
| The backbone cache becomes too large | Long GPU time | Limit the depth and the candidates (K = 8) |
| laya does not learn the costs from the text | No gain against the rules | Put the costs as numbers into the state text |
| The GPU booking on student07 is not available | Delay | Run the smaller steps on the laptop |
| The tool descriptions leak the answer | The transfer test is not fair | Write the descriptions before the results |

## 6. Files

| New file | Content |
|---|---|
| `scripts/tools_v3_match.py` | Phase 0: match the crops to the full photos |
| `scripts/tools_v3_train.py` | Phase 1: specialists and defect heads |
| `scripts/tools_v3_cache.py` | Phase 1: run all tools, write the cache |
| `scripts/tools_v3_value.py` | Phase 2: fusion, oracle, complementarity |
| `src/fedorch/select_env.py` | Phases 3 and 4: Plan, Retrieve and Select environment |
| `scripts/select_sft.py`, `scripts/select_grpo.py` | Phases 3 and 4 |
| `scripts/runs/tools_v3.sh` | The VM commands |
