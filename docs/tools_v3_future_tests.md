# tools_v3: tests for later

This document lists the tests that we do not do now. Each test can use the cached tool answers. Thus no test runs the image tools again.

The current test is as follows:

- Qwen3-0.6B selects tools step by step and writes a report with three fields (component, defect, condition).
- SFT and GRPO use one step at a time. GRPO does not run a full episode.
- The baseline is Qwen3-0.6B with no training and the same instructions.
- The metric is the accuracy of the report of Qwen against the true labels.

## Summary

| No. | Test | What changes | Effort | Order |
|---|---|---|---|---|
| 1 | Larger Qwen3 models (1.7B, 4B) | Model name only | Low | First |
| 2 | Thinking SFT with Haiku 5.5 reasons | SFT data | Medium | Second |
| 3 | GRPO on a full episode | GRPO loop and reward | High | Third |
| 4 | GRPO reward "any tool that is equally good" | Reward function | Low | Second |
| 5 | Other model families (Llama 3.2 1B and 3B, others) | Template, tokenizer, data | Medium | Fourth |
| 6 | More tool candidates (K=25) | Retriever setting | Low | Fourth |
| 7 | A plan stage before the tool steps | Prompt and data | Medium | Fifth |
| 8 | A selector that sees the image (VLM) | Model and prompt | High | Fifth |
| 9 | Clean-up: processor warning in `tv3_vlm.py` | One argument | Low | Any time |
| 10 | Remove laya from the tool list | Tool list and data | Low | Second |

## 1. Larger Qwen3 models

**Purpose:** Find out if a larger selector gives a better report. Zhou et al. (2026) show the gain from model size.

**Models:** `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`. We use Qwen3-4B, not a 3B model.

**What changes:** Nothing in the code. The script reads the model name from `TV3_SELECTOR`. Each model has its own outputs (`models/tools_v3/<tag>_sft`, `<tag>_grpo`) and its own logs (`logs/tools_v3/<tag>/`).

**Settings:** Models above 0.6B use fp16, SFT batch 2 with 8 accumulation steps, GRPO batch 8, and evaluation batch 12.

**Command (on the VM, after the 0.6B run):**

```
bash scripts/runs/tools_v3_selector.sh Qwen/Qwen3-1.7B
bash scripts/runs/tools_v3_selector.sh Qwen/Qwen3-4B
```

**Risk:** fp16 can give NaN values on the GV100 for Qwen. If this occurs, use LoRA with fp32 for the adapter, or a smaller learning rate.

## 2. Thinking SFT with Haiku 5.5 reasons

**Purpose:** Teach the model a short reason before each choice. Then find out if the reason makes the choices better.

**What changes:**

- Haiku 5.5 writes one reason for each SFT example. Each reason has 80 to 150 words. The reason explains why the gold tool (or the report) is the correct next step.
- Claude Code sub-agents write the reasons in batches. Thus there is no API cost.
- Use about 4,000 balanced examples (the same quantity of each gold tool and report, if possible).
- Do a check of each reason: the reason must agree with the gold action, and the reason must not use the true label.
- The completion becomes `<think>reason</think>` followed by the JSON. The prompt uses `enable_thinking=True`.
- Increase `max_completion_length` in GRPO and `max_new_tokens` in the evaluation (for example to 384).

**Compare:** SFT without reasons, SFT with reasons, and both after GRPO.

**Risk:** A 0.6B model can copy the style of the reason but not the logic. The time for each step also increases.

## 3. GRPO on a full episode

**Purpose:** Now GRPO gets a reward for each step. The reward compares the step with the gold tool. A full episode gives one reward at the end: the accuracy of the final report. Thus the model can find tool orders that the gold rule does not know.

**What changes:**

- Use the multi-turn rollout of TRL (`rollout_func` or `environment_factory`, TRL 1.17). The environment is `Episode` in `src/fedorch/select_env.py`.
- The model chooses a tool, the environment adds the cached answer of the tool, and the model chooses again until it writes the report.
- Reward = share of correct report fields. An option: subtract a small time cost (for example 0.01 for each second) for the open-ended budget.
- Mask the tool answers in the loss. Only the tokens of the model get a gradient.

**Risk:** The reward comes only at the end, so learning is slower. The earlier PPO test on mission segments showed this problem. Start from the SFT adapter.

## 4. GRPO reward "any tool that is equally good"

**Purpose:** Now only one gold tool gets reward 1. Frequently two or more tools add the same correct fields. The model then gets 0 for a choice that is equally good.

**What changes:** In `reward()` in `scripts/tv3_grpo_qwen.py`, give 1 to each tool that adds the same quantity of new correct fields as the gold tool. An option: give a small penalty for a slower tool. Add the flag `--reward any-best`. The GRPO data must contain the list of equally good tools for each step.

## 5. Other model families

**Purpose:** Find out if the method works on models that are not Qwen.

**Models:**

| Model | Size | Note |
|---|---|---|
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | Access approval on Hugging Face is necessary |
| `meta-llama/Llama-3.2-3B-Instruct` | 3B | Access approval on Hugging Face is necessary |
| `google/gemma-3-1b-it` | 1B | An option |
| `HuggingFaceTB/SmolLM3-3B` | 3B | An option, has a thinking mode |

**What changes:**

- The chat template is different. Llama 3.2 has no `enable_thinking` argument. `render()` in `src/fedorch/select_env.py` must use this argument only for Qwen3.
- The tokenizer is different. Thus render the SFT and GRPO data again from the episodes for each model. Do not use the Qwen text.
- Llama has no pad token. The code already uses the end token as the pad token.
- `default_dtype()` and the batch sizes must know the new model names.

## 6. More tool candidates (K=25)

**Purpose:** Now the retriever gives the 15 best tools and "report". With 25 tools, the gold tool is more frequently in the list, but the prompt is longer.

**What changes:** Set K=25 in `Retriever` and make the SFT and GRPO data again. Measure the recall of the gold tool in the candidate list for K=15 and K=25.

## 7. A plan stage

**Purpose:** Zhou et al. (2026) use three stages: plan, retrieval and selection. We use only retrieval and selection. In a plan stage, the model first writes a short plan (for example "first a classifier, then a defect tool"). The retriever then uses the plan as the query.

**What changes:** A new first step that writes the plan. The SFT data needs gold plans. These can come from the gold tool order.

## 8. A selector that sees the image

**Purpose:** Now Qwen sees only the text answers of the tools. A VLM selector (laya-vision or Qwen3-VL) can also see the image. Thus it can decide better which tool to use first.

**What changes:** Give the image to the selector in each step. The cost of each step increases. On the drone, only a small VLM is possible.

**Risk:** The earlier tests with laya as an agent showed that the tools made the result worse. Compare with care.

## 9. Clean-up: processor warning

`scripts/tv3_vlm.py` writes many "processor_kwargs" warnings. The warnings are not a problem. To stop them, give `processor_kwargs={"padding": True}`. Do this change only when no run is active on the VM.

## 10. Remove laya from the tool list

**Purpose:** Now only the laya tools (`laya`, `laya_q4`, `laya_recut`) give the component. Thus a laya tool adds three correct fields, and each other tool adds two or less. The gold rule selects the tool that adds the most new correct fields. Thus the gold is laya in almost all first steps. The select data of 2026-11-11 shows this: of 61,461 gold steps, 24,577 are `laya` and 30,511 are `report`. Each other tool is the gold in less than 3,000 steps. The agent then learns "use laya, then write the report". Without laya, the other tools must compete.

**What changes:**

- Remove `laya`, `laya_q4` and `laya_recut` from the tool list. An option: a list of excluded tools in an environment variable, so that the cache stays the same.
- Make the select data again (stage `select_data`, about 1 minute), then SFT, GRPO and the evaluation.

**Problem:** Without laya, no tool gives the component. Thus the component field of the report is only a guess. Do one of these:

- Give the component to the agent as an input. The detector that cuts the component from the photo also gives its class.
- Add a component output to the classifiers. This needs a new training of the classifiers.

**Compare:** The same agent with laya and without laya. Also compare the accuracy of the report and the time for each image.
