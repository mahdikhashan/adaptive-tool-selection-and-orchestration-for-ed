You write short teacher reasons for training data. Project: a small language model (Qwen3-0.6B) acts as a drone inspection agent. For one image of a power-line component, it picks tools step by step (image classifiers, VLMs, defect detectors; some run on the drone, some in the cloud, which needs an image upload over a weak LTE link). It then finishes with a three-part report: component type, defect yes/no, condition. The agent never sees the image, only the tool results. We want the model to "think" before each action, so we need a reason for each training step.

INPUT: read the file `results/tools_v3/select/think/batch_NNN.json` (relative to your working directory) with the Read tool. It has:
- "tools": tool id -> "location: description" (glossary).
- "items": a list of steps. Each has "i" (id), "state" (time budget, time left, cloud link, whether the image is uploaded, and "Tool results so far"), "candidates" (allowed actions, each as "id (location, time cost of that tool now)", plus "report"), and "answer" (the action the agent takes in this step: {"name":"SELECT","ids":[tool]} or {"name":"REPORT", component, defect, condition}).

TASK: for every item, write one reason (80 to 150 words) that the agent could think just before it takes "answer". Rules:
1. Write in the first person as the agent, present tense, plain English. Forward-looking: reason only from the state, the candidate list and the glossary. Never say or hint that you know the answer, a label, a "gold" or "correct" choice, or training.
2. Never invent tool results. Use only the results listed in the state. If there are none yet, say so.
3. Cover: which of the three fields (component, defect, condition) are still unknown or uncertain given the results so far; any disagreement between tools; time left vs. the time cost of options; drone vs. cloud, the link state and whether the image is already uploaded (a cloud tool pays the upload only once).
4. For SELECT: explain why this specific tool (write its id exactly as in the candidates, for example laya_q4 or clip_probe, at least once; do not only paraphrase it) is the best next step compared with one or two named alternatives. Do not name a tool that is not in the candidates.
5. For REPORT: explain why to stop now and how each field follows from the listed results (which tool you trust and why when they disagree). The reason must end consistent with the report values in "answer", but do not write JSON.
6. Vary the wording across items; do not reuse one template, also for the many first steps that pick the same tool.
7. If the answer selects a tool that is not in the candidates, write the line with the reason "SKIP".

OUTPUT: use the Write tool to create two files (JSON Lines, one object per line, no other text):
- `results/tools_v3/select/think/out_NNN_a.jsonl` for the first 50 items,
- `results/tools_v3/select/think/out_NNN_b.jsonl` for the remaining items.
Each line: {"i": <the item's i>, "reason": "<reason>"}. Escape quotes properly so each line is valid JSON. Keep the item order.

LIMITS: Use only the Read and Write tools. Do not run shell commands, scripts, git, or any compute. Do not edit any other file. Do not delegate.

DONE when both files exist with one line per item. Reply with one line: "batch NNN: <number of lines written>".
