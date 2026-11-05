# Plan F results: free-text operator intents

Date: 2026-11-05. Status: complete. We did not train nano-jev for plan F. Finding 6 gives the reason.
Source files: `results/planF/metrics_label.json`, `metrics_component.json`, `metrics_component_budget0.05.json`, `value_check_label.json`, `value_check_component.json`, `laya_zero_shot_parse.json`.
Scripts: `scripts/planF_train_mlp.py`, `scripts/planF_replay.py`, `scripts/planF_value_check.py`, `src/fedorch/intents.py`.

## Question

The operator writes an intent in free text. The intent tells which defects are most important. Can a controller that reads the text make better decisions than a controller that does not read it?

## Method

1. **Intents.** An intent has a focus set and a strict flag. A miss on a focus frame costs 20. A miss on another defect frame costs 2. A false alarm costs 1, or 3 if the intent is strict. An abstention costs 0.5.
2. **Two kinds of focus.** In the label kind, the focus set contains 1 or 2 defect types. In the component kind, the focus set contains 1 or 2 of the 6 components that have defects in the test towers.
3. **Text.** Each intent has a text from a template. The training intents use 4 templates. The test intents use 3 other templates. Thus the test wording is new. The intents are synthetic.
4. **Test groups.** `new_wording`: a focus set from training, with a new template. `new_focus`: a focus set that is not in training, with a new template.
5. **Controllers.** Each MLP learns the choice target under the risk weights of the intent. The MLPs differ only in the intent input:
   - `mlp_blind`: no intent input.
   - `mlp_bow`: a bag of words of the intent text (64 values).
   - `mlp_parsed`: the true focus set and the strict flag as numbers. This is the result of a perfect parser.
6. **Value check.** A parse-and-weight rule takes the score of `mlp_blind` and multiplies it by 10 for a focus frame. 10 is the ratio of the miss costs (20 / 2). This rule uses the true focus set. Thus it shows how much a perfect reader of the intent can gain.
7. **Budget.** Some runs limit the escalations to a share of the frames of a mission (5% or 2%).
8. **Replay.** Weak regime, 5 network seeds, 7,057 test-tower frames. Each value is the mean over the test intents of the group.

## Results

Risk per 1,000 frames, under the weights of each intent:

| Setting | Controller | New wording | New focus |
|---|---|---|---|
| Label, no budget | Static cascade | 133.8 | 111.0 |
| | `mlp_blind` | 120.8 | 106.9 |
| | `mlp_bow` | 165.8 | 140.9 |
| | `mlp_parsed` | 126.9 | 105.0 |
| | Parse-and-weight rule | 121.3 | 103.9 |
| Label, budget 2% | `mlp_blind` | 136.3 | 112.9 |
| | Parse-and-weight rule | 125.2 | 106.8 |
| Component, no budget | Static cascade | 98.5 | 132.9 |
| | `mlp_blind` | 82.6 | 125.0 |
| | `mlp_bow` | 176.8 | 209.6 |
| | `mlp_parsed` | 83.2 | 127.0 |
| | Parse-and-weight rule | 79.9 | 121.9 |
| Component, budget 2% | `mlp_blind` | 89.5 | 139.4 |
| | Parse-and-weight rule | 94.1 | 140.1 |

The oracle has a risk of 32 to 60 in all settings.

## Findings

1. Without a budget, the intent has almost no value for the decision. A perfect reader (parse-and-weight rule) changes the risk of `mlp_blind` by −3.1 to +0.5 risk points.
2. With a tight budget (2%) and the label kind, the intent has value. The parse-and-weight rule decreases the risk by 8% (new wording) and 5% (new focus). With the component kind, the same rule increases the risk.
3. The MLP with the true intent (`mlp_parsed`) does not use the intent well. It is not better than `mlp_blind`, and it is worse on new wording in the label kind (126.9 against 120.8). The explicit rule uses the same facts better. This agrees with arXiv 2610.18579: models read facts correctly but do not combine them well, and code can combine them.
4. The bag-of-words MLP fails on new wording and on new focus sets. Its risk is higher than the classifier alone. A learned encoder of the intent text does not transfer to new templates.
5. Zero-shot Laya can parse the strict flag: 56 of 56 intents are correct. It does not parse the focus set well. The exact focus set is correct for 2 of 28 label intents and 11 of 28 component intents.
6. We did not train nano-jev for plan F. The value check shows that a perfect reader of the intent gains at most 11 risk points over `mlp_blind`. nano-jev is already about 11 risk points worse than the MLP without intents (step 4b). Its seed variation is about ±10 risk points. Thus nano-jev cannot beat the MLP in this test.

## Limits

- The intents are synthetic templates. The focus names appear literally in the text. Thus a keyword match parses them perfectly. A test of language understanding needs real operator intents with paraphrases.
- The value of an intent depends on the tools. All tools here can remove errors of all defect types. If the tools differ by defect type or by component, the intent can have more value.

## Consequences

- Plan F does not show an advantage for the decision models.
- The useful design is a parser plus a rule. The parser changes the operator intent into a small set of facts. Code then combines these facts with the controller score. This is a design for an agentic controller, and the paper can show it as future work.
- The federated step uses the choice MLP.
