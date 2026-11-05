# Risk weights

Date: 2026-11-05. Status: the main setting for all experiments until we change it.

## Definition

The replay gives one risk value for each policy. The unit is risk points for each 1,000 frames.

risk per 1,000 frames = 1,000 × (10 × misses + 1 × false alarms + 0.5 × abstentions) / frames

| Error | Definition | Weight |
|---|---|---|
| Miss | The frame shows a defect. The answer is "good", or no answer arrives before the deadline. | 10 |
| False alarm | The frame shows a good part. The answer is a defect. | 1 |
| Abstention | The answer is "cannot judge". For a good part, a late answer is also an abstention. | 0.5 |

The code is in `src/fedorch/simulate.py` (`RISK` and `metrics`).

## Reasons for the weights

1. A miss has the highest weight. A missed defect stays on the power line until the next inspection. A defect such as corrosion or a missing cap can cause a failure of the line. Thus a miss has the highest cost.
2. A false alarm has the weight 1. A false alarm causes one unnecessary check by a person, for example a look at the evidence frame. The cost is work, not a failure.
3. The ratio 10 to 1 tells the controller to accept up to 10 false alarms to prevent one miss.
4. An abstention has the weight 0.5. "Cannot judge" asks a person to look at the frame. This costs less than a false alarm because the system does not claim a defect. An abstention costs more than 0 because the frame needs work.
5. A late answer on a defect frame is a miss. The answer does not arrive in time for the mission to use it.

## Limits

- The weights are assumptions. They do not come from a utility, a regulation or a cost study.
- A different ratio can change the ranking of the policies. A high miss weight helps policies that escalate more.
- Thus the paper must show the results for more than one set of weights. A sensitivity run must use at least the miss weights 5, 10 and 20.

## Where the weights act

- `scripts/step3_replay.py` and `scripts/step4_replay.py` score each policy with these weights.
- The tuned baselines and the learned controllers select their threshold or their λ on the validation towers by the lowest risk. Thus the weights also change the behaviour of these policies.
- The learned controller uses the same weights in its decision rule (`src/fedorch/controller.py`).
