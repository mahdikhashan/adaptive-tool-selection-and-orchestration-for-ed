# Plans: conditions where a decision model can beat the MLP

Date: 2026-11-05. This document builds on `docs/controller_evidence_options.md` and `docs/step4_results.md`.

## 1. The starting point

- In step 4, the MLP risk rule gives the lowest risk (136.4). nano-jev and Laya give 171 to 180.
- Step 4b changes the target and the evidence for all controllers. A pipeline test with random extra evidence gave the MLP choice controller a risk of 113.4. Thus the choice target alone can help much. This value is not a result. The real evidence comes from student07.
- The fairness rule stays: every controller gets the same facts. With the same structured facts, a small numeric model is often as good as a text model. A study with task labels found that a fine-tuned DeBERTa beat the decision models (arXiv 2610.00346).
- Thus a decision model can win only in one of these conditions:
  1. The input has information that a numeric model cannot use well (text, images, new components).
  2. The training data is small, and pretraining helps.
  3. The decision model contains the MLP features and adds to them.

## 2. Plans

### Plan A: low-data regime (data efficiency)

- **Claim:** a pretrained decision model needs fewer labelled frames than the MLP.
- **Method:** train each controller on 1%, 5%, 20% and 100% of the train towers. Use 3 seeds for each size. Plot the risk against the number of labelled frames.
- **Why it fits the paper:** in the federated setting, one drone has few labelled frames. Data efficiency is the reason to federate. If the decision model wins at small sizes, it is the better local learner for each drone.
- **Cost:** low. The MLP runs on the laptop. nano-jev needs about 1 to 3 minutes for each run on the GV100. Laya needs more time, so use only 2 sizes for Laya.
- **Chance to win:** high at 1% and 5%. Low at 100%.

### Plan B: unseen components (leave-one-component-out)

- **Claim:** a text model transfers to a component that it did not see in training. The MLP uses a one-hot vector for the component, so it cannot transfer.
- **Method:** for each of the 5 largest components, remove its frames from training. Test on that component only. Add the tool cards (option E4) as text. The MLP gets the same tool statistics as numbers.
- **Cost:** low on the laptop for the MLP. 5 runs of nano-jev on the GV100.
- **Chance to win:** medium. The component name and the tool card are text that the decision model can use. The MLP gets only the numbers.

### Plan C: retrieval of similar frames (option E3)

- **Claim:** a summary of similar past frames gives per-frame evidence that a text model reads directly.
- **Method:** save the classifier embeddings on student07. For each frame, find the 20 nearest frames in the memory of the same drone. Give the counts of correct tools as text and as numbers.
- **Cost:** medium. It needs one embedding pass and a nearest-neighbour search.
- **Chance to win:** low against an MLP that gets the same counts. High as a gain for all controllers. Use it to make the federated controller stronger, not to make the decision model win.

### Plan D: the decision model sees the image (option V2)

- **Claim:** the image has information that the classifier output does not have.
- **Method:** fine-tune `perplexity-ai/pplx-decider-flash-v1.2` (4.5B) with LoRA on the choice target. Give it the image and the evidence text. For a fair test, give the MLP the classifier embedding (2,048 values).
- **Cost:** high. It needs about 9 GB of weights and fp16 training on the GV100.
- **Limit:** this model cannot run on the drone. It is a ground-station controller or an upper bound.
- **Chance to win:** medium to high. The comparison is fair only with the image embedding in the MLP.

### Plan E: fusion head (design §2.1)

- **Claim:** a decision model with a numeric head is the deployable controller. It is at least as good as the MLP and it adds the text evidence.
- **Method:** concatenate the pooled nano-jev output with the MLP features. Train one head on the choice target.
- **Cost:** low.
- **Chance to win:** high, because the model contains the MLP input. The claim is weaker: the result shows that text adds information, not that text replaces numbers.

### Plan F: operator intents in free text

- **Claim:** an operator can give a new intent in words, for example "only report corrosion on clamps". A text model reads the intent. The MLP cannot.
- **Method:** write 10 to 20 intent templates that change the risk weights for some labels. Change the targets to match each intent. Test on intents that are not in training.
- **Limit:** the intents are synthetic. The paper must say this.
- **Chance to win:** high, because the MLP has no input for the intent. This is the clearest agentic use of a decision model.

### Plan G: training changes

- **Method:** for Laya, compare the `rlcd` loss with the `soft-ce` loss. For all models, test the soft-target temperatures 0.5, 1 and 2, and use an ensemble of 3 seeds.
- **Cost:** low for nano-jev, medium for Laya.
- **Chance to win:** low. Use these changes only after Plan A or Plan F show a gap.

## Status (2026-11-05)

- Step 4b: complete. The MLP is the best controller (114.2). nano-jev has 125.3 and Laya has 150.0.
- Plan A: complete. nano-jev is equal to the MLP at 1% and 5% and worse at 20% and 100%. See `docs/planA_results.md`.
- Plans B to G: not started.

## 3. Recommended order

1. Step 4b with the real evidence. This shows whether the choice target closes the gap for the decision models.
2. Plan A (low data), because it connects directly to federated learning.
3. Plan F (free-text intents), because it gives the clearest reason for a decision model.
4. Plan B (unseen components).
5. Plan E (fusion head) as the deployable controller for the federated step.
6. Plan D only if the GPU time is available.

## 4. Honest outcome

If the decision models lose in step 4b and in Plan A, the paper reports a negative result. This result agrees with the literature. The federated study then uses the MLP or the fusion head as the controller.
