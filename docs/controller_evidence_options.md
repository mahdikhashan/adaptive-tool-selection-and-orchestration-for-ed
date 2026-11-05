# Richer evidence for the decision models

## 1. Problem

In step 4, every controller estimates q(item, tool). This value is the probability that a tool gives the correct answer for a frame. The risk rule in `src/fedorch/controller.py` then picks the tool.

The decision models (nano-jev and Laya) read one evidence sentence. The function `evidence` in `src/fedorch/features.py` builds it. It contains:

- the task type (yes/no defect question or defect type question),
- the component,
- the first and second classifier answers, with probabilities rounded to 2 decimals.

The MLP reads the same facts as numbers, plus entropy, margin and the number of options. So the decision models get no more information than the MLP. A text model can only win if the text carries information that the MLP does not have, or information that a text model uses better.

This document lists options to give the controller more information at inference time. Each option uses only data that the drone or the mission has before the controller decides.

## 2. A correction to the first diagnosis

The first diagnosis named two losses: the rounded probabilities and the missing third answer. A check of the step 2 cache shows that the second loss is small.

- In `results/step2/tool_classifier.jsonl`, 34,841 items have 3 options and 4,387 items have 4 options.
- The third classifier answer has a probability above 0.01 on only 547 of 39,228 items (1.4%). For yes/no questions, the third option is "cannot judge", and the classifier always gives it 0.

The code shows a larger loss. The function `option_probabilities` in `src/fedorch/classifier.py` maps the 6-class output onto the options of the question. For a yes/no question, it keeps only P(good) and 1 − P(good). Thus the controller cannot see which defect type the classifier suspects. Both the MLP and the decision models lose this information.

## 3. Rules for a fair comparison

1. **Causal information only.** The controller can use a fact only if the drone knows it before the decision. Ground truth labels and the outputs of tools that did not run are not permitted.
2. **Same information for every controller.** When the text gets a new fact, the MLP gets the same fact as a number. If not, a gain can come from the input and not from the model.
3. **The rule keeps the cost terms.** The bandwidth, the backlog and the deadline stay in the risk rule. The q model predicts correctness only. Section 7 gives the one exception.
4. **Short text.** The evidence text must stay below approximately 200 tokens, so that nano-jev (MiniLM) does not cut it.

## 4. Options from the edge device, with no new tool calls

These options use the image and the classifier run that the drone already has.

### E1. The full classifier output

- **What:** all 6 class probabilities, before the mapping to the options. Also the exact margin, the entropy and the log-odds of "good".
- **Text form:** "Classifier class probabilities: good 0.981, rust 0.012, corrosion 0.004, …". Use 3 decimals, or a log-odds value, because most frames have a confidence above 0.99. Rounding to 2 decimals makes many of these frames identical.
- **Why it can help:** a yes/no frame with "rust 0.4" and a frame with "nest 0.4" can need different tools. For example, the cloud model almost never sees corrosion (recall 0.012 in step 2).
- **Cost:** one new classifier pass that saves the raw class probabilities. This is approximately 5 minutes on the GV100.

### E2. Image quality and scale

- **What:** the pixel size of the image before the resize to 384 px, and the blur (variance of the Laplacian). Also the brightness, the contrast and the JPEG size in bytes.
- **Text form:** "Image: 212 × 180 px, sharp, dark, 31 KB."
- **Why it can help:** small or blurred crops can be the frames where the classifier fails and a VLM at 512 px also fails. The JPEG size is also the uplink payload, so the rule already uses a part of this.
- **Cost:** CPU only, a few milliseconds per frame. It needs the images on student07.

### E3. Nearest training frames

- **What:** the classifier penultimate embedding (2,048 values). The drone keeps a memory of embeddings from its own past frames with known tool outcomes. For a new frame, it finds the 20 nearest frames.
- **Text form:** "Among 20 similar past frames: classifier correct 17, onboard VLM correct 19, cloud correct 11."
- **Why it can help:** this gives local, per-frame evidence that a text model can read directly. The MLP gets the same 3 counts as numbers.
- **Federated fit:** the memory stays on the drone. Only the controller weights leave the drone. This agrees with the privacy argument of the paper.
- **Cost:** one embedding per frame and one search in a small memory. The memory must contain only training-tower frames, or past frames with labels from the same client.

### E4. Tool cards

- **What:** a short description of each tool. It gives the accuracy of the tool per task type and component on the training towers.
- **Text form:** "Cloud model: weak on corrosion (recall 0.01), strong on yes/no defect questions."
- **Why it can help:** a text model can read a tool card for a tool or component that it did not see in training. The MLP cannot do this. This option is the clearest test of the claim that a decision model adds value.
- **Cost:** none at run time. In the federated setting, the clients can share the tool cards as aggregate counts.
- **Risk:** for known tools and components, the MLP already learns the same statistics from the one-hot inputs. Thus the gain can be zero in the main replay. Test it in a leave-one-component-out split.

## 5. Options from the edge device, with extra compute

### E5. Test-time augmentation agreement

- **What:** run the classifier on a flipped copy and on 2 crops. Record whether the 3 answers agree, and the spread of the confidence.
- **Cost:** approximately 3 × 7 ms on the GV100. With the 9× Jetson factor, this is approximately 190 ms per frame. Thus it uses approximately 16% of the 1.16 s frame interval.

### E6. A second decision after the onboard VLM

- **What:** now the controller decides once. After an escalation it stops (`len(history) > 1` in `risk_rule`). In this option, the controller decides a second time after SmolVLM answers. The second decision reads the SmolVLM answer, its confidence, and whether it agrees with the classifier.
- **Why it can help:** agreement between two different tools is a strong signal of a correct answer. This option is the most likely to close a part of the gap to the oracle.
- **Cost:** it changes the action space and the latency. Report it as a separate controller, not as an evidence change.

## 6. Options from the mission

### M1. Other questions about the same image

- **What:** ElecVQA asks more than one question about one image. The items have a `parent_id`, and the replay orders the items by tower and `parent_id`. The controller can read the classifier answers and the tool results for earlier questions on the same image.
- **Text form:** "The previous question on this image: the classifier said defect, the onboard VLM said rust."
- **Causal check:** use only questions that the replay processed before the current one.
- **Open point:** the number of images with more than one question in the test split. Count this on student07, because the data is not on the laptop.

### M2. The history of the current flight

- **What:** running counts for the current tower. Examples are the escalation rate, the rate of "defect" answers, and how often the escalated tool disagrees with the classifier.
- **Text form:** "On this tower so far: 12 escalations, the onboard VLM disagreed with the classifier 5 times."
- **Why it can help:** a new tower can look different from the training towers. A high disagreement rate shows this shift during the flight.
- **Limit:** the drone does not know the true labels during a flight. So it can count disagreements but not errors.

### M3. Mission metadata

- **What:** the weather, the time of day and the distance from the camera to the component. Also the age of the tower and the result of the last inspection.
- **Limit:** InsPLAD and ElecVQA do not record these values. Synthetic values would test only the simulator, not the controller.
- **Recommendation:** do not use M3 in the main results. Name it as future work, or as a requirement for a real flight data set.

## 7. Two larger variants

### V1. The decision model also sees the network

The decision model reads the bandwidth, the link state, the backlog and the deadline slack, and it picks the tool directly. This removes the risk rule. It tests whether a decision model can do the full trade-off. The design papers report that decision models read facts correctly but do not combine them well (arXiv 2610.18579). So keep V1 as an ablation against the rule, not as the main controller.

### V2. The decision model sees the image

A multimodal decision model reads the frame directly. The candidate in the design doc is `perplexity-ai/pplx-decider-flash-v1.2` (4.5B). It cannot run onboard, so it can only be a ground-station controller. It is an upper bound for "more information", not a deployable option.

## 8. Summary

| Option | Source | New compute | Needs student07 | Expected value |
|---|---|---|---|---|
| E1 full classifier output | edge | classifier pass, 5 min | yes | high |
| E2 image quality | edge | CPU | yes (images) | medium |
| E3 nearest frames | edge | embedding + search | yes | high |
| E4 tool cards | training statistics | none | no | low in main replay, high for new tools |
| E5 augmentation agreement | edge | +190 ms per frame | yes | medium |
| E6 second decision | edge | SmolVLM latency | no (cache has it) | high |
| M1 same image | mission | none | no (count on student07) | unknown until counted |
| M2 flight history | mission | none | no | medium |
| M3 metadata | mission | not available | — | not testable |
| V1 network in the model | mission | none | no | ablation |
| V2 image in the model | edge | 4.5B model | yes | upper bound |

## 9. Recommended plan

1. **Test the value of each fact with the MLP first.** Add each option to the MLP features, one group at a time. Measure the AUROC of q for each tool and the replay risk. An option that does not help the MLP probably does not help a text model. This step is fast on the laptop for E4, E6, M1 and M2.
2. **Run one extraction pass on student07.** One script saves E1 (raw class probabilities), E2 (image quality) and E3 (embeddings) for all items.
3. **Give the useful groups to nano-jev and Laya.** Use the same groups as the MLP, as short text.
4. **Report E6 separately,** because it changes the action space.
5. **Test E4 on a leave-one-component-out split.** This split shows whether text models generalize to new components better than the MLP.

The main question for the paper is this: with the same information, does a decision model estimate q better than the MLP? A second question is whether text inputs such as tool cards and retrieval summaries let it generalize where the MLP cannot.
