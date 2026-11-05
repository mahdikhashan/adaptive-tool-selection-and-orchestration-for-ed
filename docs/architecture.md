# Pipeline architecture

Date: 2026-11-05. This document shows the path of one frame from the camera to the answer, on the drone or in the cloud. The interactive diagram is `architecture/figures/pipeline.html` in the artifacts directory.

Each stage has a status:

- **Implemented**: code exists and runs in the experiments.
- **Simulated**: the replay models the stage with measured or published values.
- **Planned**: the stage is not in the code.

## 1. Overview

```
                      DRONE (Jetson, onboard)                                       CLOUD
 ┌────────┐   ┌───────────┐   ┌──────────────┐   ┌────────────┐
 │ Camera │──▶│ Key-frame │──▶│ Crop +       │──▶│ Classifier │──┐
 │ video  │   │ filter    │   │ question     │   │ ResNet-50  │  │ class probabilities
 └────────┘   └───────────┘   └──────┬───────┘   └────────────┘  ▼
                                     │ image signals       ┌──────────┐
 ┌─────────────┐                     └────────────────────▶│ Evidence │
 │ Telemetry:  │                                           └────┬─────┘
 │ bandwidth,  │──────────────────────────┐                     │
 │ backlog,    │                          ▼                     ▼
 │ deadline    │                    ┌──────────────────────────────────┐
 └─────────────┘   operator intent  │ Controller: choice MLP           │
                   (optional) ─ ─ ─▶│ + deadline and link checks       │
                                    └──┬──────────────┬─────────────┬──┘
                                ~89%   │       ~7.5%  │         ~3% │
                                       ▼              ▼             ▼
                              ┌────────────┐ ┌────────────────┐ ┌────────┐  JPEG ~35 KB  ┌────────────────┐
                              │ Accept the │ │ Onboard VLM    │ │ 4G/LTE │──────────────▶│ Cloud VLM      │
                              │ classifier │ │ SmolVLM-256M   │ │ uplink │               │ Qwen3-VL-8B    │
                              └─────┬──────┘ └───────┬────────┘ └────────┘               └───────┬────────┘
                                    │                │                                           │ answer
                                    ▼                ▼                                           ▼
                              ┌─────────────────────────────────────────────────────────────────────────┐
                              │ Answer for the frame (label or "cannot judge"; late = miss)             │
                              └───────┬───────────────────────────────┬─────────────────────────────────┘
                                      ▼                               ▼
                            ┌───────────────────┐          ┌──────────────┐  local update  ┌────────────┐
                            │ Inspection report │          │ Decision log │─ ─ ─ ─ ─ ─ ─ ─▶│ Federated  │
                            │ (ground station)  │          └──────────────┘                │ aggregation│
                            └───────────────────┘                  ▲   new weights          └─────┬──────┘
                                                                   └ ─ ─ ─ ─ to the controller ─ ─┘
```

The shares (89%, 7.5%, 3%) are from the choice MLP in the weak regime (step 4b). It escalates 10.8% of the frames. Its uplink is 1.18 MB for each 1,000 frames, which is about 33 cloud calls.

## 2. Stages

| Stage | Place | Input | Output | Time | Status |
|---|---|---|---|---|---|
| Camera | drone | — | video | — | not in the replay |
| Key-frame filter | drone | video | about 1 frame every 1.16 s | about 2 ms for each frame (EVLM) | planned, simulated as the frame rate |
| Crop and question | drone | key frame | crop of one component, task, answer options | — | implemented from ElecVQA-Bench |
| Classifier (ResNet-50, 384 px) | drone | crop | 6 class probabilities | 7.3 ms (GV100), about 65 ms (Jetson) | implemented |
| Evidence | drone | crop, classifier output | features: class probabilities, log-odds, task, component, image size, blur, brightness, contrast | less than 1 ms | implemented |
| Telemetry | drone | network and queue | bandwidth of the last second, backlog, deadline 4 s | — | simulated from the LTE traces |
| Controller (choice MLP) | drone | evidence, telemetry | accept, onboard VLM or cloud | less than 1 ms | implemented in the replay |
| Onboard VLM (SmolVLM-256M with LoRA, 512 px) | drone | crop, prompt | answer | 80.8 ms (GV100), about 0.73 s (Jetson) | implemented |
| Uplink (4G/LTE) | drone to cloud | JPEG at 512 px | about 35 KB | payload / bandwidth + 0.05 s, failure after 3 s | simulated |
| Cloud VLM (Qwen3-VL-8B, zero-shot, fp16) | cloud | image, prompt | answer | 127 ms (GV100 server) | implemented |
| Answer | drone | tool answers | label or "cannot judge" | — | implemented |
| Inspection report | ground station | answers, evidence frames | report | — | planned |
| Decision log | drone | evidence, action, outcome | training data for the controller | — | implemented as the tool cache |
| Federated aggregation | server | controller updates from each drone | new controller weights | — | planned (next step) |

The Jetson times are the GV100 times multiplied by 9. The factor is an estimate from the hardware ratio. No Jetson measurement exists.

## 3. The decision for one frame

1. The classifier runs on each frame.
2. The controller calculates the probabilities of the 3 actions from the evidence.
3. The controller removes an action that is not possible:
   - the onboard VLM, if the backlog plus its time is more than the deadline.
   - the cloud, if the link is down or the transfer takes more than 3 s.
4. The controller escalates if p(tool) − p(accept) is at least the threshold. The threshold comes from the validation towers. If not, it accepts the classifier answer.
5. The answer of the last tool that succeeded is the answer for the frame.
6. If the frame finishes more than 4 s after its arrival, the answer is a miss.

## 4. The two flows

- **Inference flow** (each frame): camera, key frame, crop, classifier, evidence, controller, one tool, answer.
- **Learning flow** (between missions): the decision log trains the controller. In the federated step, each drone trains its controller on its own log. A server averages the controller weights and sends the new weights to all drones. The images stay on the drone. Only the controller weights leave the drone.

## 5. Options that the experiments tested but do not use

| Option | Result | Document |
|---|---|---|
| Split VLM (INT8 visual tokens, 36,868 B) | Same accuracy as the onboard VLM. Almost the same payload as the JPEG at 512 px. | `docs/step2_results.md` |
| Decision models as the controller (nano-jev, Laya) | Worse than the choice MLP (125.3 and 150.0 against 114.2) | `docs/step4b_results.md` |
| Operator intent as text | Little decision value with the current tools | `docs/planF_results.md` |
