# Documentation

This folder contains the documents of the project "Adaptive Tool Selection and Orchestration for Edge-AI Drone Inspection".
The project asks two questions. Which AI tool must an inspection drone use for each image? What can a fleet of drones learn together with federated learning?

## Start here

The folder `understanding/` explains the full project from the basic concepts. The reader is a first-year master student in AI with a background in mechanical engineering.

| File | Content |
|---|---|
| `understanding/00_foundations.md` | Classification, softmax, risk, calibration, label shift, FedAvg, MDP, PPO, imitation, LoRA, quantisation |
| `understanding/01_routing_controller.md` | Route 1: a controller for each frame, federated across operator regions, one-sided feedback |
| `understanding/02_federated_perception.md` | Route 2: fine-tuning and federated training of laya-vision, 4-bit models |
| `understanding/03_agentic_tool_selection.md` | Route 3: a VLM agent, fusion of tool answers, PPO tool selection, deferral, mission segments |

## The three routes

| Route | Branch | Question | Main result |
|---|---|---|---|
| 1. Routing controller | `vision-fl` (and earlier) | Can a small federated controller select the tool for each frame? | FedAvg equals central training (115 against 156 for a cascade) with about 7 MB of upload. With one-sided feedback, a 5% audit is necessary. |
| 2. Federated perception | `vision-fl`, `vision-quant` | Can the fleet improve the perception model itself? | Yes, but at 0.4 to 41 GB of upload for each run. Small updates need a new calibration. |
| 3. Agentic tool selection | `agentic-ppo-v2` | Can a VLM agent or a PPO policy select tools under time limits? | The VLM agent gets worse with tools. Fusion and deferral give the gains. PPO matches tuned rules but does not reliably beat them. |

## Documents by route

### General

| File | Content |
|---|---|
| `research_direction.md` | Research question and anchor papers |
| `novelty_check.md` | Literature check for the novelty |
| `architecture.md` | Software architecture |
| `asset_verification.md` | Step 1: models, datasets and the GPU server |
| `risk_weights.md` | Why a miss costs 10, a false alarm 1 and an abstention 0.5 |

### Route 1: routing controller

| File | Content |
|---|---|
| `step2_results.md` | The tool cache: accuracy, time and upload of each tool |
| `step3_results.md` | The replay and the baselines |
| `step4_results.md` | Learned controllers with a correctness target |
| `step4b_results.md` | Learned controllers with a choice target |
| `decision_model_plans.md`, `decision_model_usage.md`, `controller_evidence_options.md` | How the decision models (nano-jev, Laya) enter the experiments |
| `planA_results.md` | Controllers with few labels |
| `planF_results.md` | Free-text operator intents |
| `findings.md` | All findings of steps 1 to 4b, plan A and plan F |
| `federated_learning_design.md` | Design of federated learning across the fleet |
| `feedback_model.md` | The one-sided feedback model of the federated controller |

### Route 2: federated perception

| File | Content |
|---|---|
| `vision_fl_design.md`, `vision_results.md` | Zero-shot test of laya-vision |
| `vision_finetune.md`, `vision_finetune_results.md` | Fine-tuning, round 1 |
| `vision_finetune_round2.md`, `vision_finetune_round2_results.md` | Fine-tuning, round 2 |
| `vision_fl_plan.md`, `vision_fl_results.md` | Federated round A and two start models |
| `vision_fl_round_b_plan.md`, `vision_fl_bcd_results.md` | Federated rounds B, C and D |
| `vision_fl_round_e_plan.md`, `vision_fl_round_e_results.md` | Federated round E |
| `quant_plan.md`, `quant_results.md` | 4-bit models |

### Route 3: agentic tool selection

| File | Content |
|---|---|
| `ppo_v2_gpu.md`, `ppo_v2_interfaces.md` | The GPU pilot for PPO with Flower and its interfaces |
| `agentic_demo.md` | A local demo of tool selection on the laptop |
| `tool_selection_results.md` | All results of route 3, sections 1 to 14 |
| `tool_corpus_plan.md` | Plan: about 24 tools and laya-vision as a tool selector (Plan, Retrieve, Select) |

## Other places

- The paper: `paper_2col.tex` (current) and `paper.tex` (earlier version about federated fine-tuning).
- The figures: `figs/`.
- The result files: `results/`.

## Writing rule

The documents in this folder use ASD-STE100 Simplified Technical English. The linter of the `asd-ste100` skill must report 0 violations.
