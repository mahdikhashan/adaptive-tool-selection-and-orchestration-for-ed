# Research direction (adopted 2026-11-04)

**Working title:** Adaptive Tool Orchestration for Resource-Constrained UAV Inspection:
Comparing Agentic and Learned Policies Across the Edge-Cloud Continuum.

**Core claim:** the contribution is a controlled comparison of decision mechanisms
(fixed, rule-based, supervised router, contextual bandit, tool-calling SLM, hybrid SLM+router)
for selecting *and escalating* between heterogeneous perception tools under changing
network, hardware and mission constraints. Not "an LLM agent on a drone"; EVLM
(Farahani et al., IEEE EDGE 2026) already covers intent-driven edge VLM inspection on Jetson.

## Research questions
- RQ1 Policy comparison for heterogeneous tool selection under task and resource constraints.
- RQ2 Does result-aware escalation beat one-shot selection on quality per unit cost?
- RQ3 Adaptation to changing network, hardware and mission resources.
- RQ4 Controller overhead of an SLM, and when it pays for itself.
- RQ5 Sensitivity of selection to tool descriptions, schema design and ordering.

Hypotheses H1–H8 are in the original expansion document (sections 10).

## Tools
Alternative inference tools (routing candidates): `detect_defect`, `interpret_defect_local`,
`interpret_defect_cloud`. Complementary tools: `retrieve_asset_context`.
Non-inference action: `request_additional_observation`. Plus `accept` / `abstain`.

## Evaluation levels
- A. Single-step selection (oracle = cheapest tool meeting the quality threshold, offline only).
- B. Confidence-aware escalation (calibrate detector confidence first).
- C. Mission-level sequential decisions (contextual bandit first; CMDP later).

## Methodological decisions
1. **Replay benchmark first.** Run every eligible tool once per task input, cache outputs,
   quality scores and measured costs. All policies, cascades and network scenarios (S1–S10)
   are then evaluated by replay against the same cache, with network latency and failures
   injected from seeded traces. Live execution is used only to validate the replay model
   and to measure controller overhead.
2. Split by video / asset / location, never by frame.
3. Every policy sees the same telemetry and the same feasible tool set.
4. Controller latency, memory and energy count against every agentic policy.
5. Desktop numbers are development only; edge claims require Jetson measurements.
   Cloud cost is reported as calls and bytes, not energy.
6. Flight control is out of scope for the agent.

## Scope
Core: tool interfaces, per-tool benchmark, fixed/rule/supervised/bandit/SLM policies,
confidence escalation, scripted faults. Second stage: online bandit, mission constraints,
schema robustness, hybrid. Out of scope initially: multi-drone, Kubernetes, federated learning.

## Open decisions (need user input)
- Dataset: which power-line / infrastructure defect dataset, and who provides severity labels.
- Hardware: is a Jetson (which model) available, and when.
- Models: detector, local VLM, cloud VLM, and SLM controller choices.

## Positioning against the two anchor papers (added 2026-11-04)

Anchors (same group, FFG EdgeAI-Drone project):
- **EVLM** (Farahani et al., IEEE EDGE 2026, doi:10.1109/EDGE72783.2026.00014): fixed onboard pipeline:
  histogram key-frame filter -> LoRA-tuned SmolVLM-256M on every key frame -> rule-based semantic packaging.
  Baselines RVO / FOT / DCP (YOLOv8n) / VPR are separate pipelines, not choices made at runtime.
- **Cloud, Edge, or Split?** (Azimi et al., arXiv:2609.25415): profiles one model (SmolVLM-256M) in three
  placements (onboard / cloud / split after projection) under three bandwidth intervals of a 4G/LTE trace.
  No accuracy; static guideline table as the result.

Both papers name our problem as their future work:
- Split paper: "develop an adaptive runtime orchestrator that dynamically selects the deployment strategy
  according to network conditions, resource availability, input characteristics, and application requirements"
  and "incorporate accuracy into these trade-offs".
- EVLM: "adaptive runtime control that tunes filtering and evidence selection based on mission objectives and
  available uplink capacity".

### What we take from them
- Cost model priors: onboard 2.77/2.73/2.73 s, cloud 3.87/0.83/0.77 s, split 3.59/2.19/2.17 s (low/mod/high BW);
  split payload is deterministic: 36.9*T kB INT8 with T = 4*ceil(4r)+1 (r = aspect ratio); detector ~29 ms/frame
  (DCP 3.5 s per 120 frames, derived).
- Network trace: van der Hooft 4G/LTE HTTP trace (same as split paper), extended with outages.
- Data: InsPLAD (real defects) and CODEBRIM (multi-label concrete defects) for quality labels; simulated power-line
  videos (Xing et al.) for mission timing only (no defect ground truth in those videos).
- EVLM's MapIntent (keyword rules) = rule baseline for the semantic decision; EvaluateIntent = result validator.
- EVLM pipeline and the split-paper guideline become two fixed/rule baselines.

### What they leave open (our contribution)
1. Accuracy-aware placement: they vary placement of one model, we vary capability x placement.
2. Per-request decisions inside a time-varying trace, not per-interval averages.
3. Escalation: detector -> onboard VLM -> larger cloud VLM, with accept/abstain/next-key-frame.
4. Real-time feasibility: EVLM needs 6.45 s per 4 s segment (Table V), so a queue builds up; backlog is a
   state variable and a deadline constraint.
5. Agentic vs learned vs rule controllers under identical replay.

### Design consequences
- Action space: a1 YOLO defect detector onboard; a2 SmolVLM-256M-LoRA onboard; a3 same model split (INT8);
  a4 same model cloud; a5 larger VLM in cloud; a6 skip / wait for next key frame; accept / abstain.
- Energy is a secondary metric: EVLM's own model puts the compute at 5.6 W vs ~873 W propulsion. Primary constraints:
  deadline/backlog, uplink bytes, GPU memory.
- Placement given a bandwidth estimate is nearly closed-form (payloads known in advance), so an SLM probably
  adds little there; test its value on the semantic side (open intents, sufficiency judgement). Hybrid is the lead design.
- Gate detector must be defect-specific: EVLM's DCP kept 110/120 frames because components appear in almost every frame.
- Phase 0: reproduce split-paper Table III and EVLM Table V on our Jetson to calibrate the replay cost model.

## Target venue and federated framing (added 2026-11-04)

FGCS special issue "Agentic Federated Learning for the Edge Intelligence", deadline 2027-01-31.
Compute: A10 24 GB + Colab T4; no Jetson (edge costs from published EVLM / split-paper measurements,
with a sensitivity analysis).

Proposed framing: each drone is an agent that learns its tool-orchestration policy from its own execution
traces; the fleet federates that policy without sharing images.
- Split the policy: a quality estimator (does tool a solve task x?) transfers across drones, so it is federated.
  The cost/network model is site-specific, so it stays local (personalized FL).
- Clients are non-IID by visual domain (InsPLAD components, CODEBRIM) and by network regime (different trace segments).
- Compare local-only, FedAvg, personalized FL, and centralized (upper bound, needs data sharing).
- Agentic arm: a small LLM controller fine-tuned with federated LoRA (Flower) on tool-call traces, compared
  with federated bandit and router policies.
- FL communication counts against the same uplink budget as inference offloading.

## Controller choice: System One decision models (added 2026-11-04)

Per-frame tool call = closed-set decision over ~6 options with a calibrated probability (for accept/escalate thresholds).
Ladder of controllers, all under the same replay:
1. Rules (EVLM MapIntent, split-paper table, INAR-VL-style guard)
2. Numeric MLP / LinUCB over telemetry + classifier confidence
3. Fine-tuned small decision encoder (NanoJev-33M / Laya-421M class), per-option scoring  <- lead, federated object
4. Hybrid: decision encoder for intent/semantics + numeric head / code for telemetry composition
5. Zero-shot option-logit over Qwen (AnyJev / SemIf), no training
6. LLM per frame (1-3B, generates the call)
Hosted Jev excluded onboard (needs the link that fails); may appear as cloud-side reference.
Verify every checkpoint in the user's table on Hugging Face before use.
