# Novelty check: federated orchestration of perception tools (2026-11-04)

Scope: alphaXiv keyword + embedding search, 3 loops, ~200 hits screened, 13 read in full or by report;
then an IEEE Xplore pass (~300 records, 12 read at metadata level; see "Rerun" below).
Elsevier, ACM and Springer venues are still not covered. Rerun before submission.

## Closest prior work

| Paper | What it does | What it leaves open (for us) |
|---|---|---|
| Federate the Router (Askin et al., 2601.22318) | First federated LLM router. MLP and K-means routers estimate acc(x,m) and cost(x,m), FedAvg over clients with offline query-model evaluations, adaptive personalization that interpolates federated and local estimators separately for acc and cost. | Text LLMs only; cost = API price, global and static; offline only. Authors: "Online routing ... remains a promising direction for future work." No escalation, no placement, no network. |
| INAR-VL (Sabanovic, Maliakel, Brandic, TU Wien, 2605.18853) | Per-request edge-cloud VLM routing from cheap image-quality and text-complexity signals; Pareto over model x precision x resolution x placement; static bandwidth guard (15 Mbps). | Generic VQA; bandwidth swept statically, not a time-varying trace; no learning across devices; future work: "uncertainty-aware escalation policies". |
| AVERY (Bhattacharjya et al., 2511.18151) | UAV disaster response; dual-stream split of LISA-7B; onboard controller picks compression model from network + intent at runtime. | Single model, chooses compression level not capability; no federation. |
| Typed Federated Artifacts / SYNAPSE (2609.06815) | Federates tool-routing knowledge as typed JSON artifacts across frozen LLM agents. | Text tools. Cautionary result: a TF-IDF classifier beats every LLM routing arm by 26-48 points. Supports our H2/H4. |
| MHR-Router (2610.13278) | VLM expert routing with success/failure memories. | Accuracy only; no cost, placement or federation. |
| Agarwal, McMahan, Xu (2303.10218) | Empirical federated contextual bandits; softmax exploration surprisingly strong. | Generic; gives our federated bandit baseline. |
| FSCLB (2605.00500) | Communication-efficient federated linear contextual bandits via sketching. | Algorithmic; usable for the FL communication budget. |
| From Pixels to Semantics (Farahani ... Carnevale ... Dustdar, 2609.18448) | Same group (Carnevale is an SI guest editor). Jetson Orin: YOLO26M 0.14 s vs SmolVLM-256 2.8 s per image; latency, not energy, is the bottleneck; calls for YOLO-triggered VLM cascades and "collaborative edge intelligence" invoking larger models selectively. | Position + feasibility only; cascade not implemented; no federation. |
| ElecVQA-Bench (2609.27457) | 56,972-item VQA benchmark from InsPLAD; 7-way defect typing; tower-grouped splits; explicit abstention; released with code, splits, adapters. | Finds no stable VLM advantage over ResNet/Swin once resolution is matched -> cheap classifiers are often enough -> cascades matter. Tower groups = natural non-IID FL clients. |
| Agentic-FL position papers (2604.04895, 2609.35914, 2610.01195 FAO survey, 2606.03143 FederatedSkill) | Mostly "agents orchestrate FL training". FAO lists tool use as an axis of federated agent optimization. | Framing only; no edge perception orchestration. |

Also relevant: CEMA-Router (2610.26171), CR^2 (2605.12001), online constrained LLM selection (2606.17489),
risk-adaptive edge-cloud VLM for driving (2608.14991), FL head personalization for split computing (2610.27952),
FAuNO federated RL offloading (2506.02668).

## Rerun 2026-11-04: IEEE Xplore pass (journals and conferences)

Tool: the ieee-xplore MCP server (OpenAlex index restricted to IEEE publications; no IEEE API key, no
institutional proxy, so abstracts and open-access text only). About 300 IEEE records screened over 9 searches;
12 read at full-metadata level. **Still not covered: Elsevier (including FGCS itself), ACM, Springer.**
The orx OpenAlex search was still rate-limited.

New prior work that changes our claims:

| Paper | What it does | Effect on our claims |
|---|---|---|
| **FedHLM**, Solat, Lee, Seif, Niyato, Poor, IEEE IoT-J 2025, [doi:10.1109/jiot.2025.3618095](https://doi.org/10.1109/jiot.2025.3618095) | Federated learning of the token-level **uncertainty thresholds** that decide when an edge SLM escalates to a cloud LLM; hierarchical aggregation | **Federated escalation is not new in principle.** Ours differs: multi-tier perception tools over capability x placement, a learned per-option estimator instead of scalar thresholds, site-specific network cost, online delayed feedback |
| **CAFED**, Ayyat & Nadeem, Globecom 2025, [doi:10.1109/globecom59602.2025.11432532](https://doi.org/10.1109/globecom59602.2025.11432532) | Class-aware federated early-exit networks; shared lower layers, personalized exits per client's class priorities | Federated "when to stop", inside one network; no placement, no network |
| **Hierarchical Inference (HIL-F / HIL-N)**, Moothedath, Champati, Gross, IEEE TMLCN 2024, [doi:10.1109/tmlcn.2024.3366501](https://doi.org/10.1109/tmlcn.2024.3366501) | Online learning of when a small edge model's answer is accepted vs offloaded to a large server model; full-feedback and **no-local-feedback** regimes with regret bounds | Single device, two tiers, not federated. **Must cite and use as the online-escalation baseline**; it already handles missing feedback |
| **Zhao et al.**, IEEE IoT-J 2026, [doi:10.1109/jiot.2026.3744411](https://doi.org/10.1109/jiot.2026.3744411) | UAV edge network: a multi-play MAB picks which UAVs train FL so that training does not hurt inference requests (delay, rejected access) | **The coupling of FL and inference on UAVs is studied.** Ours is narrower: the federated model *is* the inference controller, and one drone's own uplink carries both |
| **Yang et al.**, IEEE IoT-J 2026, [doi:10.1109/jiot.2026.3664516](https://doi.org/10.1109/jiot.2026.3664516) | Joint federated continual learning and inference offloading via SAC | Same coupling theme, generic IoT |
| **RACE**, Wang et al., IEEE TSC 2026, [doi:10.1109/tsc.2026.3726297](https://doi.org/10.1109/tsc.2026.3726297) | Risk-controlled edge-to-cloud LLM escalation with online adaptive calibration (conformal-style) | Escalation with error-rate guarantees, not federated; useful calibration method |
| **Rahbari et al.**, IEEE Access 2021, [doi:10.1109/access.2021.3104117](https://doi.org/10.1109/access.2021.3104117); federated DRL offloading (e.g. Huang et al., IEEE TVT 2024, [doi:10.1109/tvt.2024.3429507](https://doi.org/10.1109/tvt.2024.3429507)) | Federated learning of computation-offloading policies in drone swarms / UAV MEC | Federated offloading policies exist, but they are accuracy-blind (latency/energy of generic tasks) |
| **Li et al.**, ACEGRS 2026, [doi:10.23919/acegrscps00016.2026.00046](https://doi.org/10.23919/acegrscps00016.2026.00046) | FL + edge computing for real-time defect identification in UAV transmission-line images (abstract unavailable) | Federates the perception model in our exact domain, not the controller. Get the full text before submission |

No IEEE record found for: federated System One / decision models; federated tool selection for perception;
quality-vs-cost split of what to federate.

## Verdict (revised after the IEEE pass)

Not novel by itself: federated routing (Federate the Router), **federated escalation thresholds (FedHLM)**,
**federated early exits (CAFED)**, **online accept/offload learning (HIL)**, accuracy-aware edge-cloud VLM
routing (INAR-VL), network-adaptive UAV VLM split (AVERY), **FL-inference resource coupling on UAVs (Zhao 2026)**,
federated offloading policies (Rahbari, FDRL line), and "agentic FL" as a term.

Unclaimed intersection among the papers screened (bounded claim, not proof):
1. **Federated System One decision model as a fleet's tool-orchestration controller.** No hit in arXiv or IEEE. Strongest claim.
2. **Federated escalation over a multi-tier capability x placement action set** with a learned per-option
   quality estimator, under time-varying networks and delayed/noisy online feedback. Refines FedHLM
   (scalar thresholds, 2 tiers, text) and HIL (single device, not federated).
3. **What to federate:** quality estimator shared, cost/network model local and online, as an explicit,
   tested design rule. Partial prior: Federate the Router's per-estimator interpolation; CAFED's shared-lower /
   personal-upper split is architectural, not quality vs cost.
4. **The federated controller and inference offloading share one drone's uplink**, with participation gated by
   the drone's own trace. Now a refinement of Zhao 2026 / Yang 2026, not a headline claim.
5. Real non-IID clients from InsPLAD tower groups (ElecVQA splits), with inspection-grade error metrics.

Still to search: Elsevier (FGCS, Computer Networks, JPDC), ACM (MobiSys, SenSys, MobiCom), Springer.

## Design consequences
- Mandatory baselines: INAR-VL-style rule (complexity signals + bandwidth guard), Federate-the-Router MLP
  (offline FedAvg), local-only, centralized, EVLM fixed pipeline, split-paper rules,
  **FedHLM-style federated confidence threshold** (scalar per tier), **HIL-F / HIL-N** online accept/offload.
- Tier 1 tool = defect classifier/detector on crops at matched resolution (ElecVQA finding).
- Report misses / hallucinations / abstentions separately (ElecVQA framework), with critical-defect recall.
- Clients = InsPLAD tower groups x network regime.

## Elsevier / FGCS pass: PROVISIONAL (2026-11-04)

The Elsevier MCP server failed to connect ("Connection closed", then "recent failure cached"), and the orx OpenAlex
search was still rate-limited. This pass therefore used **web search restricted to sciencedirect.com (snippets only;
nothing read in full)**. Redo it with the Elsevier server before submission.

FGCS papers are identified by the ISSN in the ScienceDirect PII (FGCS = 0167-739X).

| Paper | Venue (from PII) | What the snippet says | Risk to our claims |
|---|---|---|---|
| *From open-vocabulary detection to structural assessment: An agentic federated approach for guardrail monitoring*, [S2772503026000459](https://www.sciencedirect.com/science/article/pii/S2772503026000459) | Elsevier journal, ISSN 2772-5030 (not FGCS) | Edge-fog-cloud infrastructure inspection; edge validates detections, **uncertain cases escalate to the cloud**; edge nodes upload encrypted **adapter updates to a federated coordinator**; an orchestration layer manages **inference routing and FL rounds** | **HIGH: closest overlap found so far** (agentic + federated + escalation + infrastructure inspection). Must read in full. Likely difference (unverified): it federates the perception adapters, not a decision-model controller |
| *Multi-type task offloading ... federated deep Q-network (M2FD)*, [S0167739X23001383](https://www.sciencedirect.com/science/article/abs/pii/S0167739X23001383) | **FGCS** 2023 | Federated DQN for IoT offloading, time/energy objective | Low: accuracy-blind federated offloading (same class as Rahbari / FDRL) |
| *FedOffloading for edge intelligent controllers*, [S0167739X24004904](https://www.sciencedirect.com/science/article/abs/pii/S0167739X24004904) | **FGCS** 2024 | Offloads FL training to edge servers with Laplace-noise privacy | Low: offloading of training, not inference decisions |
| *Client selection of multi-criteria and adaptive deadline in FL*, [S0167739X26004115](https://www.sciencedirect.com/science/article/abs/pii/S0167739X26004115) | **FGCS** 2026 | FL client selection with deadlines | Low: relevant to deadline-gated participation (cite in §5 of the design doc) |
| *Modelling federated learning workflows for the cloud-to-edge continuum*, [S1569190X26000821](https://www.sciencedirect.com/science/article/pii/S1569190X26000821) | Elsevier (ISSN 1569-190X, not FGCS) | FL workflow modelling across the continuum | Low |

Supporting signal from arXiv found in the same pass: *Smart Adaptive Computing Across the Continuum: LLMs in
IoT-Edge-Cloud Resource Management* ([2609.09348](https://www.alphaxiv.org/abs/2609.09348)), a survey whose snippet states that
no work combines LLM orchestration with federated training of the agent layer. Snippet only; read before citing.
Also new from alphaXiv: *Edge-Cloud Collaborative Pothole Detection via Onboard Event Screening and Federated Temporal
Segmentation* ([2605.10055](https://www.alphaxiv.org/abs/2605.10055)), unread.

No FGCS paper found (snippet level) on federated tool or model selection for inference, or on decision-model controllers.
Verdict unchanged, with one new HIGH-risk item to resolve: the guardrail paper.
