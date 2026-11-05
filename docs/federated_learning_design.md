# Federated Learning Across a Drone Fleet: Design, Data, Model, Novelty

*Draft 2026-11-04. Target: FGCS special issue "Agentic Federated Learning for the Edge Intelligence" (deadline 2027-01-31).*

---

## 1. The idea in one paragraph

Every drone in an inspection fleet has to decide, frame by frame, **which perception tool answers the operator's question and where it runs**: a cheap onboard classifier, the onboard VLM, a split VLM, or a larger cloud VLM. It can also skip the frame or abstain. These decisions are made by a small **decision model** (the "controller"). A good controller has to know *which tool gets which kind of defect right*, and one drone learns that slowly from its own few missions. **Federated learning lets the fleet learn this together.** Each drone trains its controller on its own decision logs, sends only model updates (no images) to a server, and receives back a controller that has learned from the whole fleet. Network conditions are site-specific, so the part of the controller that models *cost* (bandwidth, latency, backlog) stays on each drone and is not shared.

---

## 2. What is federated, and what is not

| Component | What it is | Federated? | Why |
|---|---|---|---|
| **Perception tools** (classifier, SmolVLM-256M, cloud VLM) | The models that look at the image | **No** (frozen in this paper) | Out of scope. Federating the perception models themselves is a separate, more crowded line of work (e.g. FL for drone object detection, [2607.02636](https://www.alphaxiv.org/abs/2607.02636)). Possible extension. |
| **Quality estimator** $q_\theta(x,a)$ | Predicts the probability that tool $a$ answers item $x$ correctly | **Yes, the core federated object** | Visual knowledge transfers across sites: a rusty clamp looks the same everywhere |
| **Cost model** $c_\phi(a,s)$ | Predicts latency and uplink bytes of tool $a$ under drone state $s$ | **No, stays local** | Bandwidth, outages and backlog are site- and time-specific. Averaging a city drone with a mountain drone hurts both (tested as a hypothesis, §8 E3) |
| **Decision rule** | Picks $a^\* = \arg\max_a\; q_\theta(x,a) - \lambda\, c_\phi(a,s)$ under a deadline; accept / escalate / abstain by thresholding calibrated $q$ | Not learned (code) | Combining facts in code repairs the main failure mode of decision models ([2610.18579](https://www.alphaxiv.org/abs/2610.18579)) |

### 2.1 The federated model concretely

The quality estimator is a **System One decision model**: a single-pass option scorer that outputs calibrated probabilities and generates no text.

- **Backbone:** a System One decision model in three size tiers (small NanoJev-class 33M = lead, medium Laya-class 421M, large Decider-Flash-class 4B); see §2.2 for the selection and §5.1 for how each is trained in Flower.
- **Input per (item, option) pair, scored per option:**
  - text: the compiled mission spec (target defect classes, criticality) plus the tool's description
  - fused features: the tier-1 classifier's output (class, confidence), its penultimate-layer embedding (already computed, so no extra cost), and image-quality signals (blur, exposure, crop size)
- **Output:** $q_\theta(x,a) \in [0,1]$ for each tool option.
- **What is trained and sent:** LoRA adapters (only B, with A frozen) plus a small fusion head; sizes per tier are in §5.1.

**Estimated update size** (my calculation, to be measured): LoRA rank 8 on the query/value projections of a 12-layer, 384-wide encoder is about 150k parameters, roughly **0.3 MB in fp16**. A fusion head adds tens of KB. So **one federated round costs a drone about as much uplink as one JPEG frame upload** (~0.8 MB in the split paper). This comparison is the basis of the shared-uplink analysis (§6).

Per-frame inference: about six option scorings with a 33M encoder. My estimate is tens of milliseconds, inside the ~1.2 s per-key-frame budget derived from EVLM Table V. This must be measured.

### 2.2 Decision-model candidates for the experiments (small / medium / large)

Selected from the System One decision-model list the user supplied. Criteria: open weights, permissive licence, a single-pass option-scoring head (no text generation), and a size tier that maps to a deployment place.

> **Verified on Hugging Face on 2026-11-04** (details, licences and model-card lessons in `docs/asset_verification.md`). The fallback backbones remain the plan B if a checkpoint misbehaves when fine-tuned.

| Tier | Checkpoint | Size | Licence | Where it runs | How it is fine-tuned | Fallback backbone |
|---|---|---|---|---|---|---|
| **Small (lead)** | [`sdmlai/nano-jev`](https://huggingface.co/sdmlai/nano-jev) v1.0: MiniLM cross-encoder, per-option scoring | 33.4M | MIT | **Onboard, every frame** | `transformers` + PEFT LoRA | MiniLM-L12-H384 |
| **Medium** | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya): ModernBERT-large + decision head with **act/escalate head** (alt: GLiNER2.5-Decide, 486M) | 421M | Apache-2.0 | Onboard if Jetson latency allows (~35–40 ms on T4 per its card); else ground station | Laya fine-tuning notebook + PEFT LoRA | ModernBERT-large |
| **Large** | [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b): LoRA r16 adapter + pointer head on frozen Qwen3.5-4B-Base | 4B (33.8M trainable) | Apache-2.0 | **Ground station / edge server only** | Continue its LoRA via `kev.train --init_from`; federate the adapter | Qwen3.5-4B-Base + option-logit head |
| *Optional multimodal* | [`perplexity-ai/pplx-decider-flash-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2): Qwen3.5-4B decision model that **reads images** | 4.5B | Apache-2.0 | Ground station / cloud | LoRA via its bundled source package | — |
| *Training-free baseline* | Own option-logit readout on `Qwen/Qwen3.5-4B-Base` (AnyJev exists only at 31B; SemIf only as MLX ports) | 4B | Apache-2.0 | Ground station | none (federate calibration only) | — |

`LiquidAI/d1-3B` exists but uses the non-standard LFM 1.0 licence. It is dropped in favour of pplx-decider-flash as the multimodal arm.

Why these three tiers: the **small** tier is the real deployable controller (tens of ms per frame, my estimate), and it is cheap to federate. The **medium** tier tests whether a 10x bigger encoder decides better. The **large** tier tests whether LLM-scale capacity helps at all, given that a fine-tuned DeBERTa beat decision models when labels exist ([2610.00346](https://www.alphaxiv.org/abs/2610.00346)). If the large tier adds little, that is a publishable System 1 result.

**Same input for every tier**, scored per option ([2610.30970](https://www.alphaxiv.org/abs/2610.30970) found per-option scoring beats joint scoring for small encoders):
```
[INTENT] targets: rust, corrosion | critical: corrosion | abstain allowed
[OPTION] interpret_defect_local: SmolVLM-256M onboard, 7-class answer + reason
[EVIDENCE] classifier: rust p=0.58 (2nd: good p=0.31) | crop 212x180 px | blur 0.21 | exposure ok
```
Numeric evidence goes **both** into the text (for the encoder) and as a raw feature vector into a small fusion head on top of the encoder's pooled output. Decision models read numbers poorly, and they read facts but fail to combine them ([2610.18579](https://www.alphaxiv.org/abs/2610.18579)). **Telemetry (bandwidth, backlog) is not an input to $q_\theta$**: it belongs to the local cost model $c_\phi$ (§2).

Output: $q_\theta(x,a)$ = probability that tool $a$ answers item $x$ correctly, per option, followed by temperature calibration on each client's held-out towers.

---

## 3. The dataset and how it becomes a fleet

### 3.1 Data

- **ElecVQA-Bench** ([2609.27457](https://www.alphaxiv.org/abs/2609.27457)): 56,972 VQA items built from the public **InsPLAD** dataset. It provides binary screening (defect / normal) and **7-way defect typing** (rust, corrosion, missing cap, nest, torn-up, peeling paint, good), an explicit "cannot judge" option, and **tower-grouped split manifests**. The authors say it is released with construction code, splits, prompts and adapter weights. *The download location still has to be confirmed.*
- **InsPLAD** ([2311.01619](https://www.alphaxiv.org/abs/2311.01619)): the source images (10,607 UAV images; 11,525 fault-classification crops over 5 asset types and 6 defect categories, as reported by ElecVQA).
- **Network trace:** the 4G/LTE HTTP throughput trace used by the split paper (van der Hooft et al., 2016), extended with synthetic outages.
- **Cost priors:** measured Jetson numbers from EVLM, the split paper and *From Pixels to Semantics*, plus our own A10 measurements for the cloud tier.

### 3.2 Turning the dataset into drones (clients)

A client is one simulated drone, defined by **where it flies** (visual data) and **what network it has** (cost conditions).

| Client property | Source | What makes clients differ |
|---|---|---|
| Visual data | One **tower group** from ElecVQA's tower-clustered splits (InsPLAD `parent_id` grouping) | Each tower has its own mix of assets and defects, and rare defects appear on only a few towers. **This heterogeneity is real, not synthetic.** |
| Network regime | A different segment of the LTE trace per client (low / moderate / high bandwidth intervals, as in the split paper) plus a per-client outage schedule | City vs rural coverage. *This part is constructed and must be stated as such.* |
| Data volume | Unequal, following the tower sizes | Realistic: some drones fly more than others |

Planned fleet: **K ≈ 8–14 clients**, one per tower group, merging very small towers. ElecVQA's replication used 14 towers. Each client's data is split by tower into train and test, and **some towers are held out entirely** to test a drone that joins the fleet later.

---

## 4. Where the training data comes from: decision traces

The controller isn't trained on raw images. It's trained on **decision traces** built from a cached tool benchmark:

1. **Run every tool once on every item** (on the A10): classifier, SmolVLM-256M onboard-equivalent, SmolVLM split with INT8 tokens, a ~7B cloud VLM. Record for each (item, tool): correct or not (VLM text is mapped to ElecVQA's closed answer options; unparseable answers count as wrong), plus output, confidence and compute time.
   The trained tools (classifier and SmolVLM LoRA) are **cross-fitted by tower**: 3 folds over the 28 train and val towers. The model for fold N trains on the other two folds and fills the rows of fold N. One model trained on all 28 towers fills the test-tower rows. Thus no row comes from a tool that trained on the same tower, and the controller does not learn tool accuracy that is too high.
2. **Each client owns the rows of its own towers.** Nothing is pooled except in the centralized upper-bound baseline.
3. **Training example** = (item features, tool option) → label "was this tool correct?". This is a per-option binary target, which is exactly what a per-option System One scorer trains on.

There are two learning regimes, and the distinction matters for novelty:

| Regime | Feedback | What the drone knows | Prior work |
|---|---|---|---|
| **Offline (full information)** | Every tool's correctness on every training item | Complete labels from the benchmark | Federate the Router covers this for text LLMs |
| **Online (bandit feedback)** | Only the tool actually chosen, with **delayed and noisy** labels: operator verification of transmitted evidence frames (EVLM already sends these), or agreement with a stronger tool when the drone escalated anyway | Partial, late | **Open when federated.** Federate the Router lists online routing as future work; HIL-F/HIL-N (IEEE TMLCN 2024) solve the single-device, two-tier version, including the no-feedback case |

---

## 5. One federated round

```
          ┌──────────── Ground station / cloud server ─────────────┐
          │   aggregate LoRA + head updates (FedAvg / FedProx)     │
          └───────▲───────────────┬──────────────────────▲─────────┘
     update ~0.3MB│               │ global q_θ           │ (drone C offline:
                  │               ▼                      │  skips this round)
   ┌──────────────┴───┐  ┌──────────────────┐  ┌─────────┴────────┐
   │ Drone A (city)   │  │ Drone B (rural)  │  │ Drone C (mountain)│
   │ towers 1-2       │  │ towers 3-4       │  │ towers 5-6       │
   │ local q_θ train  │  │ local q_θ train  │  │ local q_θ train  │
   │ local c_φ (kept) │  │ local c_φ (kept) │  │ local c_φ (kept) │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
        images never leave the drone; only q_θ parameters do
```

1. **Server broadcasts** the current global quality estimator $q_\theta$.
2. **Each reachable drone** fine-tunes $q_\theta$ for E local epochs on its own decision traces. In reality this happens at the dock between missions; in our study, in Flower simulation on the A10.
3. **The drone uploads** only the LoRA and head deltas. **Participation depends on the drone's network trace**: a drone in an outage misses the round. This is realistic and part of the experiment.
4. **The server aggregates** (FedAvg; FedProx for heterogeneity).
5. **Personalization:** each drone keeps a small local head on top of the global model (FedPer-style). The baseline is Federate the Router's interpolation between federated and local estimators.
6. **The cost model $c_\phi$ never leaves the drone.** It is updated online from the drone's own measured latencies and bandwidth (EWMA estimator), and payload size is computed exactly from the image aspect ratio (split-paper formula: $36.9\,T$ kB INT8, with $T = 4\lceil 4r\rceil + 1$).

Framework: **Flower** (simulation mode), which suits the special issue (guest editor Nicholas Lane is a co-founder of Flower Labs).

### 5.1 Training each tier with Flower: full fine-tuning or LoRA?

**Short answer: use LoRA for all three tiers. For the small tier, run full fine-tuning once as an ablation. For the large tier, use QLoRA.** The reason is the uplink, not the GPU: on our A10 every tier trains comfortably. What decides the method is how many MB each drone must upload per round over a link that is sometimes 2 Mbps.

| Tier | Method | Trainable / uploaded | Upload per round (estimate) | Rounds × local epochs (initial guess) | A10 simulation |
|---|---|---|---|---|---|
| Small, 33M | **LoRA r=8** on attention (q,k,v,o) **+ full fusion head**; ablation: full fine-tune | ~0.2–0.4M params | **~0.5 MB** fp16 (full FT: ~67 MB, dock-only) | 50 × 1 | ~10 clients in parallel (0.1 GPU each) |
| Medium, 421M | **LoRA r=8** on attention + head | ~2M params | **~2–4 MB** | 30 × 1 | 3–4 clients in parallel (0.25 GPU each) |
| Large, 4B (Kev-4B) | Kev's own LoRA r=16 (33.8M params) on a frozen base, optionally QLoRA (4-bit base); for FL, federate only B, or retrain at rank 8 | 17–34M params | **~34–68 MB** fp16 (dock / ground-station only) | 15–20 × 1 | 1 client at a time (1.0 GPU) |
| Training-free (AnyJev/SemIf) | No weights trained. Federate only a **calibration layer**: per-option temperature + escalation thresholds (FedHLM-style) | a few dozen numbers | < 1 KB | 10 | trivial |

All upload sizes are my estimates from parameter counts and must be measured. The small-tier number is about one JPEG frame (~0.8 MB in the split paper), which is the comparison the shared-uplink experiment (§6, E5) is built on.

**A LoRA-specific pitfall in FL.** Plain FedAvg averages LoRA's A and B matrices separately, and avg(B)·avg(A) ≠ avg(B·A). The aggregated update is therefore not the average of the clients' updates, and this gets worse as clients become more heterogeneous. Two standard fixes:
- **Default: freeze A, train and federate only B** (FFA-LoRA-style). Averaging is then exact, and uploads are halved.
- **Ablation:** standard LoRA with FedAvg, and a rank-aware variant (e.g. [2608.09742](https://www.alphaxiv.org/abs/2608.09742)) if clients use different ranks.

**Training objective (same for all tiers).** For each (item, option): binary cross-entropy on "was this tool correct?", with class weights that up-weight critical-defect items. After local training, fit a temperature on the client's held-out towers so that thresholds on $q_\theta$ mean the same thing on every drone.

**Flower implementation (simulation on the A10):**
1. **ClientApp** (one per drone): loads that drone's decision traces; `get_parameters` returns **only the trainable tensors** (LoRA B matrices + fusion head) via PEFT's adapter state dict; `fit` runs E local epochs; `evaluate` returns BCE, ECE, per-class recall and selection accuracy on local held-out towers.
2. **ServerApp strategy:** FedAvg as the baseline, FedProx (μ≈0.01) for heterogeneity, and FedAdam as an option.
3. **Personalization (FedPer-style):** the fusion head's last layer stays local and is excluded from aggregation. Comparison: Federate the Router's interpolation of the federated and local estimators.
4. **Network-gated participation:** a custom `configure_fit` only selects drones whose network trace is up at that round's timestamp, and drops any update that cannot finish uploading before the round deadline at the trace bandwidth. This is our realistic participation model, and the reason for citing FGCS's deadline-aware client selection work.
5. **Byte accounting:** log the bytes of every update so that E5 can put FL traffic and inference offload traffic on the same axis.
6. **Ray backend resources:** set `num_gpus` per client from the table above; seeds fixed per run.

**Online phase (E4).** The same ClientApp, but local data arrive over time with **bandit feedback** (only the chosen tool's correctness) and **delay/noise** (operator verification of evidence frames; agreement with an escalated stronger tool). Use importance weighting for off-policy labels, plus ε-greedy or softmax exploration (softmax was strong in federated contextual bandits, [2303.10218](https://www.alphaxiv.org/abs/2303.10218)). HIL-F/HIL-N is the single-drone reference.

---

## 6. The systems angle: learning and inference share one uplink

A drone's uplink carries **cloud inference requests** (images or visual tokens) and **federated updates**. Every MB spent on FL is a MB not available for offloading a hard frame. We measure:
- FL traffic per drone per mission vs inference offload traffic
- the drop in accuracy or deadline-met rate when FL rounds run during a mission, compared with rounds at the dock
- the accuracy gain per MB of federated traffic

The general coupling of FL training and inference on UAVs is studied by Zhao et al. (IEEE IoT-J 2026), who choose which UAVs train so that training doesn't hurt inference service. Our narrower case: the model being federated *is* the inference controller, and one drone's own uplink carries both its offloading and its FL updates.

---

## 7. Baselines

| Group | Baseline |
|---|---|
| No learning | EVLM fixed pipeline; split-paper guideline table; INAR-VL-style rule (complexity signals + 15 Mbps bandwidth guard); always-classifier / always-VLM / always-cloud |
| Learning, no federation | **Local-only** controller per drone; **centralized** controller trained on pooled data (upper bound, requires sharing data) |
| Federated, prior design | **Federate the Router** MLP router, offline FedAvg, with its adaptive personalization; **FedHLM-style** federated scalar confidence thresholds per tier (IEEE IoT-J 2025) |
| Online escalation, prior design | **HIL-F / HIL-N** hierarchical-inference online learning (IEEE TMLCN 2024), per drone, not federated; HIL-N also covers the no-feedback case |
| Federated, ours (ablations) | Federated numeric MLP/LinUCB; federated decision encoder (lead); federate q only vs q **and** c; with and without personalization |
| Controller references | Zero-shot option-logit (AnyJev/SemIf over Qwen); LLM generating the call per frame; offline oracle (cheapest correct tool, analysis only) |

---

## 8. Experiments

| ID | Question | Expected outcome (hypothesis, not result) |
|---|---|---|
| **E1** | Does federation beat local-only, and how close does it get to centralized? | Federated beats local, especially for small clients |
| **E2** | Does federation improve **rare-defect** recall (defects seen on only a few towers)? | Largest gain is here; this is the main argument for FL |
| **E3** | Federate q only, or q and c? | Sharing c hurts under site-specific networks |
| **E4** | Online phase with bandit, delayed and noisy feedback | Federated online beats local online; quantify sensitivity to label delay and noise |
| **E5** | Shared uplink: FL traffic vs inference offloading | Dock-time rounds are nearly free; mid-mission rounds have a measurable cost |
| **E6** | A new drone joins (held-out towers) | Federated model gives a good cold start |
| **E7** | Which controller benefits most from federation: numeric MLP, decision encoder, or LLM LoRA? | The decision encoder gives the best accuracy per uplink byte |

All experiments use the cached replay, the same seeded network traces for every policy, several seeds, and tower-level splits.

---

## 9. Metrics

- **Final-answer quality:** separate rates for misses, false alarms and abstentions, following ElecVQA. Critical-defect recall. Cost-weighted risk $R = c_{\text{miss}}\cdot\text{miss} + c_{\text{FA}}\cdot\text{FA} + \kappa\cdot\text{abstain}$. **An answer after its deadline counts as a miss.**
- **Accuracy–cost frontier:** area under the curve while sweeping $\lambda$; accuracy at matched cost.
- **Decision quality:** agreement with the oracle, regret, over- and under-escalation rates.
- **Federated:** per-client, global and **worst-client** accuracy; rare-class recall; MB uplinked per drone; rounds needed to converge.
- **Calibration:** ECE of $q_\theta$ on held-out towers. Thresholds are only meaningful if $q$ is calibrated.

---

## 10. How novel is it?

Search scope (2026-11-04): alphaXiv keyword and embedding search (about 250 hits screened, about 20 papers read), then an **IEEE Xplore pass** (about 300 IEEE records over 9 searches, 12 read at metadata level; details in `docs/novelty_check.md`). **Elsevier (including FGCS itself), ACM and Springer are not yet covered.** These are bounded claims, not proof of absence.

### 10.1 Closest prior work and the gap each leaves

| Prior work | Covers | Leaves open |
|---|---|---|
| **Federate the Router** ([2601.22318](https://www.alphaxiv.org/abs/2601.22318)) | Federated routing among text LLMs; offline; accuracy and cost estimators; adaptive personalization | Cost is a global API price; offline only (authors: online routing is future work); no perception, placement, escalation, or network |
| **INAR-VL** ([2605.18853](https://www.alphaxiv.org/abs/2605.18853)) | Accuracy-aware edge–cloud VLM routing per request | No federation; static bandwidth guard; escalation is future work |
| **AVERY** ([2511.18151](https://www.alphaxiv.org/abs/2511.18151)) | Network-adaptive VLM split on UAVs | Single model, chooses compression level; no federation |
| **SYNAPSE** ([2609.06815](https://www.alphaxiv.org/abs/2609.06815)) | Federates tool-routing knowledge as typed text artifacts across frozen LLMs | Text tools; no parameters learned; no edge cost |
| **Federated contextual bandits** ([2303.10218](https://www.alphaxiv.org/abs/2303.10218), [2605.00500](https://www.alphaxiv.org/abs/2605.00500)) | Generic federated exploration | Not applied to tool or placement choice |
| **Decision models for edge orchestration** ([2609.22753](https://www.alphaxiv.org/abs/2609.22753)), **JEV-as-a-Judge cascades** ([2609.26550](https://www.alphaxiv.org/abs/2609.26550)) | System One models replacing LLMs on latency-critical paths; confidence-based accept/escalate | **No federation found**; no perception tools |
| **EdgeAI-Drone group** (EVLM, split paper, [2609.18448](https://www.alphaxiv.org/abs/2609.18448)) | Onboard VLM, split profiling, calls for detector-triggered cascades and fleet collaboration | Decisions are fixed; no learning across drones |
| **FedHLM** (IEEE IoT-J 2025, [doi](https://doi.org/10.1109/jiot.2025.3618095)) | **Federated learning of uncertainty thresholds** for SLM-to-LLM escalation; hierarchical aggregation | Scalar thresholds; 2 tiers; text classification; no placement or network-varying cost |
| **CAFED** (Globecom 2025, [doi](https://doi.org/10.1109/globecom59602.2025.11432532)) | Class-aware **federated early-exit** networks; personalized exits for each client's critical classes | Exits inside one network; no tool or placement choice |
| **Hierarchical Inference, HIL-F / HIL-N** (IEEE TMLCN 2024, [doi](https://doi.org/10.1109/tmlcn.2024.3366501)) | **Online** learning of accept-vs-offload from the small model's confidence, with full and no-local feedback, regret bounds | Single device, two tiers, not federated |
| **Zhao et al.** (IEEE IoT-J 2026, [doi](https://doi.org/10.1109/jiot.2026.3744411)); **Yang et al.** (IEEE IoT-J 2026, [doi](https://doi.org/10.1109/jiot.2026.3664516)) | **FL training and inference compete for UAV/IoT resources**; MAB or SAC scheduling of FL participation | The FL model is a generic task model, not the inference controller; contention between nodes, not one drone's own uplink |
| **Federated offloading policies** (Rahbari et al., IEEE Access 2021, [doi](https://doi.org/10.1109/access.2021.3104117); federated DRL offloading, e.g. IEEE TVT 2024, [doi](https://doi.org/10.1109/tvt.2024.3429507)) | Drones/UAVs federate offloading decisions | Accuracy-blind: generic tasks, latency/energy only |
| **Li et al.** (ACEGRS 2026, [doi](https://doi.org/10.23919/acegrscps00016.2026.00046)) | FL + edge computing for UAV transmission-line defect identification | Federates the perception model, not the controller; abstract unavailable, so get the full text |

### 10.2 What we can claim (revised after the IEEE pass)

Not new on its own: federated routing (Federate the Router), **federated escalation thresholds (FedHLM)**, **federated early exits (CAFED)**, **online accept/offload learning (HIL)**, accuracy-aware edge VLM routing (INAR-VL), network-adaptive UAV VLM splitting (AVERY), **FL–inference resource coupling on UAVs (Zhao 2026)**, federated offloading policies, System One decision models, "agentic FL" as a term.

**New, as far as both searches show:**
1. **Federated training of a System One decision model as a fleet's tool-orchestration controller.** No hit in arXiv or IEEE: decision-model papers have no federation; federated routing and escalation papers use MLPs, K-means or scalar thresholds. **This is now the lead claim.**
2. **Federated escalation over a multi-tier capability × placement action set**: a learned per-option quality estimator rather than FedHLM's scalar thresholds or CAFED's in-network exits, under time-varying networks, with delayed and noisy online feedback (federating what HIL does on one device).
3. **What to federate:** sharing the quality estimator while keeping the site- and time-dependent cost model local, as an explicit, tested design rule. Goes beyond Federate the Router's per-estimator interpolation; CAFED's shared/personal split is by layer, not by quality vs cost.
4. **The federated controller and inference offloading share one drone's uplink**, with participation gated by that drone's own network trace. *Downgraded from headline to refinement:* Zhao et al. 2026 already study FL–inference contention on UAVs.
5. **Real non-IID fleet clients** from InsPLAD tower groups, evaluated with inspection-grade error metrics (misses vs hallucinations vs abstentions).

### 10.3 Novelty risks

- The field is moving fast: many relevant preprints are from Sep–Oct 2026. **Re-run the search in early January**, just before submission, and add the **Elsevier (especially FGCS itself), ACM and Springer** venues not yet covered. Reviewers from this special issue will know the FGCS back catalogue.
- Claims 2 and 4 are refinements of IEEE work (FedHLM, HIL, Zhao 2026). Lead with claim 1, and describe 2–4 as extensions.
- **Unresolved high-risk overlap (Elsevier, snippet only):** *From open-vocabulary detection to structural assessment: An agentic federated approach for guardrail monitoring* ([ScienceDirect S2772503026000459](https://www.sciencedirect.com/science/article/pii/S2772503026000459)) combines edge-to-cloud escalation of uncertain detections, federated adapter updates and an orchestration layer for inference routing and FL rounds, in infrastructure inspection. It must be read in full before claims 1–2 are finalized. The Elsevier pass so far is provisional (web snippets); see `docs/novelty_check.md`.
- The EdgeAI-Drone group has announced an adaptive orchestrator as future work. Our work should be framed as building on theirs, which fits the PhD goal anyway.

---

## 11. Honest weaknesses to address in the paper

1. **Why federate, if one utility owns all drones?** The privacy argument is weak in that case. Lead with: (a) bandwidth, since shipping raw imagery is what EVLM was built to avoid; (b) multi-operator or contractor fleets that cannot share imagery; (c) rare defects too sparse per drone. Then show (c) with E2.
2. **Network heterogeneity is constructed** from one LTE trace. State this clearly, and run a sensitivity analysis over the regime assignment.
3. **No Jetson.** Edge costs come from published measurements; add a ±50% cost-model sensitivity analysis.
4. **Clients may be too similar** for FL to matter. Measure the heterogeneity between clients (label-distribution distance) and report it.
5. **Unverified items:** the ElecVQA-Bench download location; the decision-model checkpoints (NanoJev, Laya: weights and licence).

---

## 12. Possible extension (not in this paper)

Federated LoRA of the **perception VLM** itself (SmolVLM-256M) across towers. This is closer to standard federated fine-tuning and more crowded, but it would let the fleet improve the tools, not just the choice among them.

---

## 13. Next steps (dated plan to the 31 January 2027 deadline)

| When | Step | Output | Blocks |
|---|---|---|---|
| **Nov 4–7** | **Verify assets.** Check NanoJev, Laya, Decider Flash, d1, AnyJev/SemIf on Hugging Face (exists? licence? input format? fine-tunable?). Locate the ElecVQA-Bench release and InsPLAD. Get the LTE trace. Fix the Elsevier server and **read the guardrail paper** in full. | Asset table with links/licences; final novelty verdict | Everything |
| **Nov 5–8** | **Project setup.** uv project with transformers, peft, flwr, ultralytics; repo layout; A10 access configured as the compute target | Reproducible environment | Tool cache |
| **Nov 8–21** | **Tool cache.** (1) Train the tier-1 defect classifier on the training towers at the matched resolution. (2) LoRA-tune SmolVLM-256M on the training towers following EVLM's recipe, since EVLM's adapter is not public. (3) Run classifier, SmolVLM onboard-equivalent, SmolVLM split (INT8 tokens), and a ~7B cloud VLM on **every** item; log correctness, outputs and timings. | `tool_cache` table: one row per (item, tool) | All controller work |
| **Nov 22–Dec 1** | **Replay simulator + baselines.** Cost model (split-paper payload formula, published Jetson latencies, A10 cloud timings, LTE trace + outages). Rule baselines: EVLM fixed, split-paper table, INAR-VL-style guard, FedHLM-style threshold, HIL-F/N. Oracle. | First accuracy–cost frontier | Learned controllers |
| **Dec 2–12** | **Centralized controllers.** Numeric MLP / LinUCB; small, medium and large decision models (LoRA); training-free option-logit; LLM-per-frame reference. Calibration. | Controller ladder, centralized (upper bound) | FL runs |
| **Dec 13–29** | **Flower FL.** E1 (local vs FedAvg vs centralized), E2 (rare defects), E3 (federate q vs q+c), E7 (which tier benefits), with network-gated participation. Then E4 (online, bandit/delayed feedback), E5 (shared uplink), E6 (new drone). | Main results | Paper |
| **Dec 30–Jan 5** | Seeds, ablations (full FT vs LoRA, frozen-A vs standard LoRA, FedProx μ), ±50% cost-model sensitivity | Robustness section | Paper |
| **Jan 6–19** | **Writing**: FGCS long paper, figures, related work (with the final Elsevier/ACM search) | Draft | Submission |
| **Jan 20–31** | Internal review, polish, **submit ~Jan 24** | Submission | — |

**Decision points:**
- **After Nov 7:** if NanoJev/Laya/Decider Flash don't exist as described, switch to the fallback backbones (§2.2). The design does not change.
- **After Nov 21:** if the tools' correctness hardly differs across items, there is little for any controller to learn. Then increase tool diversity (add a stronger cloud VLM, or a second classifier) before going further.
- **After Dec 12:** if the small decision model doesn't beat the numeric MLP centrally, report it honestly. In that case the FL study runs with both, and the paper's message becomes "what to federate", not "which model".
- **If the guardrail paper turns out to federate a controller**, reframe the lead claim around online escalation over capability × placement, plus decision models.
