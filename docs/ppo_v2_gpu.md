# Protocol v2: single-seed GPU pilot

Implementation branch: `agentic-ppo-v2`. This is runnable pipeline code, not a
completed experiment. No model/controller training was run during implementation.
Train only on CUDA. The target is one booked RTX 2080 Ti (11 GB), native PyTorch,
FP32 edge-tool inference, NF4/fp16 Qwen8B inference, one pretrained YOLO checkpoint, seed 0. Quantization
scripts remain unchanged. Extra seeds come after the pilot.

Federated clients are the **drones themselves**. Each owns its actor, local
critic, optimizer and trajectories. This VM emulates five drones plus an
actor-only aggregation coordinator; there are no base stations. Perception
models remain frozen while drone controller policies learn. The cloud role is
the larger Qwen model, loaded on this same VM for this simulation.

Latest user decision: **do not fine-tune YOLO now**. The default is official
pretrained YOLO26n-cls as downloaded, frozen for inference. Its ImageNet outputs
provide generic visual context to Laya; they are not defect predictions. The
six-condition classification comes from frozen Laya. Native YOLO top-k labels,
entropy, margin and validity are recorded; six-way YOLO condition probabilities
are explicitly missing. A separately fine-tuned six-class checkpoint remains an
optional later tool comparison, not a prerequisite for this pilot.

## Pull and provision

From your existing repository directory on the VM:

```bash
git fetch github
git switch --track github/agentic-ppo-v2
git pull --ff-only github agentic-ppo-v2
source scripts/server_env.sh
export CUDA_VISIBLE_DEVICES=0
uv sync --locked --extra gpu-tools
```

Use the booked GPU's actual index. If the remote is called `origin`, replace
`github` with `origin`. If the local branch already exists, use
`git switch agentic-ppo-v2` instead of `--track`.

`gpu-tools` pins Ultralytics 8.4.208 alongside the existing Linux CUDA12.6 torch
lock. No TensorRT conversion, export or runtime is used. Pipeline dependencies
must resolve from the committed lock. Put data/model/cache outputs under
`FEDORCH_WORK` rather than committing them.

If the datasets are not already provisioned:

```bash
uv run --locked python scripts/get_insplad.py
uv run --locked python scripts/step1_verify_assets.py data
```

This fetches data only. It does not train anything. The existing setup script
uses the server's scratch-storage layout; set `FEDORCH_WORK` before sourcing
`server_env.sh` if your VM uses another location.

## Download initialization checkpoints and prepare data

```bash
uv run --locked --extra gpu-tools python scripts/ppo_v2_fetch_models.py \
  --output "$FEDORCH_WORK/models/ppo_v2"
bash scripts/runs/ppo_v2_gpu.sh prepare
```

The download command fetches one pretrained ImageNet YOLO26n classifier,
the pinned published Laya checkpoint, SmolVLM-256M base and Qwen3-VL-8B. It does not train or
run inference. Your existing fine-tuned Smol LoRA adapter is still required:

```bash
export PPO_V2_SMOL_ADAPTER="$FEDORCH_WORK/runs/step2/smolvlm_lora_full"
```

Override `PPO_V2_LAYA` or `PPO_V2_SMOL_BASE` if using different existing local
directories. Laya's native loader may also need its referenced backbone in the
HF cache. First-load/download time is outside the declared warm-service mission
and separately recorded. Frozen checkpoint choice/provenance is recorded; no
Laya/Smol weights are trained by the controller pipeline.

Preparation writes the immutable `small` bundle under
`$FEDORCH_WORK/runs/ppo_v2/`, retaining approximately10% source-grouped training data,
with all validation/test metadata unchanged. The existing five tower-balanced
partitions are assigned to five simulated drones. The realized audit mask is independent Bernoulli p=.05,
fixed by seed and item ID, persisted once and reused across all methods.

Source-image questions are sampled together, no source/tower may cross splits,
and labels are kept separate from tool observations. Privileged class-aware
sampling can use training labels: **5% is the controller feedback budget, not
the total label budget of all previously trained tools**. No new YOLO fitting
occurs in this pilot. Record
the provenance of the existing Smol adapter too; a tool trained on validation
towers is a development condition, not a clean held-out tool validation claim.

## Optional later: fine-tune one YOLO classifier

**Skip this section for the current pilot.** The profile/cache stages use the
downloaded pretrained checkpoint through `PPO_V2_YOLO_BASE` (or an explicit
`PPO_V2_YOLO_CHECKPOINT` override). Continue to the profiling section below.

If you decide to fine-tune later, the existing optional stage first prepares a
`full` bundle and then trains one checkpoint:

```bash
bash scripts/runs/ppo_v2_gpu.sh yolo
```

Equivalent explicit command:

```bash
uv run --locked python scripts/ppo_v2_prepare.py \
  --output "$FEDORCH_WORK/runs/ppo_v2/full" --fraction 1 --seed 0
uv run --locked --extra gpu-tools python scripts/ppo_v2_yolo.py \
  --bundle "$FEDORCH_WORK/runs/ppo_v2/full" \
  --output "$FEDORCH_WORK/runs/ppo_v2/yolo" \
  --base "$FEDORCH_WORK/models/ppo_v2/yolo26n-cls.pt" \
  --epochs 20 --batch 64 --imgsz 224 --workers 0 --device 0
```

This produces `frozen_yolo.pt` plus class-order, dataset and checkpoint hashes.
To select it later, set `PPO_V2_YOLO_CHECKPOINT` to that file and create a fresh
tool cache with matched settings for every controller. It refuses to overwrite an
existing frozen checkpoint. Training and validation use the same deterministic
whole-image letterbox transform, with no synthetic augmentation/random crop.
Only train images receive gradient updates; validation selects the epoch; test
images are never fitted. Full-label YOLO training is separate from audit-limited
PPO/MLP controller learning. No cross-fit YOLO replicas or seed sweep are created.

## Profile before building the longer cache

For the quickest initial comparison, use a bounded 32-token explanation setting:

```bash
export PPO_V2_SMOL_TOKENS=32
bash scripts/runs/ppo_v2_gpu.sh profile
```

This measures 20 original training source images and their four fixed crops;
it is an incomplete profiling cache, not a scientific training result. The
per-stage progress/ETA and `small/cache_profile.json` give a hardware-based
forecast. Keep the same token bound, checkpoints, portfolio and image processing
for the subsequent cache so profiling work resumes. Default without the variable
is96 tokens; changing32→96 requires a new bundle because the tool changes.

Frozen tool families load sequentially to limit GPU memory. The launcher loads
Qwen3-VL-8B **locally on the VM** in4-bit NF4 with FP16 computation, releases it,
then runs the Laya-conditioned cache pass. Its weights remain frozen. This is
runtime loading of the cloud tool, not the deferred Laya/Smol quantization study.
Memory fit and kernel compatibility still require the user's GPU profile; there
is no CPU fallback. FP16 Qwen is available only on a sufficiently larger GPU.
For the first pilot, Qwen has a32-token output bound and original-image calls
only; ROI-specific Qwen calls are masked. `--cloud-views all` builds the expanded
cache in a fresh bundle. This declared restriction limits expensive8B cache work.

The generated `network.json` is explicitly an **idealized zero-added-transport
VM scenario**, not measured drone networking. It enables cloud selection without
inventing a measured bandwidth. Replace it using `PPO_V2_NETWORK_CONFIG` for a
measured/specified network scenario. All comparators use the same configuration.

## Cache train/validation, then run the pilot

```bash
bash scripts/runs/ppo_v2_gpu.sh cache
bash scripts/runs/ppo_v2_gpu.sh pilot
```

The pilot executes, sequentially, a label-free heuristic, a capability-matched
central supervised MLP, central PPO, five independent PPO agents, FedAvg-PPO and
FedProx-PPO. Add `--modes ...,fedavgm,fedadam` for the server-optimizer variants.

## Federation with Flower

Federated modes run in the Flower simulation engine (`flwr` 1.43, Ray backend,
`run_simulation`). Code: `src/fedorch/ppo_v2_flower.py`.

- Each drone is one Flower `NumPyClient` (five virtual SuperNodes). The drone
  receives the global actor, collects PPO missions on its own partition, trains
  its actor and its private critic, and returns the FP16 actor delta.
- The server uses Flower `FedAvg` or `FedProx` (`--flower-strategy`). The
  strategy averages the deltas, weighted by transitions (`--fed-weighting
  sample`) or equally (`equal`). FedProx sends `proximal_mu` to the drones. The
  local PPO loss then adds mu/2 times the squared distance to the received actor.
- `--server-opt momentum` applies server momentum to the averaged delta
  (FedAvgM). `--server-opt adam` applies Adam to it (FedAdam). The default is none.
- `--fraction-fit` samples a share of the five drones in each round (partial
  participation). Flower selects the drones. The summary records them.
- Only the actor delta goes up. Critics, optimizer state, rollouts and images
  stay in `drone-<id>/` folders. Upload size is the Flower `FitRes` protobuf size.
- The drones run one after another on the one GPU (`num_gpus: 1` per client).
- Laptop check without a GPU: `--device cpu --allow-synthetic` on a synthetic
  fixture bundle only. Real bundles still train on CUDA only.
All have S3/A2 capabilities supported by the current cache, the same audit mask,
2/6/15-second budgets and nominal mission counts. It uses seed0, 3 rounds,
64 missions/client/round and validation only. These are early development results,
not the final20-round or multiseed estimates. Source/component counts also vary
with grouped missions/action lengths: report them alongside mission counts.

Supervised mode is explicitly an exploratory observed-action discounted-return
regressor with20% random action exploration. Its labels use the same audit-IPW
error costs and known operational costs as PPO. It is not an exhaustive
counterfactual action oracle, and a stronger audited counterfactual/imitated
selector remains a baseline extension before a strong PPO-superiority claim.

Outputs are in `$FEDORCH_WORK/runs/ppo_v2/pilot_seed0/`: per-method logs, per-round
validation summaries, actors, private rollouts/critics, and fed actor-delta files.
No images/trajectories/critics are part of federation. Local agents are evaluated
separately on the common held-out set; they are not combined into an ensemble.
Held-out drone ownership is not inferred from a fabricated geographic map.

For a quicker validation execution check (clearly incomplete held-out coverage):

```bash
export PPO_V2_OUTPUT="$FEDORCH_WORK/runs/ppo_v2/pilot_seed0_val128"
bash scripts/runs/ppo_v2_gpu.sh pilot --max-eval-items 128
```

Once the path works, run the full validation set and longer training, still seed0:

```bash
uv run --locked --extra gpu-tools python scripts/ppo_v2_compare.py \
  --bundle "$FEDORCH_WORK/runs/ppo_v2/small" \
  --output "$FEDORCH_WORK/runs/ppo_v2/seed0_round20" \
  --device cuda --feedback audit --seed 0 --rounds 20 \
  --episodes-per-client 256 --budgets 2,6,15 --eval-split val \
  --network-config "$FEDORCH_WORK/runs/ppo_v2/network.json"
```

Use fresh output directories. Full-feedback upper bounds use an explicit
`--feedback full` and another output directory; never mix them with5%-audit
results. No CPU fallback is permitted for controller updates or YOLO training.

## Test after policy selection

Cache the test split with identical frozen settings, retaining train/val work:

```bash
bash scripts/runs/ppo_v2_gpu.sh cache-test
uv run --locked --extra gpu-tools python scripts/ppo_v2_train.py \
  --bundle "$FEDORCH_WORK/runs/ppo_v2/small" \
  --output "$FEDORCH_WORK/runs/ppo_v2/fed_pilot_final_test" \
  --evaluate-only "$FEDORCH_WORK/runs/ppo_v2/pilot_seed0/fed/round-02.pt" \
  --device cuda --eval-split test --budgets 2,6,15 \
  --network-config "$FEDORCH_WORK/runs/ppo_v2/network.json"
```

Use the checkpoint selected under your validation-only rule. Evaluate all
comparators' frozen checkpoints using the same test protocol, without additional
training or test-based selection. JSON actor exports include ordered features,
actions and input transform. Their S3 schema differs from the earlier8-feature
Mac demo, so they are not directly importable by that provisional UI.

## Optional ENHANCE and cloud

User amendment adds optional enhancement, disabled in the original A2 pilot:
median3 denoising, bounded unsharp masking and regularized Wiener deblurring
under an explicit Gaussian-kernel assumption. All operate on original source
pixels, retain the original, and invoke frozen Laya afterward. At most one
enhancement per component. Artifact creation/erasure is possible; measure gain
and harm, not just confidence. No learned restoration model is introduced.

Create a fresh small bundle, reuse the SAME pretrained YOLO checkpoint, then run the
explicit cache command with `--enhance`, and train with `--enhance`. Keep other
settings and audit IDs matched. Do not change the portfolio in a populated cache;
the immutable signature prevents stale reuse.

The default launcher already includes cloud Qwen loaded locally. The direct
cache command uses `--cloud-local-checkpoint PATH --cloud-precision nf4
--cloud-max-tokens 32 --cloud-views original` with `CLOUD_QWEN8B` in `--tools`.
Only Qwen3-VL-8B is accepted. Local Qwen computation contains no network transfer;
nominal request/response sizes for a simulated link must not be presented as
actual uploaded network bytes. If a network service is desired later, the
optional `--cloud-url` plus `--cloud-revision` path supports it; set
`PPO_V2_CLOUD_API_KEY` for endpoint authorization. HTTP wall time already includes
transport, so replay must not add it twice. Failed-call traffic can be unknown.

For cloud-enabled controller runs, also pass `--network-config` with a JSON file
containing the observed network configuration, for example `available`, `rtt_s`,
`bandwidth_mbps`, and any explicitly added `delay_s`. Cloud remains masked without
an enabled network configuration. For HTTP timings already containing transport,
the environment does not add a second baseline RTT/upload time; only an explicitly
requested extra delay is added. Do not present invented example rates as measured.

## Interpretation and remaining validation

Cache-replay mission time = measured frozen-tool durations plus measured actor
execution and explicitly configured network additions. It is not a live Jetson
measurement. Warm/cold loading, tool/Laya input conditions and processing are
recorded separately where native APIs expose them. Energy is null when not
measured; no power/energy claim is fabricated.

Federation sends FP16 actor deltas through Flower and enforces <30,000,000 bytes
of cumulative Flower `FitRes` protobuf payload. One drone update is about
120 kB. This is **not a packet-capture measurement of real network wire bytes**. Deployment
transport overhead/retries must be measured before claiming the protocol's
on-wire success criterion. Downlink/cloud/body traffic are separate. The actor
is bounded at100k parameters; no payload is inflated to fill a budget.

The implementation covers the first executable pipeline. Delayed feedback
retraining, stratified audits, clipped/self-normalized IPW comparisons, realistic
time-varying network failures, hardware energy and multiseed/tower-bootstrap
analysis remain explicit follow-up experiments. Present no full E0–E8 completion
claim. Sparse verified labels may make early PPO unstable or inferior to the
matched supervised baseline.

Runtime cannot be measured on the2080Ti from this laptop. Planning ranges for
one GPU, already downloaded data/models: no YOLO training is required;
train/validation cache with5edge views,32-token Smol and original-only32-token
Qwen calls roughly3–16h;
single-seed pilot comparison roughly15–90min after caching. These are unverified
estimates and can be exceeded.96-token explanations/full test caching can take
substantially longer. Use the20-image profile's measured rates to replace these
ranges before scheduling. Additional seeds do not repeat frozen-tool caching.
