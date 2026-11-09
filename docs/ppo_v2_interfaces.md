# PPO v2 implementation contract

Internal interface shared by the data exporter, replay environment and learner.
The user's protocol in msg_33ff2172 is the design of record. Single seed (0) first.

Latest architecture amendment: clients are drones themselves, not base stations.
Each drone owns its actor/local critic/optimizer and local trajectories. The VM
emulates five drones and the actor-only coordinator. Qwen3-VL-8B loads on this VM
in a separate staged cloud-tool pass; NF4/fp16 computation is the11GB2080Ti option.
Perception weights remain frozen. No training is executed during implementation.

Subsequent user amendment: optional ENHANCE denoise/sharpen/deblur extension.
Default disabled for original A2 pilot, explicitly enabled for an ablation.
Append ENHANCE_DENOISE, ENHANCE_SHARPEN, ENHANCE_DEBLUR to fixed ACTIONS.
Corresponding records tool=ENHANCE, view_id=5/6/7, respectively, from the original
source only; at most one enhancement per component. Retain original and refresh
frozen Laya assessment, charge processing + Laya. Denoise median3, sharpen bounded
unsharp mask, deblur regularized Wiener with an explicitly assumed Gaussian PSF.
No learned restoration model, no truth or inferred detail claims. `allow_enhance`
environment flag / `--enhance` CLI enables. Exact transform parameters recorded.

## Bundle

A bundle directory contains `manifest.json`, `items.jsonl`, `observations.jsonl`,
`labels.jsonl`, `audits.json`. Files are distinct to enforce label access.

Items: `item_id`, `split` (`train`, `val`, `test`), `tower`, `source_id` (original
image path), `image`, `client` (0..4 for train, -1 otherwise), `task` (`T2`/`T3`).
No labels or correctness in item/observation records. Labels: `item_id`, `label`
(one of good,rust,corrosion,missing_cap,bird_nest,other). Audits is a dict keyed
training item_id: `{audited: bool, propensity: 0.05}`. Select once and persist;
never resample across epochs/rounds. Non-audited labels are not loaded by the
principal training path. Validation/test labels are evaluator-only.

Observations keyed `(item_id, view_id, tool)` with view_id 0 original, 1..4 grid.
Fields: `item_id`, `view_id`, `tool`, `probs` (optional 6-vector), `laya_probs`
(optional 6-vector), `duration_s` (measured processing+tool+downstream Laya; no
network), `laya_ms`, `tool_ms`, `bytes_up`, `bytes_down`, `valid`, optional
`explanation`, `quality` (blur,contrast,brightness,compression numeric list),
`timing` (breakdown), `error`. No silent synthetic result/time substitution.

Tools: INITIAL, YOLO_CLS, SMOL_EXPLAIN, CLOUD_QWEN8B, ZOOM_ROI, IMAGE_QUALITY,
ROI_PROPOSE. INITIAL = mandatory frozen YOLO then Laya conditioned on its output.
YOLO_CLS = YOLO then Laya(current image + YOLO evidence).
SMOL_EXPLAIN = Smol explanation then Laya(current image + that explanation).
CLOUD_QWEN8B = Qwen assessment then Laya(current image + that assessment).
ZOOM_ROI = pixel crop/resize + Laya(image only) on target view. The actor retains
history, but Laya reads ONLY the latest tool's evidence, making these bounded
cache transitions exact and reproducible; no hidden prior outputs are added.
INITIAL exists for original view only. Other tools optional/masked if absent;
manifest lists available portfolio and completeness. Cache original and four
fixed grid crops, no synthetic corruptions/new images/restoration. Quality and
ROI timings measured deterministically during cache export. Probs order is the
six-label order above. All observations are frozen tool outputs.

Manifest: `schema: fedorch-ppo-v2`, `classes`, `clients:5`, `seed:0`, `fraction`,
`audit_rate:0.05`, `available_tools`, `provenance`, `timing_hardware`,
`controller_only_audit_budget:true`, `cache_complete` and counts. Fixtures must
set `synthetic_fixture:true` and can never be reported as experimental results.

## Environment API (environment agent owns)

`Bundle(path, feedback='audit')`: .items list; .observations mapping triple to
record; .training_labels contains audited labels only (or all train for explicit
'full'); .evaluation_labels only accessed by evaluator; .audits; .manifest.

`InspectionEnv(bundle, items, budget_s=6.0, feedback='audit', seed=0,
 network=None, reward=None, mask_mode='hard', policy_version=0)`.
items = list of one or more item dicts; grouped must share tower and have unique
source_id. .reset() -> numeric float observation; .mask() -> bool vector;
.step(action_index, policy_duration_s=0.) -> (observation,reward,done,info).
.action_count, .feature_count, .trace list, .results list, .elapsed_s.
`ACTIONS`, `FEATURES` exported ordered names. Expand ZOOM_ROI into four concrete
actions ZOOM_ROI_1..4; other 7 names unchanged. Cap at 16 nonterminal steps per
component, max 2 zooms, prevent duplicate same-input calls. Fresh component runs
mandatory INITIAL, charged even if it exhausts budget. Exact stored mask used
for collection and optimization. Default network measured/explicit configuration,
no undocumented randomly invented latency. Missing cloud masked if not configured.
Audit reward: observed time/bytes/abstain/late costs on all items, only verified
classification error inverse-weighted on audited items. Never weight known costs.
Unlabeled absence is missing information, not correct. `info` includes duration_s,
terminal, policy_version, observed_error, audited, audit_propensity, trace data.
No unverified labels/correctness are put in observations or training info.
Time discount handled by learner exp(-kappa*duration_s). Evaluator uses results
and held-out labels AFTER completion. Native replay is simulated from measured
cache durations, not claimed wall-clock target deployment or wire networking.

## Learner API (training agent owns)

`src/fedorch/ppo_v2_train.py`, CLI `scripts/ppo_v2_train.py` with --bundle,
--output, --mode fed|central|local|supervised|heuristic, --feedback audit|full,
--seed 0, --rounds (pilot3, main20), --episodes-per-client (pilot64), --device
cpu|cuda, --budgets 2,6,15. No trainer invokes perception models. Separate local
critics, <=100k actor, FP16 serialized deltas <30MB cap. Report serialized bytes
as simulated transport payload, not measured network-wire bytes. Persist delta
files+metadata to count actual file payload; real on-wire claim requires transport.
Fresh rollouts each round; masks/logprobs/version stored; evaluate only val while
training, once test after final policy. Do not select on test. Group15 missions
same tower 2..5, singletons2/6; matched total observations across modes. Export
actor state/checkpoint and JSON; actor supervised baseline learns on audited
decision states with inverse-propensity losses; unverified future tool correctness
is never an action-oracle target. Prefer evaluated terminal cost targets from
audited states as label-dependent targets; document chosen method precisely.

## Exporter (GPU agent owns)

`src/fedorch/ppo_v2_tools.py`, `scripts/ppo_v2_cache.py`,
`scripts/ppo_v2_yolo.py`: native inference, no TensorRT, no quantization jobs.
YOLO classifier training is separate full-label tool pretraining and versioned;
5% applies ONLY controller feedback. Train-only towers; never train on val/test.
Sequential tool loading on 11GB2080Ti. Cloud Qwen8B loads locally on the VM,
NF4/fp16 compute on11GB; FP16 option requires more memory. HTTP endpoint remains
optional. Checkpoints specified, never pretend absent.
Root owns manifest/data generation and launcher/docs/dependencies/integration.
