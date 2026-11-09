# Local Mac M1 tool-selection demo

This is an inference-only, provisional UI for trying a sample image and a separate
controller. It does not train PPO, perform federation, quantize, fine-tune Laya,
or download checkpoints. Federated controller/PPO clients are drones themselves;
the VM can emulate their GPU training. This Mac demo runs inference only. The proposed state/action
schema below awaits the user's selection and is not a final experimental contract.

From the repository root, using its existing locked environment:

```sh
uv run --locked python scripts/agentic_demo.py
```

In this sandbox the regular `uv run --locked` check was blocked by cache access
permissions. The tests and help command were verified with the already-provisioned
`.venv/bin/python`, without changing dependencies or the lock. If needed, launch
with `.venv/bin/python scripts/agentic_demo.py` in that existing environment.

Open `http://127.0.0.1:8765`. Upload a still image (up to 8 MiB, 20M pixels),
optionally choose a controller JSON file, select the mode and run. The browser
shows events as each tool finishes, updates the image after cropping, and can
save the full trace JSON. Stop the server with Ctrl-C. It binds only to loopback;
use the exact printed address. One mission executes at a time.

The default is **DEMO**, with a **rule controller, not trained PPO**, and fixed
illustrative model outputs. Real crop/resize processing happens in both modes.
No fake model latency is generated: demo records are measured Python/processing
wall time and explicitly unsuitable as model inference measurements. DEMO does
not invoke model adapters even when checkpoint paths are supplied.

For **REAL**, supply existing local checkpoint directories, then choose REAL in
the browser:

```sh
uv run --locked python scripts/agentic_demo.py \
  --device mps \
  --laya-checkpoint /absolute/path/to/frozen-laya-checkpoint \
  --smol-checkpoint /absolute/path/to/complete-finetuned-smolvlm-checkpoint
```

`--device cpu` is available. No cloud endpoint is implemented or required. The
existing environment must already contain dependencies; `uv run --locked` may
need dependency installation if it has not been provisioned. `--help` itself
imports only stdlib modules, with no torch/Pillow/model initialization:

```sh
python3 scripts/agentic_demo.py --help
```

The Laya adapter uses `laya.vlm.load_vlm(path, device=..., dtype="fp32")` and
`agent.predict({"image": PIL_image, "explanation_context": explanation}, questions)`.
The installed API serializes the non-image state fields as context text. Both
the defect and judgeability typed questions are retained in raw outputs. Only
defect probability currently enters the controller state; this is a demo choice,
not a validated detector/calibration or final mission schema. Laya stays frozen
with inference mode, evaluation mode and disabled gradients. HF/Transformers
are set offline before model imports. A missing backbone/cache/checkpoint fails
visibly; there is no automatic network fallback or fabricated real result.

The built-in Smol adapter accepts a **complete local fine-tuned Transformers
checkpoint**, or an existing **local base plus PEFT/LoRA adapter**:

```sh
uv run --locked python scripts/agentic_demo.py --device mps \
  --laya-checkpoint /absolute/path/to/frozen-laya-checkpoint \
  --smol-checkpoint /absolute/path/to/local-smolvlm-base \
  --smol-adapter /absolute/path/to/step2-smolvlm-lora
```

`--smol-adapter` requires `--smol-checkpoint`. The PEFT wrapper loads with
`local_files_only=True`, remains frozen in evaluation/inference mode, and is not
merged or trained. Both directories must already exist. Smol generates at most
96 tokens and passes the explanation into a fresh Laya call. Existing
letter-scoring fine-tuning does not establish explanation-generation quality or
faithfulness; the UI marks explanations as unvalidated. For a different local
runtime, use the plugin interface below; do not merge or train models here.

## Actions and guards

Exact action order: `laya`, `crop_zoom`, `smol_explain`, `reacquire_simulated`,
`accept`, `flag`, `abstain`.

- Laya evaluates the current image with any current explanation context.
- Crop/zoom takes the center half in each dimension and resizes to original size.
  This is image processing with no new visual evidence, not object detection.
- Smol explanation invalidates the earlier evaluation until Laya runs again.
- Reacquire resets the uploaded original image, explicitly **simulation only**.
  No camera, new frame, or video is claimed.
- Accept/flag require a successful Laya evaluation of the current image/context.
  They are controller choices, not ground truth. Abstain is always legal.

Structural action masking removes unavailable adapters, already-used processing
steps, failed tools and invalid terminal choices. The JSON actor takes the highest
logit among legal actions; masked actions never execute. The rule controller
tries Laya, crop, Laya, Smol explanation, Laya, then accepts/flags at probability
0.5. If a preferred adapter is missing it abstains; it does not substitute a fake.
An actor can choose different paths. At 12 controller steps the mission abstains.

The budget is a **soft wall-clock guard**, not a real-time guarantee. Remaining
budget is checked before deciding and again after controller inference. Expiry
forces abstention. An in-flight model invocation cannot be interrupted, and its
actual overrun is reported. No unmeasured latency estimate is used to promise
feasibility. Structural/deadline guards remain enabled for every controller;
mask ablation belongs in later experiments, not this UI.

## Safe controller import

Only JSON dense actor weights are accepted. No uploaded Python, pickle, torch
checkpoint or executable deserialization is used. Uploaded JSON never changes
adapters. This supports either a separately trained MLP or a dense PPO actor
export if it uses exactly this observation/normalization/action contract. It is
not an importer for arbitrary PPO frameworks or the prior project's feature
schema. Training provenance is an unverified label, always displayed as such;
importing a file does not prove PPO training or superiority over the MLP.

Exact ordered feature vector (`fedorch-demo-v1`):

| Feature | Definition |
| --- | --- |
| remaining_fraction | max(0, remaining wall seconds / requested seconds) |
| steps_fraction | zero-based controller step / 12 |
| has_evaluation | 1 when Laya evaluated current image/context successfully |
| defect_probability | last successful current-image Laya probability, else 0.5 |
| has_explanation | 1 when current image has explanation context |
| is_cropped | 1 after center crop, cleared on simulated reacquire |
| reacquired | 1 after simulated reacquire, never cleared |
| last_failed | 1 when previous tool failed, cleared at next tool start |

JSON must have exactly these fields:

```json
{
  "schema": "fedorch-demo-v1",
  "features": ["remaining_fraction", "steps_fraction", "has_evaluation", "defect_probability", "has_explanation", "is_cropped", "reacquired", "last_failed"],
  "actions": ["laya", "crop_zoom", "smol_explain", "reacquire_simulated", "accept", "flag", "abstain"],
  "mean": [0, 0, 0, 0, 0, 0, 0, 0],
  "scale": [1, 1, 1, 1, 1, 1, 1, 1],
  "provenance": "replace with training/export provenance",
  "layers": []
}
```

The empty layers above is deliberately a schema illustration, **not a usable
trained policy**. Supply 1–4 layers with exactly `weight`, `bias`, `activation`.
Weights are **output × input** arrays; biases have output length; widths are
1–128. Hidden activation must be `relu` or `tanh`; final activation must be
`linear`, with exactly 7 output logits. Input is normalized as `(x - mean)/scale`;
all 8 means/scales are required, scales must be positive, and all values must be
finite numeric values with absolute magnitude at most 1e6. Feature/action order,
version, dimensions and fields are checked strictly. Inference is deterministic
argmax over allowed logits (no softmax sampling). The browser caps policy files
at 1 MiB and requests at 12 MiB on the server.

## Stable trusted adapter interface

Launch `--adapter your_installed_module:factory`. The server calls
`factory(argparse_namespace)` once. The returned object must expose:

```python
device = "mps"  # or "cpu"; controls outer synchronization

def available(action: str) -> bool:
    # Called only for "laya" and "smol_explain".
    ...

def invoke(action: str, image, context: str) -> dict:
    # image: current RGB Pillow image; context: prior explanation or empty string
    # laya: {"defect_probability": float_in_0_1, "raw": optional_json_result}
    # smol_explain: {"explanation": nonempty_string_up_to_10000_characters}
    # Optional "timing" fields must identify how they were measured.
    ...
```

Plugins are trusted startup code, never browser uploads. Keep checkpoints frozen,
use inference-only calls, do not download models, and raise on unsupported or
failed calls. The built-in adapter already supports local Smol PEFT/LoRA without merging;
this interface supports alternative runtimes. No cloud plugin ships here. If one is added,
the only allowed cloud model for this project is **Qwen3-VL-8B**, with explicit
operator configuration and image-transfer disclosure.

## Timing and checks

Each controller decision and each Laya invocation is a separate trace entry with
step number and elapsed wall time. Tool wall time includes adapter startup, and
built-in REAL adapters additionally report `load_or_lookup_ms` and Laya
`predict_ms` or Smol `preprocessing_ms`/`generate_ms`. The Laya predict metric
includes its internal image preprocessing; it is not pure GPU kernel time.
MPS/CUDA synchronization brackets the measured work when available. First-use
loading is identified separately; the demo does not prewarm models. A summary
reports controller, tool, residual processing and total wall time, including
image decoding and trace emission. Crop processing is part of tool time.
No ground-truth accuracy, F1 or quantization comparison is computed here.

```sh
uv run --locked python -m unittest discover -s tests -p test_agentic_demo.py -v
```

Tests cover strict policy import/masking, image/trace flow, the local HTTP upload,
missing/failed adapters, explanation handoff and the deadline guard using stubs.
Local PEFT loading, freezing and offline flags are checked with mocks; no heavy
checkpoint is loaded by tests. Actual Mac M1 model compatibility,
checkpoint availability, explanation fidelity and latency must be validated with
the intended local artifacts; this implementation makes no measured-model claim.
