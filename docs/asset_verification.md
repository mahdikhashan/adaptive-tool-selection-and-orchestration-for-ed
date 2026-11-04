# Step 1: asset verification (2026-11-04)

What was verified from the laptop (Hugging Face API, PyPI, GitHub, dataset pages), and what the A10 run of
`scripts/step1_verify_assets.py` still has to confirm.

## Decision models (controller candidates)

All confirmed to exist on Hugging Face with open licences. The table the user supplied (from ChatGPT) was mostly right.
Corrections are noted.

| Tier | Checkpoint | Params (HF) | Licence | Architecture / interface | How we fine-tune it | Status |
|---|---|---|---|---|---|---|
| **Small (lead)** | [`sdmlai/nano-jev`](https://huggingface.co/sdmlai/nano-jev) v1.0 | 33,360,385 | MIT | MiniLM-L12-H384 cross-encoder + 1-logit head; **per-option scoring**: `"question: {q} option: {o}"` + context; per-decision temperature in `calibration.json` | Plain `transformers` `AutoModelForSequenceClassification` + PEFT LoRA | **Verified.** Not "C-Tianyu/NanoJev", which is a Qwen3-0.6B variant with no licence |
| **Medium** | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | 421,293,830 | Apache-2.0 | ModernBERT-large (395M) + decision head (2 transformer layers, option-marker scorer, **act/escalate head**); `pip install laya`; ~35–40 ms per question on a T4 (model card) | Official fine-tuning notebook (2×T4); we wrap it with PEFT LoRA | **Verified** |
| Medium (alt.) | [`fastino/GLiNER2.5-Decide`](https://huggingface.co/fastino/GLiNER2.5-Decide) | 486,444,053 | Apache-2.0 | DeBERTa-based decision model | PEFT LoRA | Verified (table said 340M; HF says 486M) |
| **Large** | [`jaredpalmer/kev-4b`](https://huggingface.co/jaredpalmer/kev-4b) | LoRA r16 (33.8M) on frozen `Qwen/Qwen3.5-4B-Base` | Apache-2.0 | Prefill-only causal LM + pointer head over options; one temperature (2.41) | **Already a LoRA adapter**: continue with `kev.train --init_from jaredpalmer/kev-4b` (Kev repo), federating the adapter | **Verified.** Serving needs the Kev repo (fused DeltaNet kernels, ~16 GB GPU) |
| Large / **multimodal** (optional) | [`perplexity-ai/pplx-decider-flash-v1.2`](https://huggingface.co/perplexity-ai/pplx-decider-flash-v1.2) | 4,539,265,536 | Apache-2.0 | Qwen3.5-4B decision model; **accepts images** (up to 1.6 MP); joint option format; source package ships inside the HF repo (not on PyPI) | LoRA on the backbone, via its source package | Verified. Replaces d1 as the multimodal arm |
| Optional | [`LiquidAI/d1-3B`](https://huggingface.co/LiquidAI/d1-3B) | 3,123,483,888 | **LFM 1.0 (other)** | VLM decision model | — | Licence is not Apache/MIT. Keep only if the terms allow research use |
| Training-free baseline | AnyJev / SemIf | — | framework | Option-logit inference on a frozen LLM | none | AnyJev checkpoints exist only at 31B; SemIf only as MLX ports. **Replace with our own option-logit readout on Qwen3.5-4B-Base**, same idea |

Fallback backbones (exist, verified): `microsoft/MiniLM-L12-H384-uncased`, `answerdotai/ModernBERT-large`, `Qwen/Qwen3.5-4B-Base`.

Perception tools (exist): `HuggingFaceTB/SmolVLM-256M-Instruct` (EVLM's onboard VLM), `Qwen/Qwen3-VL-8B-Instruct` (cloud tier).

Lessons from the model cards that change the design:
- nano-jev: *"Accuracy drops by about 15–20 points on unseen datasets, and calibration does not transfer"*, and custom option sets are its weaker mode. **We must fine-tune and re-calibrate** on our decision traces; zero-shot numbers are only a baseline.
- Laya already has an **act/escalate head**. Use it directly for accept / escalate.
- Kev-4B notes *"Option order can change an answer"*. Randomise option order during training and evaluation.

## Data

| Asset | Location | Licence | Verified |
|---|---|---|---|
| **ElecVQA-Bench** | <https://github.com/GhostDino/elecvqa-bench> (MIT code; derived metadata) | MIT + source terms | Cloned, statistics computed (below) |
| **InsPLAD images** | Mendeley Data <https://data.mendeley.com/datasets/5n3fjgvfyz/1>; repo <https://github.com/andreluizbvs/InsPLAD> | "other" (see the repo) | **Manual download needed**; the A10 run checks every image path |
| **4G/LTE traces** (van der Hooft et al.) | <https://users.ugent.be/~jvdrhoof/dataset-4g/logs/logs_all.zip> | public dataset | Downloaded, 40 traces parsed |

### ElecVQA-Bench tower14 split (`results/step1/data.json`)

| Split | Items | Towers | Items per tower (min / median / max) |
|---|---|---|---|
| train | 25,499 | 18 | 16 / 742 / 6,449 |
| val | 6,672 | 10 | 13 / 221 / 3,191 |
| test | 7,057 | 14 | 11 / 99 / 2,054 |

Labels (train): good 22,377 · rust 1,009 · defect (binary items) 1,002 · corrosion 468 · missing-cap 351 · nest 273 · torn-up 17 · peeling-paint 2.

What this means for the design:
- **Prompts are in Chinese.** Our tools are English-tuned, so we rebuild English prompts from the item metadata (asset, options, label) in `src/fedorch/labels.py`. The labels parse cleanly, with no unknowns.
- **42 towers with very unequal sizes and rare defects on only some towers** (nest/missing-cap on 10 of 18 train towers): natural non-IID clients. Plan: **clients = groups of the 28 train+val towers** (about 8–10 clients, each split locally by `parent_id`); the **14 test towers are an unseen fleet** for global testing and the new-drone experiment (E6).
- **torn-up (17) and peeling-paint (2) are too rare to learn.** Merge them into "other-defect" or exclude them, and report it.
- About 88% of items are "good". Use class-weighted loss and report per-class recall (ElecVQA's warning about macro averages over long tails applies).

### LTE traces (`results/step1/trace.json`)

40 traces (bicycle, bus, car, foot, train, tram), ~5.0 h in total; mean throughput 31.4 Mbps (lowest trace mean 14.1 Mbps).
Only about 1% of seconds fall below 2 Mbps (median trace). **The raw traces are too benign for the low-bandwidth
regime**, so per-client regimes need **scaling and injected outages**. This is stated as a constructed condition in the paper.

## Still to confirm on the A10 (`--all`)

1. GPU, CUDA and package versions (`env.json`).
2. nano-jev and Laya load and run; latency of one decision on the A10 (`smoke.json`).
3. Every ElecVQA image path resolves in the downloaded InsPLAD (`data.json`, `images_found`).
4. Large-tier downloads succeed (`--download-large`), and the Kev repo serves Kev-4B (separate step).
