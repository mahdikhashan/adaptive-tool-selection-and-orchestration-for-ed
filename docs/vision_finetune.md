# Laya-vision fine-tuning on part of the dataset (branch `vision-fl`)

Date: 2026-11-05. Status: code complete, not run on the GPU.

## 1. Idea

Fine-tune laya-vision on a small part of the train towers. Then use the same questions and the same evaluation as the zero-shot run (`docs/vision_fl_design.md`, `docs/vision_results.md`). The input is the image only. The pipeline has no routing and no other model.

## 2. Method

The run follows the fine-tune guide of laya-vision (`site-docs/how-to/fine-tune.md`, section 2a, at the commit `9e1e241`).

1. Write the data as JSONL. Each line is one question about one image, with the index of the correct answer (`label`).
2. Load the published checkpoint (`thaitea/laya-vision`, revision `f2fe3c1`). The guide uses the revision `8b318c9`. The revision `f2fe3c1` only adds a line to the README of `8b318c9`, so the weights are the same.
3. Measure the validation questions before the training (`collect_logits`, `metrics_from`).
4. Train with `laya.vlm_train.train`.
5. Fit the temperatures on held-out questions (`fit_temperatures_from`).
6. Measure the validation questions after the training.
7. Save the model (`agent.save`) and load it again with `load_vlm`.
8. Answer the 4 questions for all val and test images with `agent.predict`, as in the zero-shot run.

## 3. Data

| Part | Images | Questions | Source |
|---|---|---|---|
| Train | 1,793 | 5,379 | Train towers. Up to 400 images for each condition, selected at random (seed 0) |
| Temperatures | 100 | 300 | The same selection, held out of the training |
| Validation during training | 600 | 1,800 | Val towers, at random |
| Final evaluation | 11,716 | — | All val and test images |

- The selection has 400 images each of good, rust and corrosion. It has all images of missing cap (383), bird nest (289) and other defect (21). The 100 held-out images come from this selection, so the train counts are a little lower.
- Each image gives 3 questions: `component`, `defect` and `condition`. The `judgeable` question has no label, so the training does not use it. The evaluation still asks it.
- The questions of one image are always in the same part. The train, val and test towers do not share images.
- The training data has more defects than the test data (about 79% against 10%). Thus the accuracy and the risk depend on the threshold. The balanced accuracy, the macro F1 and the AUROC do not depend on the class shares. The threshold for `defect` comes from the val towers, which have the real class shares.

## 4. Training

| Setting | Value | Source |
|---|---|---|
| `freeze` | `last_n`, the head and the last 4 language layers (22.6M parameters) | The guide: "a few thousand questions" |
| Passes | 2.5 over the training questions, about 1,680 steps | The guide: "aim for 2 to 3 passes" |
| Batch | 8 | The guide |
| Learning rate | head 1e-4, layers 2e-5, warm-up 25 steps | The guide |
| Precision | fp32 | See below |

The `train` function of laya-vision uses bf16 autocast on CUDA. The GV100 and the RTX 2080 Ti have no native bf16. On these GPUs, the script stops the autocast, so the training is in fp32. On a GPU with bf16, the script does not change the autocast.

## 5. Outputs

- `results/vision/lastn_pc400/run.json`: settings, loss for each 50 steps, temperatures, validation metrics before and after the training, latency.
- `results/vision/lastn_pc400/predictions.jsonl`: the answers for the val and test images.
- `results/vision/lastn_pc400/{train,calib,val}_images.txt`: the images of each part.
- `results/vision/lastn_pc400/summary.json`: the evaluation, from `scripts/vision_summarize.py lastn_pc400`.
- The model: `$FEDORCH_WORK/models/vision/lastn_pc400`. It is not in git.

## 6. Comparison

The main comparison is the test towers: the fine-tuned model against the zero-shot model, with the same metrics. Values from step 2 on the test items, for comparison only: ResNet-50 classifier 0.961, SmolVLM-256M with LoRA 0.956.
