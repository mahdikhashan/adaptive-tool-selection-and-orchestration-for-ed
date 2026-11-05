import contextlib
import json
import random
import statistics
import time

import torch

from laya.vlm import load_vlm

from fedorch.items import INSPLAD

CHECKPOINT = "thaitea/laya-vision"
REVISION = "f2fe3c12cb6d04c59d8a190250bf3fb40fc828dc"

COMPONENT = {
    "damper-stockbridge": "stockbridge damper",
    "damper-preformed": "preformed damper",
    "glass-insulator": "glass insulator",
    "polymer-insulator": "polymer insulator",
    "glass-insulator-big-shackle": "shackle",
    "glass-insulator-small-shackle": "shackle",
    "glass-insulator-tower-shackle": "shackle",
    "lightning-rod-shackle": "shackle",
    "polymer-insulator-lower-shackle": "shackle",
    "polymer-insulator-tower-shackle": "shackle",
    "polymer-insulator-upper-shackle": "shackle",
    "lightning-rod-suspension": "lightning rod suspension",
    "vari-grip": "vari-grip clamp",
    "yoke": "yoke",
    "yoke-suspension": "yoke suspension",
    "spacer": "spacer",
    "plate": "plate",
}
CONDITION = {
    "good": "good, no defect",
    "rust": "rust",
    "corrosion": "corrosion",
    "missing-cap": "missing cap",
    "nest": "bird nest",
    "other-defect": "other defect (torn part or peeling paint)",
}
FOLDER = {"good": "good", "normal": "good", "rust": "rust", "corrosão": "corrosion", "missing-cap": "missing-cap",
          "missingcap": "missing-cap", "bird-nest": "nest", "nest": "nest", "peeling-paint": "other-defect",
          "torned-up": "other-defect"}

QUESTIONS = {
    "component": {"type": "choice", "instructions": "Which power line component is in this image?",
                  "criteria": list(dict.fromkeys(COMPONENT.values()))},
    "defect": {"type": "noul", "instructions": "Does this power line component have a defect?"},
    "condition": {"type": "choice", "instructions": "Which condition does this power line component have?",
                  "criteria": list(CONDITION.values())},
    "judgeable": {"type": "noul", "instructions": "Is the image clear enough to judge the condition of the component?"},
}


def load(device, path=None):
    if path:
        return load_vlm(str(path), device=device, dtype="fp32")
    return load_vlm(CHECKPOINT, revision=REVISION, device=device, dtype="fp32")


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"


def peak_mb():
    return round(torch.cuda.max_memory_allocated() / 1e6, 1) if torch.cuda.is_available() else 0.0


def images(items):
    out = {}
    for item in items:
        condition = FOLDER[item["image"].split("/")[-2]]
        out.setdefault(item["image"], {"image": item["image"], "split": item["split"], "tower": item["tower"],
                                       "asset": item["asset"], "component": COMPONENT[item["asset"]],
                                       "condition": CONDITION[condition], "defect": condition != "good"})
    return list(out.values())


def compact(answer):
    return answer["noul"] if answer["type"] == "noul" else answer["probabilities"]


def predict(agent, image, questions=QUESTIONS):
    start = time.perf_counter()
    result = agent.predict({"image": str(INSPLAD / image)}, questions)
    ms = 1000 * (time.perf_counter() - start)
    answers = result["answers"]
    return {q: compact(a) for q, a in answers.items()}, ms, sorted(q for q, a in answers.items() if "truncated" in a)


def stats(values):
    values = sorted(values)
    pick = lambda q: values[min(len(values) - 1, int(q * len(values)))]
    return {"frames": len(values), "mean": round(statistics.mean(values), 2), "median": round(pick(0.5), 2),
            "p90": round(pick(0.9), 2), "p95": round(pick(0.95), 2), "max": round(values[-1], 2)}


def latency(agent, rows, questions, warmup=10):
    times = [predict(agent, row["image"], questions)[1] for row in rows]
    return stats(times[min(warmup, len(times) // 2):])


def timing(agent, rows, frames):
    sample = random.Random(0).sample(rows, min(len(rows), frames + 10))
    return {"all_questions_ms": latency(agent, sample, QUESTIONS),
            "condition_only_ms": latency(agent, sample, {"condition": QUESTIONS["condition"]})}


def labelled(row):
    component = QUESTIONS["component"]["criteria"].index(row["component"])
    condition = QUESTIONS["condition"]["criteria"].index(row["condition"])
    image = str(INSPLAD / row["image"])
    return [{"id": f"{row['image']}#{q}", "image": image, "question": QUESTIONS[q], "label": label}
            for q, label in (("component", component), ("defect", int(row["defect"])), ("condition", condition))]


def fp32_on_old_gpus():
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported(including_emulation=False):
        torch.autocast = lambda *args, **kwargs: contextlib.nullcontext()
        return True
    return False


def predict_all(agent, rows, path):
    done = {json.loads(line)["image"] for line in path.open()} if path.exists() else set()
    todo = [r for r in rows if r["image"] not in done]
    print(f"{len(rows)} images, {len(done)} done, {len(todo)} to do", flush=True)
    for row in rows[:10]:
        predict(agent, row["image"])
    start = time.time()
    with path.open("a") as file:
        for n, row in enumerate(todo, 1):
            answers, ms, truncated = predict(agent, row["image"])
            file.write(json.dumps({"image": row["image"], "answers": answers, "ms": round(ms, 2),
                                   "truncated": truncated}, ensure_ascii=False) + "\n")
            if n % 500 == 0:
                file.flush()
                print(f"{n}/{len(todo)} images, {(time.time() - start) / n * 1000:.1f} ms each", flush=True)
