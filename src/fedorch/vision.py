import statistics
import time

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


def load(device):
    return load_vlm(CHECKPOINT, revision=REVISION, device=device, dtype="fp32")


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
    return stats(times[warmup:])
