import argparse
import json
import os
import time

import numpy as np
import torch

from fedorch.common import results_dir, write_json
from fedorch.route_env import Network
from fedorch.select_env import FIELDS, episode, parse, render, setup
from fedorch.tools_v3 import default_dtype, models_dir, selector_tag


def macro_f1(res, field):
    labels = sorted({r["truth"][field] for r in res})
    said = [r["report"][field] if r["report"] else None for r in res]
    scores = []
    for c in labels:
        tp = sum(s == c and r["truth"][field] == c for s, r in zip(said, res))
        fp = sum(s == c and r["truth"][field] != c for s, r in zip(said, res))
        fn = sum(s != c and r["truth"][field] == c for s, r in zip(said, res))
        scores.append(2 * tp / max(1, 2 * tp + fp + fn))
    return round(float(np.mean(scores)), 4)


def summarize(res):
    defect = [r for r in res if r["truth"]["defect"] == "yes"]
    good = [r for r in res if r["truth"]["defect"] == "no"]
    tools = sorted({t for r in res for t in r["called"]})
    return {**{f"{f}_accuracy": round(float(np.mean([f in r["right"] for r in res])), 4) for f in FIELDS},
            "all_three_correct": round(float(np.mean([len(r["right"]) == len(FIELDS) for r in res])), 4),
            "component_macro_f1": macro_f1(res, "component"), "condition_macro_f1": macro_f1(res, "condition"),
            "defect_recall": round(float(np.mean(["defect" in r["right"] for r in defect])), 4) if defect else None,
            "good_recall": round(float(np.mean(["defect" in r["right"] for r in good])), 4) if good else None,
            "no_valid_report": round(float(np.mean([r["report"] is None for r in res])), 4),
            "mean_time_s": round(float(np.mean([r["time"] for r in res])), 3),
            "upload_kb": round(float(np.mean([r["bytes"] for r in res])) / 1000, 2),
            "tool_steps": round(float(np.mean([len(r["called"]) for r in res])), 2),
            "calls": {t: round(float(np.mean([t in r["called"] for r in res])), 3) for t in tools}}


def run(env, rows, budgets, choose_batch, seed):
    trace = Network("weak", 10_000 + seed).trace
    rng = np.random.default_rng(seed)
    out = {}
    for budget in budgets:
        eps = [episode(env, r, budget, trace[rng.integers(len(trace))]) for r in rows]
        while True:
            active = [e for e in eps if not e.done]
            if not active:
                break
            actions = [e.candidates() for e in active]
            for e, choice in zip(active, choose_batch(active, actions)):
                e.step(choice)
        out[str(budget)] = summarize([e.outcome() for e in eps])
    return out


def qwen_chooser(model_name, adapter, batch_size, dtype, think=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def choose(eps, actions):
        out = []
        texts = [render(tokenizer, e, a, think) for e, a in zip(eps, actions)]
        for k in range(0, len(texts), batch_size):
            batch = tokenizer(texts[k:k + batch_size], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=384 if think else 96, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [parse(d, a) for d, a in zip(decoded, actions[k:k + batch_size])]
        return out
    return choose


def main():
    parser = argparse.ArgumentParser(description="Compare the Qwen inspection agent without training (base), after SFT, and after SFT + GRPO. "
                                                 "Accuracy of its own reports against the true labels on test images.")
    parser.add_argument("methods", nargs="*", default=["base", "sft", "grpo"])
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[4.0, 10.0, 30.0, float("inf")])
    parser.add_argument("--images", type=int, default=1500)
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-0.6B"))
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--batch-size", type=int, default=None, help="default 32 for 0.6B, 12 above")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = setup()
    rng = np.random.default_rng(args.seed)
    rows = [env["test"][k] for k in rng.permutation(len(env["test"]))[: args.images]]
    tag, dtype = selector_tag(args.model), args.dtype or default_dtype(args.model)
    batch_size = args.batch_size or (32 if "0.6b" in args.model.lower() else 12)
    path = results_dir("tools_v3") / "select" / "eval.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"images": len(rows), "budgets": args.budgets, "tools": list(env["tools"])})
    report.setdefault("methods", {})
    adapters = {"base": None, "sft": str(models_dir() / f"{tag}_sft"), "grpo": str(models_dir() / f"{tag}_grpo"),
                "base_think": None, "sft_think": str(models_dir() / f"{tag}_sft_think"),
                "grpo_think": str(models_dir() / f"{tag}_grpo_think")}
    for method in args.methods:
        name, start = f"{tag} {method}", time.time()
        try:
            report["methods"][name] = run(env, rows, args.budgets, qwen_chooser(args.model, adapters[method], batch_size, dtype, method.endswith("_think")), args.seed)
            line = {b: (v["component_accuracy"], v["defect_accuracy"], v["condition_accuracy"], v["all_three_correct"], v["mean_time_s"])
                    for b, v in report["methods"][name].items()}
        except Exception as error:
            report["methods"][name] = {"error": repr(error)}
            line = report["methods"][name]
        print(f"[eval] {name} ({(time.time() - start) / 60:.1f} min): (component, defect, condition, all three, time s)", json.dumps(line), flush=True)
        write_json(path, report)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
