import argparse
import json
import time

import numpy as np
import torch

from fedorch.common import results_dir, write_json
from fedorch.route_env import Network
from fedorch.select_env import COMPONENTS, CONDITION_OF, episode, parse, render, setup
from fedorch.tools_v3 import models_dir


def teacher_now(ep, actions):
    return ep.teacher_report()


def laya_own(ep, actions):
    if "laya" in actions:
        return "laya"
    out = ep.r["out"].get("laya")
    if "laya" not in ep.called or out is None:
        return ep.teacher_report()
    condition = max(out["condition"], key=out["condition"].get)
    return {"component": max(out["component"], key=out["component"].get), "defect": "yes" if out["defect"] >= 0.5 else "no",
            "condition": CONDITION_OF[condition]}


def all_tools(ep, actions):
    return next((a for a in actions if a != "report"), ep.teacher_report())


def oracle(ep, actions):
    values = ep.values(actions)
    best = min(actions, key=values.get)
    return ep.teacher_report() if best == "report" else best


def rule(fn):
    return lambda eps, actions: [fn(e, a) for e, a in zip(eps, actions)]


def summarize(res):
    truths = [r["true_condition"] for r in res]
    f1 = []
    for c in sorted(set(truths)):
        tp = sum(r["said_condition"] == c and r["true_condition"] == c for r in res)
        fp = sum(r["said_condition"] == c and r["true_condition"] != c for r in res)
        fn = sum(r["said_condition"] != c and r["true_condition"] == c for r in res)
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    said = np.array([r["defect_said"] for r in res])
    true = np.array([r["defect_true"] for r in res])
    tools = sorted({t for r in res for t in r["called"]})
    return {"loss_per_1000": round(1000 * float(np.mean([r["loss"] for r in res])), 1),
            "component_accuracy": round(float(np.mean([r["component_ok"] for r in res])), 4),
            "condition_accuracy": round(float(np.mean([r["condition_ok"] for r in res])), 4),
            "condition_macro_f1": round(float(np.mean(f1)), 4),
            "defect_recall": round(float(said[true].mean()), 4) if true.any() else None,
            "defect_false_alarm": round(float(said[~true].mean()), 4),
            "defect_risk_per_1000": round(1000 * float(np.mean([r["defect_risk"] for r in res])), 1),
            "all_three_correct": round(float(np.mean([r["component_ok"] and r["condition_ok"] and r["defect_said"] == r["defect_true"] for r in res])), 4),
            "invalid_or_missing_report": round(float(np.mean([r["report"] is None for r in res])), 4),
            "mean_time_s": round(float(np.mean([r["time"] for r in res])), 3),
            "upload_kb": round(float(np.mean([r["bytes"] for r in res])) / 1000, 2),
            "tool_steps": round(float(np.mean([len(r["called"]) - 1 for r in res])), 2),
            "calls": {t: round(float(np.mean([t in r["called"] for r in res])), 3) for t in tools}}


def run(env, rows, budgets, choose_batch, seed):
    trace = Network("weak", 10_000 + seed).trace
    rng = np.random.default_rng(seed)
    out = {}
    for budget in budgets:
        eps = [episode(env, r, budget, trace[rng.integers(len(trace))], train=False) for r in rows]
        while True:
            active = [e for e in eps if not e.done]
            if not active:
                break
            actions = [e.candidates() for e in active]
            for e, choice in zip(active, choose_batch(active, actions)):
                e.step(choice)
        out[str(budget)] = summarize([e.outcome() for e in eps])
    return out


def qwen_chooser(model_name, adapter, batch_size):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def choose(eps, actions):
        out = []
        texts = [render(tokenizer, e, a) for e, a in zip(eps, actions)]
        for k in range(0, len(texts), batch_size):
            batch = tokenizer(texts[k:k + batch_size], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=64, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [parse(d, a) for d, a in zip(decoded, actions[k:k + batch_size])]
        return out
    return choose


def main():
    parser = argparse.ArgumentParser(description="Evaluate inspection agents on test images against the true labels: rules with the teacher report, "
                                                 "laya alone, the oracle, and Qwen3-0.6B base/SFT/GRPO.")
    parser.add_argument("methods", nargs="*", default=["teacher_resnet", "laya_own", "all_tools", "oracle", "qwen_base", "qwen_sft", "qwen_grpo"])
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.75, 1.5, 3.0])
    parser.add_argument("--images", type=int, default=1500)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = setup()
    rng = np.random.default_rng(args.seed)
    rows = [env["test"][k] for k in rng.permutation(len(env["test"]))[: args.images]]
    path = results_dir("tools_v3") / "select" / "eval.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"images": len(rows), "budgets": args.budgets, "tools": list(env["tools"]), "components": COMPONENTS})
    report.setdefault("methods", {})
    makers = {"teacher_resnet": lambda: rule(teacher_now), "laya_own": lambda: rule(laya_own), "all_tools": lambda: rule(all_tools),
              "oracle": lambda: rule(oracle),
              "qwen_base": lambda: qwen_chooser(args.model, None, args.batch_size),
              "qwen_sft": lambda: qwen_chooser(args.model, str(models_dir() / "qwen_sft"), args.batch_size),
              "qwen_grpo": lambda: qwen_chooser(args.model, str(models_dir() / "qwen_grpo"), args.batch_size)}
    for name in args.methods:
        start = time.time()
        try:
            report["methods"][name] = run(env, rows, args.budgets, makers[name](), args.seed)
            line = {b: (v["loss_per_1000"], v["component_accuracy"], v["condition_accuracy"], v["defect_risk_per_1000"], v["mean_time_s"])
                    for b, v in report["methods"][name].items()}
        except Exception as error:
            report["methods"][name] = {"error": repr(error)}
            line = report["methods"][name]
        print(f"[eval] {name} ({(time.time() - start) / 60:.1f} min): (loss, component acc, condition acc, defect risk, time)", json.dumps(line), flush=True)
        write_json(path, report)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
