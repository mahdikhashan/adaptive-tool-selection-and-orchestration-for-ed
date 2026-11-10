import argparse
import json
import time

import numpy as np
import torch

from fedorch.common import write_json, results_dir
from fedorch.route_env import Network
from fedorch.select_env import episode, parse, prompt, setup
from fedorch.tools_v3 import models_dir


def rho(ep):
    q = ep.q()
    return min(10 * q, 1 - q)


def resnet_only(ep, actions):
    return "report"


def all_tools(ep, actions):
    return next((a for a in actions if a not in ("report", "defer")), "report")


def cascade_defer(ep, actions):
    if rho(ep) <= 0.5:
        return "report"
    return next((a for a in actions if a not in ("report", "defer")), "defer")


def oracle(ep, actions):
    values = ep.values(actions)
    return min(actions, key=values.get)


def run(env, items, budgets, choose_batch, max_steps, seed):
    trace = Network("weak", 10_000 + seed).trace
    rng = np.random.default_rng(seed)
    out = {}
    for budget in budgets:
        eps = [episode(env, i, budget, trace[rng.integers(len(trace))], train=False) for i in items]
        for _ in range(max_steps):
            active = [e for e in eps if not e.done]
            if not active:
                break
            actions = [e.candidates() for e in active]
            for e, a, choice in zip(active, actions, choose_batch(active, actions)):
                e.step(choice if choice in a else "report")
        for e in eps:
            if not e.done:
                e.done, e.final = True, "report"
        res = [e.outcome() for e in eps]
        tools = sorted({t for r in res for t in r["called"]})
        out[str(budget)] = {"risk_per_1000": round(1000 * float(np.mean([r["risk"] for r in res])), 1),
                            "mean_time_s": round(float(np.mean([r["time"] for r in res])), 3),
                            "deferred": round(float(np.mean([r["deferred"] for r in res])), 4),
                            "upload_kb": round(float(np.mean([r["bytes"] for r in res])) / 1000, 2),
                            "steps": round(float(np.mean([len(r["called"]) - 1 for r in res])), 2),
                            "calls": {t: round(float(np.mean([t in r["called"] for r in res])), 3) for t in tools}}
    return out


def rule(fn):
    return lambda eps, actions: [fn(e, a) for e, a in zip(eps, actions)]


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
        texts = [prompt(e, a) for e, a in zip(eps, actions)]
        for k in range(0, len(texts), batch_size):
            batch = tokenizer(texts[k:k + batch_size], return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**batch, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            decoded = tokenizer.batch_decode(gen[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
            out += [parse(d, a) or "report" for d, a in zip(decoded, actions[k:k + batch_size])]
        return out
    return choose


def main():
    parser = argparse.ArgumentParser(description="Evaluate tool selectors (rules, oracle, Qwen3-0.6B base/SFT/GRPO) on test items.")
    parser.add_argument("methods", nargs="*", default=["resnet_only", "all_tools", "cascade_defer", "oracle",
                                                       "qwen_base", "qwen_sft", "qwen_grpo"])
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.75, 1.5, 3.0])
    parser.add_argument("--items", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    env = setup()
    rng = np.random.default_rng(args.seed)
    items = [env["test"][k] for k in rng.permutation(len(env["test"]))[: args.items]]
    path = results_dir("tools_v3") / "select" / "eval.json"
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"items": len(items), "budgets": args.budgets, "test_prior_em": env["test_prior"], "tools": list(env["tools"])})
    report.setdefault("methods", {})
    makers = {"resnet_only": lambda: rule(resnet_only), "all_tools": lambda: rule(all_tools),
              "cascade_defer": lambda: rule(cascade_defer), "oracle": lambda: rule(oracle),
              "qwen_base": lambda: qwen_chooser(args.model, None, args.batch_size),
              "qwen_sft": lambda: qwen_chooser(args.model, str(models_dir() / "qwen_sft"), args.batch_size),
              "qwen_grpo": lambda: qwen_chooser(args.model, str(models_dir() / "qwen_grpo"), args.batch_size)}
    for name in args.methods:
        start = time.time()
        try:
            chooser = makers[name]()
            report["methods"][name] = run(env, items, args.budgets, chooser, args.max_steps, args.seed)
        except Exception as error:
            report["methods"][name] = {"error": repr(error)}
        print(f"[eval] {name} ({(time.time() - start) / 60:.1f} min):",
              json.dumps({b: (v["risk_per_1000"], v["mean_time_s"], v["deferred"]) for b, v in report["methods"][name].items()}
                         if "error" not in report["methods"][name] else report["methods"][name]), flush=True)
        write_json(path, report)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
