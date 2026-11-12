import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from tv3_select_eval import summarize

from fedorch.common import results_dir, write_json
from fedorch.episode_play import play
from fedorch.route_env import Network
from fedorch.select_env import episode, setup
from fedorch.tools_v3 import default_dtype, excluded_tools, models_dir


def load(model_name, adapter, dtype):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=getattr(torch, dtype))
    if adapter:
        model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
    return model.to("cuda" if torch.cuda.is_available() else "cpu").eval()


def main():
    parser = argparse.ArgumentParser(description="Evaluate agents that play the whole inspection of an image as one conversation "
                                                 "(the same format as GRPO on full episodes). Greedy decoding, accuracy of the report against the true labels.")
    parser.add_argument("methods", nargs="*", default=["sft_think", "grpo_think", "grpo_episode"])
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-1.7B"))
    parser.add_argument("--name", default="newtools", help="suffix of the full-episode adapter")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[4.0, 10.0, 30.0, float("inf")])
    parser.add_argument("--images", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-turn-tokens", type=int, default=384)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tag = args.model.split("/")[-1].lower()
    env = setup()
    rng = np.random.default_rng(args.seed)
    rows = [env["test"][k] for k in rng.permutation(len(env["test"]))[: args.images]]
    trace = Network("weak", 10_000 + args.seed).trace
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    adapters = {"sft_think": f"{tag}_sft_think", "grpo_think": f"{tag}_grpo_think", "grpo_episode": f"{tag}_grpo_episode_{args.name}",
                "sft": f"{tag}_sft", "grpo": f"{tag}_grpo", "base_think": None}
    path = results_dir("tools_v3") / "select" / (args.out or f"eval_episode_{tag}_{args.name}.json")
    report = json.loads(path.read_text()) if path.exists() else {}
    report.update({"images": len(rows), "budgets": args.budgets, "tools": list(env["tools"]), "excluded": sorted(excluded_tools()), "format": "multi-turn"})
    report.setdefault("methods", {})
    for method in args.methods:
        name, start = f"{tag} {method}", time.time()
        think = "think" in method or method == "grpo_episode"
        try:
            model = load(args.model, str(models_dir() / adapters[method]) if adapters[method] else None, args.dtype or default_dtype(args.model))
            res = {}
            for budget in args.budgets:
                b_start, rng = time.time(), np.random.default_rng(args.seed)
                outcomes = []
                for k in range(0, len(rows), args.batch_size):
                    eps = [episode(env, r, budget, trace[rng.integers(len(trace))]) for r in rows[k:k + args.batch_size]]
                    outcomes += [r["ep"].outcome() for r in play(model, tokenizer, eps, think, False, args.max_turn_tokens)]
                    if (k // args.batch_size) % 10 == 0:
                        print(f"[progress] {name} budget {budget:g}: {len(outcomes)}/{len(rows)} images, {(time.time() - b_start) / 60:.1f} min", flush=True)
                res[str(budget)] = summarize(outcomes)
                v = res[str(budget)]
                print(f"[budget] {name} {budget:g} s ({(time.time() - b_start) / 60:.1f} min): component {v['component_accuracy']}, defect {v['defect_accuracy']}, "
                      f"condition {v['condition_accuracy']}, all three {v['all_three_correct']}, time {v['mean_time_s']} s, tool steps {v['tool_steps']}", flush=True)
                report["methods"][name] = dict(res)
                write_json(path, report)
            line = {b: (v["component_accuracy"], v["defect_accuracy"], v["condition_accuracy"], v["all_three_correct"], v["mean_time_s"]) for b, v in res.items()}
        except Exception as error:
            report["methods"][name] = {"error": repr(error)}
            line = report["methods"][name]
        print(f"[eval] {name} ({(time.time() - start) / 60:.1f} min): (component, defect, condition, all three, time s)", json.dumps(line), flush=True)
        write_json(path, report)
        model = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
