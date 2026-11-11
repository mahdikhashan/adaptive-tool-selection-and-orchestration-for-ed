import argparse
import json
import os
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from fedorch.common import results_dir
from fedorch.route_env import Network
from fedorch.tools_v3 import run_suffix
from fedorch.select_env import episode, render, report_json, select_json, setup, truth_of


def main():
    parser = argparse.ArgumentParser(description="SFT data and GRPO prompts for the Qwen inspection agent. Tool step target: the tool whose own "
                                                 "answer adds the most correct report fields (fastest on ties). Report step target: the true labels.")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[4.0, 10.0, 30.0, float("inf")])
    parser.add_argument("--explore", type=float, default=0.3)
    parser.add_argument("--grpo-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("TV3_SELECTOR", "Qwen/Qwen3-0.6B"), help="tokenizer for the chat template")
    args = parser.parse_args()
    env = setup()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = np.random.default_rng(args.seed)
    trace = Network("weak", args.seed).trace
    out = results_dir("tools_v3") / "select"
    out.mkdir(exist_ok=True)
    sft, grpo, golds = [], [], Counter()
    for n, row in enumerate(env["val"]):
        truth = truth_of(row)
        for budget in args.budgets:
            ep = episode(env, row, budget, trace[rng.integers(len(trace))])
            while not ep.done:
                actions = ep.candidates()
                gold = ep.gold(actions)
                text = render(tokenizer, ep, actions)
                golds[gold] += 1
                sft.append({"prompt": text, "completion": report_json(truth) if gold == "report" else select_json(gold)})
                if rng.random() < args.grpo_share:
                    grpo.append({"prompt": text, "candidates": json.dumps(actions), "truth": json.dumps(truth), "gold": gold})
                move = actions[rng.integers(len(actions))] if rng.random() < args.explore else gold
                ep.step(truth if move == "report" else move)
        if (n + 1) % 1000 == 0:
            print(f"[select data] {n + 1}/{len(env['val'])} val images, {len(sft)} SFT steps", flush=True)
    for name, rows in (("sft_qwen", sft), ("grpo_qwen", grpo)):
        with (out / f"{name}{run_suffix()}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"[select data] SFT steps {len(sft)}, GRPO prompts {len(grpo)}", flush=True)
    print("[select data] gold step counts:", json.dumps(dict(golds.most_common())), flush=True)


if __name__ == "__main__":
    main()
