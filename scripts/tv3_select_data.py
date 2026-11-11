import argparse
import json
from collections import Counter

import numpy as np
from transformers import AutoTokenizer

from fedorch.common import results_dir
from fedorch.route_env import Network
from fedorch.select_env import episode, render, report_json, select_json, setup, truth_of


def main():
    parser = argparse.ArgumentParser(description="Gold steps from the tool cache: SFT data and GRPO prompts for the Qwen3-0.6B inspection agent (TRL). "
                                                 "Tool steps: the tool that best improves the evidence-based report. Final step: the report with the true labels.")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.75, 1.5, 3.0])
    parser.add_argument("--explore", type=float, default=0.3)
    parser.add_argument("--grpo-share", type=float, default=0.3)
    parser.add_argument("--report-keep", type=float, default=0.5, help="share of final report steps kept for SFT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="tokenizer for the chat template")
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
        answer = dict(truth, defect="yes" if truth["defect"] else "no")
        for budget in args.budgets:
            ep = episode(env, row, budget, trace[rng.integers(len(trace))], train=True)
            while not ep.done:
                actions = ep.candidates()
                values = ep.values(actions)
                gold = min(actions, key=values.get)
                text = render(tokenizer, ep, actions)
                golds[gold] += 1
                completion = report_json(answer) if gold == "report" else select_json(gold)
                if gold != "report" or rng.random() < args.report_keep:
                    sft.append({"prompt": text, "completion": completion})
                if rng.random() < args.grpo_share:
                    grpo.append({"prompt": text, "values": json.dumps(values), "truth": json.dumps(truth), "gold": gold,
                                 "time_cost": env["lam"] * ep.elapsed / budget})
                move = actions[rng.integers(len(actions))] if rng.random() < args.explore else gold
                ep.step(answer if move == "report" else move)
        if (n + 1) % 1000 == 0:
            print(f"[select data] {n + 1}/{len(env['val'])} val images, {len(sft)} SFT steps", flush=True)
    for name, rows in (("sft_qwen", sft), ("grpo_qwen", grpo)):
        with (out / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"[select data] SFT steps {len(sft)}, GRPO prompts {len(grpo)}", flush=True)
    print("[select data] gold step counts:", json.dumps(dict(golds.most_common())), flush=True)


if __name__ == "__main__":
    main()
