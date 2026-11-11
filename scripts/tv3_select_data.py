import argparse
import json

import numpy as np

from fedorch.common import results_dir
from fedorch.route_env import Network
from transformers import AutoTokenizer

from fedorch.select_env import answer, episode, render, setup


def main():
    parser = argparse.ArgumentParser(description="Gold actions from the tool cache: SFT data and GRPO prompts for the Qwen3-0.6B tool selector (TRL).")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.75, 1.5, 3.0])
    parser.add_argument("--explore", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--grpo-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="tokenizer for the chat template")
    parser.add_argument("--answer-keep", type=float, default=0.5, help="share of steps whose gold is a final answer kept for SFT")
    args = parser.parse_args()
    env = setup()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rng = np.random.default_rng(args.seed)
    trace = Network("weak", args.seed).trace
    out = results_dir("tools_v3") / "select"
    out.mkdir(exist_ok=True)
    sft, grpo = [], []
    for n, item in enumerate(env["val"]):
        for budget in args.budgets:
            ep = episode(env, item, budget, trace[rng.integers(len(trace))], train=True)
            for step in range(args.max_steps):
                actions = ep.candidates()
                values = ep.values(actions)
                gold = min(actions, key=values.get)
                text = render(tokenizer, ep, actions)
                if not gold.startswith("answer_") or rng.random() < args.answer_keep:
                    sft.append({"prompt": text, "completion": answer(gold)})
                if rng.random() < args.grpo_share:
                    grpo.append({"prompt": text, "values": json.dumps(values), "true_values": json.dumps(ep.values(actions, true_answers=True)), "gold": gold})
                move = actions[rng.integers(len(actions))] if rng.random() < args.explore else gold
                ep.step(move)
                if ep.done:
                    break
        if (n + 1) % 1000 == 0:
            print(f"[select data] {n + 1}/{len(env['val'])} val items, {len(sft)} steps", flush=True)
    for name, rows in (("sft_qwen", sft), ("grpo_qwen", grpo)):
        with (out / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    golds = [json.loads(r["completion"])["ids"][0] for r in sft]
    print(f"[select data] SFT steps {len(sft)}, GRPO prompts {len(grpo)}", flush=True)
    print("[select data] gold action counts:", json.dumps({a: golds.count(a) for a in sorted(set(golds), key=golds.count, reverse=True)}), flush=True)


if __name__ == "__main__":
    main()
