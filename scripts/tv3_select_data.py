import argparse
import json
import math

import numpy as np

from fedorch.common import WORK, results_dir
from fedorch.route_env import Network
from fedorch.select_env import episode, prompt, setup
from fedorch.tools_v3 import image_file

LAYA_QUESTION = "Which step should the drone take next?"


def short_option(ep, a):
    if a in ("report", "defer"):
        return a
    return f"{a} ({ep.cost(a):.2f} s)"


def laya_record(ep, actions, values, gold, uid, temperature):
    v = np.array([values[a] for a in actions])
    target = np.exp(-(v - v.min()) / temperature)
    return {"id": uid, "image": str(image_file(ep.i).resolve()), "state_text": ep.describe(),
            "question": {"type": "choice", "instructions": LAYA_QUESTION, "criteria": [short_option(ep, a) for a in actions]},
            "label": actions.index(gold), "target": (target / target.sum()).round(4).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Gold actions from the tool cache: SFT data for Qwen (TRL) and laya, prompts for GRPO.")
    parser.add_argument("--budgets", type=lambda s: [float(x) for x in s.split(",")], default=[0.25, 0.75, 1.5, 3.0])
    parser.add_argument("--explore", type=float, default=0.3)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--grpo-share", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report-keep", type=float, default=0.2, help="share of steps with gold 'report' kept for SFT")
    args = parser.parse_args()
    env = setup()
    rng = np.random.default_rng(args.seed)
    trace = Network("weak", args.seed).trace
    out = results_dir("tools_v3") / "select"
    out.mkdir(exist_ok=True)
    laya_dir = WORK / "data" / "tools_v3_laya" / "select"
    laya_dir.mkdir(parents=True, exist_ok=True)
    sft, grpo, laya = [], [], []
    for n, item in enumerate(env["val"]):
        for budget in args.budgets:
            ep = episode(env, item, budget, trace[rng.integers(len(trace))], train=True)
            for step in range(args.max_steps):
                actions = ep.candidates()
                values = ep.values(actions)
                gold = min(actions, key=values.get)
                text = prompt(ep, actions)
                uid = f"{item['item_id']}-{budget}-{step}"
                if gold != "report" or rng.random() < args.report_keep:
                    sft.append({"prompt": text, "completion": " " + gold})
                    laya.append(laya_record(ep, actions, values, gold, uid, args.temperature))
                if rng.random() < args.grpo_share:
                    grpo.append({"prompt": text, "values": json.dumps(values)})
                move = actions[rng.integers(len(actions))] if rng.random() < args.explore else gold
                ep.step(move)
                if ep.done:
                    break
        if (n + 1) % 1000 == 0:
            print(f"[select data] {n + 1}/{len(env['val'])} val items, {len(sft)} steps", flush=True)
    order = rng.permutation(len(laya))
    split = int(0.95 * len(order))
    for name, rows in (("sft_qwen", sft), ("grpo_qwen", grpo)):
        with (out / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    for name, idx in (("train", order[:split]), ("calib", order[split:])):
        with (laya_dir / f"{name}.jsonl").open("w") as f:
            f.writelines(json.dumps(laya[k]) + "\n" for k in idx)
    golds = [r["completion"].strip() for r in sft]
    print(f"[select data] SFT steps {len(sft)}, GRPO prompts {len(grpo)}, laya records {len(laya)}", flush=True)
    print("[select data] gold action counts:", json.dumps({a: golds.count(a) for a in sorted(set(golds), key=golds.count, reverse=True)}), flush=True)


if __name__ == "__main__":
    main()
