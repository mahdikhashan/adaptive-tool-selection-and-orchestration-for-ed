import argparse
import json
import random
import re
from collections import defaultdict

from pathlib import Path

CANDIDATE = re.compile(r"\[([^\]]+)\]\n\S+ \((drone|cloud)\):.*?Time now: ([\d.]+) s\.")


def parts(prompt):
    user = prompt.split("<|im_start|>user\n", 1)[1].split("<|im_end|>", 1)[0]
    state = user.split("Sub-step: ", 1)[1].split("\n\nCandidates", 1)[0]
    candidates = [f"{t} ({loc}, {s} s)" for t, loc, s in CANDIDATE.findall(user)] + ["report"]
    return state, candidates


def action(completion):
    c = json.loads(completion)
    return c["ids"][0] if c["name"] == "SELECT" else "report"


def supported(state, completion):
    c = json.loads(completion)
    seen = state.split("Tool results so far:\n", 1)[1]
    return c["component"] in seen and c["condition"] in seen


def main():
    parser = argparse.ArgumentParser(description="Pick balanced SFT steps and write compact batch files for teacher reasons.")
    parser.add_argument("--total", type=int, default=4000)
    parser.add_argument("--cap", type=int, default=300)
    parser.add_argument("--reports", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path("results/tools_v3") / "select"
    by_action = defaultdict(list)
    for i, line in enumerate(open(root / "sft_qwen.jsonl")):
        r = json.loads(line)
        a = action(r["completion"])
        state, candidates = parts(r["prompt"])
        if a == "report" and not supported(state, r["completion"]):
            continue
        by_action[a].append({"i": i, "state": state, "candidates": candidates, "answer": json.loads(r["completion"])})
    rng = random.Random(args.seed)
    for v in by_action.values():
        rng.shuffle(v)
    pick = []
    for a, v in by_action.items():
        if a not in ("report", "laya"):
            pick += v[: args.cap]
    pick += by_action["report"][: args.reports]
    pick += by_action["laya"][: max(0, args.total - len(pick))]
    rng.shuffle(pick)
    tools = json.loads((Path("results/tools_v3") / "tools.json").read_text())
    glossary = {t: f"{v['location']}: {v['description']}" for t, v in tools.items()}
    out = root / "think"
    out.mkdir(exist_ok=True)
    for k in range(0, len(pick), args.batch):
        (out / f"batch_{k // args.batch:03d}.json").write_text(json.dumps({"tools": glossary, "items": pick[k:k + args.batch]}, indent=1))
    counts = defaultdict(int)
    for p in pick:
        counts[action(json.dumps(p["answer"]))] += 1
    print(f"{len(pick)} steps in {-(-len(pick) // args.batch)} batches; per action:", json.dumps(dict(sorted(counts.items(), key=lambda x: -x[1]))))


if __name__ == "__main__":
    main()
