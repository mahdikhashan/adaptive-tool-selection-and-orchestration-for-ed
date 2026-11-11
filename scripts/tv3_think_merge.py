import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("results/tools_v3/select")
BANNED = re.compile(r"\b(gold|ground truth|true label|correct answer|training)\b", re.I)


def check(item, reason):
    words = len(reason.split())
    answer = item["answer"]
    tools = {c.split(" ")[0] for c in item["candidates"]}
    named = {t for t in set(re.findall(r"\b[a-z][a-z0-9_\-]*[a-z0-9]\b", reason.lower())) & set(TOOLS) if "_" in t}
    if answer["name"] == "SELECT" and answer["ids"][0] not in tools:
        return "gold tool not feasible"
    if not 60 <= words <= 190:
        return "length"
    if BANNED.search(reason):
        return "leak"
    if "{" in reason:
        return "json"
    if named - tools - set(re.findall(r"^(\S+):", item["state"], re.M)):
        return "unknown tool"
    if answer["name"] == "SELECT" and answer["ids"][0] not in reason.lower():
        return "tool not named"
    if answer["name"] == "REPORT" and not (answer["component"].split()[0] in reason.lower() and answer["condition"].split(",")[0].split()[0] in reason.lower()):
        return "report not stated"
    return None


def main():
    parser = argparse.ArgumentParser(description="Check the teacher reasons and write the thinking SFT data.")
    parser.parse_args()
    items = {}
    for f in sorted(glob.glob(str(ROOT / "think" / "batch_*.json"))):
        for it in json.load(open(f))["items"]:
            items[it["i"]] = it
    reasons, bad = {}, Counter()
    for f in sorted(glob.glob(str(ROOT / "think" / "out_*.jsonl"))):
        for line in open(f):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad["bad json"] += 1
                continue
            problem = check(items[r["i"]], r["reason"]) if r.get("i") in items else "unknown id"
            if problem:
                bad[problem] += 1
            else:
                reasons[r["i"]] = " ".join(r["reason"].split())
    out = []
    for i, line in enumerate(open(ROOT / "sft_qwen.jsonl")):
        if i in reasons:
            r = json.loads(line)
            prompt = r["prompt"].replace("<|im_start|>assistant\n<think>\n\n</think>\n\n", "<|im_start|>assistant\n")
            out.append({"prompt": prompt, "completion": f"<think>\n{reasons[i]}\n</think>\n\n{r['completion']}"})
    with open(ROOT / "sft_qwen_think.jsonl", "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in out)
    print(f"items {len(items)}, kept {len(out)}, rejected {sum(bad.values())}:", dict(bad))


TOOLS = json.loads(Path("results/tools_v3/tools.json").read_text())

if __name__ == "__main__":
    main()
