import json

from fedorch.common import results_dir

TOOLS = ["classifier", "smolvlm_local", "smolvlm_split", "cloud"]


def load_cache(tools=TOOLS):
    cache = {}
    for tool in tools:
        with (results_dir("step2") / f"tool_{tool}.jsonl").open() as file:
            for row in map(json.loads, file):
                cache.setdefault(row["item_id"], {})[tool] = row
    return cache
