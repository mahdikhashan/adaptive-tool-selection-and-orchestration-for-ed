import collections

K = 3
MODELS = [f"fold{f}" for f in range(K)] + ["full"]


def tower_buckets(items, k):
    sizes = collections.Counter(item["tower"] for item in items)
    load, bucket = [0] * k, {}
    for tower, size in sorted(sizes.items(), key=lambda pair: (-pair[1], pair[0])):
        bucket[tower] = load.index(min(load))
        load[bucket[tower]] += size
    return bucket


def owner_of(items):
    bucket = tower_buckets([item for item in items if item["split"] != "test"], K)
    return {item["item_id"]: "full" if item["split"] == "test" else f"fold{bucket[item['tower']]}" for item in items}


def fit_and_cache(items, model):
    owner = owner_of(items)
    fit = [item for item in items if item["split"] != "test" and owner[item["item_id"]] != model]
    cache = [item for item in items if owner[item["item_id"]] == model]
    return fit, cache


def select_split(fit):
    bucket = tower_buckets(fit, 5)
    return [item for item in fit if bucket[item["tower"]] != 4], [item for item in fit if bucket[item["tower"]] == 4]
