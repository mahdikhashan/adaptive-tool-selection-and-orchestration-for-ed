import statistics

from fedorch.costs import COST
from fedorch.network import bandwidth_trace
from fedorch.simulate import metrics, run_mission


def mission(split, items):
    return sorted((i for i in items if i["split"] == split), key=lambda i: (i["tower"], i["parent_id"], i["item_id"]))


def trace_for(items, regime, seed, base):
    return bandwidth_trace(regime, int(len(items) * COST["frame_interval_s"]) + 60, seed, base)


def tune(make, grid, val, cache, regime, base):
    trace = trace_for(val, regime, 1000, base)
    scores = {value: metrics(run_mission(val, cache, trace, make(value)))["risk_per_1000"] for value in grid}
    return min(scores, key=scores.get), scores


def mean_over_seeds(runs):
    return {k: round(statistics.mean(r[k] for r in runs), 4) for k in runs[0]}


def evaluate(policy, test, cache, traces):
    return mean_over_seeds([metrics(run_mission(test, cache, trace, policy)) for trace in traces])
