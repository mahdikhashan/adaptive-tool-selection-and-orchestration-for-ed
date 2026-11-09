import functools
import random

from fedorch.items import DATA

REGIMES = {
    "good": {"scale": 1.0, "outages_per_hour": 0, "outage_s": 0},
    "weak": {"scale": 0.1, "outages_per_hour": 6, "outage_s": 20},
    "poor": {"scale": 0.02, "outages_per_hour": 20, "outage_s": 30},
}


@functools.cache
def lte_mbps():
    series = []
    for log in sorted((DATA / "lte4g").glob("**/*.log")):
        for line in log.read_text().splitlines():
            cols = line.split()
            if len(cols) == 6 and int(cols[5]) > 0:
                series.append(int(cols[4]) * 8 / int(cols[5]) / 1000)
    return tuple(series)


def bandwidth_trace(regime, seconds, seed, base=None):
    base = base or lte_mbps()
    rng = random.Random(seed)
    start = rng.randrange(len(base))
    params = REGIMES[regime]
    trace = [base[(start + t) % len(base)] * params["scale"] for t in range(seconds)]
    outage_count = round(params["outages_per_hour"] * seconds / 3600)
    for _ in range(outage_count):
        begin = rng.randrange(seconds)
        for t in range(begin, min(seconds, begin + params["outage_s"])):
            trace[t] = 0.0
    return trace
