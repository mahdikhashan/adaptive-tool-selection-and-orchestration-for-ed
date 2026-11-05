from fedorch.costs import COST, onboard_s, server_s, uplink_s

ONBOARD = {"classifier", "smolvlm_local"}
RISK = {"miss": 10.0, "false_alarm": 1.0, "abstain": 0.5}


class Drone:
    def __init__(self, trace):
        self.trace = trace
        self.onboard_free = 0.0
        self.uplink_free = 0.0

    def mbps(self, t):
        return self.trace[int(t) % len(self.trace)]

    def run(self, tool, row, clock):
        if tool in ONBOARD:
            start = max(clock, self.onboard_free)
            self.onboard_free = start + onboard_s(tool, row)
            return self.onboard_free, True, 0
        ready = clock
        if tool == "smolvlm_split":
            ready = max(clock, self.onboard_free) + onboard_s(tool, row)
            self.onboard_free = ready
        send = max(ready, self.uplink_free)
        transfer = uplink_s(row, self.mbps(send))
        if transfer is None:
            self.uplink_free = send + COST["timeout_s"]
            return self.uplink_free, False, 0
        self.uplink_free = send + transfer
        return self.uplink_free + server_s(tool, row), True, row["payload_bytes"]


def run_mission(items, cache, trace, policy):
    drone, records = Drone(trace), []
    for n, item in enumerate(items):
        arrival = n * COST["frame_interval_s"]
        clock, history, answer, sent = arrival, [], None, 0
        while True:
            state = {"mbps": drone.mbps(clock - 1), "backlog_s": max(0.0, drone.onboard_free - clock)}
            tool = policy(item, history, state)
            if tool is None:
                break
            row = cache[item["item_id"]][tool]
            clock, ok, payload = drone.run(tool, row, clock)
            sent += payload
            history.append((tool, row, ok))
            if ok:
                answer = row["prediction"]
        late = clock - arrival > COST["deadline_s"]
        records.append({"item": item, "answer": None if late else answer, "late": late, "latency_s": clock - arrival,
                        "tools": [h[0] for h in history], "failed": sum(not h[2] for h in history), "uplink_bytes": sent})
    return records


def metrics(records):
    n = len(records)
    defects = [r for r in records if r["item"]["label"] != "good"]
    goods = [r for r in records if r["item"]["label"] == "good"]
    miss = sum(r["answer"] in (None, "good") for r in defects)
    false_alarm = sum(r["answer"] not in (None, "good", "cannot-judge") for r in goods)
    abstain = sum(r["answer"] in (None, "cannot-judge") for r in goods) + sum(r["answer"] == "cannot-judge" for r in defects)
    latencies = sorted(r["latency_s"] for r in records)
    return {
        "items": n,
        "accuracy": sum(r["answer"] == r["item"]["label"] for r in records) / n,
        "defect_recall": sum(r["answer"] == r["item"]["label"] for r in defects) / len(defects),
        "miss_rate": miss / len(defects),
        "false_alarm_rate": false_alarm / len(goods),
        "abstain_rate": abstain / n,
        "late_rate": sum(r["late"] for r in records) / n,
        "risk_per_1000": 1000 * (RISK["miss"] * miss + RISK["false_alarm"] * false_alarm + RISK["abstain"] * abstain) / n,
        "escalation_rate": sum(r["tools"] != ["classifier"] for r in records) / n,
        "cloud_failures_per_1000": 1000 * sum(r["failed"] for r in records) / n,
        "uplink_mb_per_1000": sum(r["uplink_bytes"] for r in records) / n / 1000,
        "median_latency_s": latencies[n // 2],
        "p95_latency_s": latencies[int(0.95 * n) - 1],
    }
